import functools,hashlib,importlib.abc,importlib.machinery,json,os,sys,threading,time
from pathlib import Path
ROOT=Path(os.environ['SA_LOAD_PROBE_DIR']);ROOT.mkdir(parents=True,exist_ok=True)
LOCK=threading.Lock();STOP=threading.Event();RANK=-1

def emit(event,memory=True,**fields):
    row=dict(time=time.time(),pid=os.getpid(),rank=RANK,thread=threading.current_thread().name,event=event,**fields)
    if memory:
        try:
            import torch
            if torch.cuda.is_initialized():
                dev=torch.cuda.current_device();free,total=torch.cuda.mem_get_info(dev);stats=torch.cuda.memory_stats(dev)
                row.update(device=dev,free=free,total=total)
                for k in ('allocated_bytes.all.current','reserved_bytes.all.current','allocated_bytes.all.peak','reserved_bytes.all.peak','inactive_split_bytes.all.current','num_alloc_retries','num_ooms','num_device_free','num_oom_rejections','num_sync_all_streams'):row[k]=stats.get(k)
                row['driver_used_minus_torch_reserved']=total-free-stats.get('reserved_bytes.all.current',0)
        except Exception as exc:row['memory_error']=repr(exc)
    data=(json.dumps(row,default=str)+'\n').encode()
    with LOCK:
        fd=os.open(ROOT/f'events-pid{os.getpid()}.jsonl',os.O_APPEND|os.O_CREAT|os.O_WRONLY,0o644)
        try:os.write(fd,data)
        finally:os.close(fd)

def describe(value):
    if isinstance(value,dict):return {str(k):describe(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [describe(v) for v in value]
    if isinstance(value,(str,Path)):
        try:
            p=Path(value)
            if p.is_file():return dict(path=str(p),bytes=p.stat().st_size)
        except OSError:pass
        return str(value)
    return str(value)

def wrap(cls,name,details=False):
    original=getattr(cls,name)
    @functools.wraps(original)
    def call(self,*args,**kwargs):
        info=dict(cls=cls.__name__,method=name,object_id=id(self))
        if details:info.update(args=describe(args),kwargs=describe(kwargs))
        emit('method_enter',**info)
        delay=float(os.environ.get('SA_LOAD_PROBE_CLOSE_DELAY_S','0'))
        if cls.__name__=='FilesBufferOnDevice' and name=='close' and delay:
            emit('timing_perturbation',delay_seconds=delay,**info)
            time.sleep(delay)
        try:result=original(self,*args,**kwargs)
        except BaseException as exc:
            emit('method_error',error=repr(exc),**info);raise
        emit('method_exit',**info)
        return result
    setattr(cls,name,call)

def sampler():
    import torch
    torch.cuda.set_device(RANK)
    dumped=False
    while not STOP.wait(.1):
        emit('memory_sample')
        if RANK==4 and not dumped:
            try:
                free,_=torch.cuda.mem_get_info()
                if free<8*1024**3:
                    dumped=True
                    torch.cuda.memory._dump_snapshot(str(ROOT/'rank4-low-free.pickle'))
                    emit('low_free_snapshot',observed_free=free)
            except Exception as exc:
                dumped=True;emit('snapshot_unavailable',error=repr(exc))

def worker_hook(module):
    original=module.Worker.load_model
    @functools.wraps(original)
    def load(self,*args,**kwargs):
        global RANK
        RANK=self.rank
        emit('worker_load_enter',local_rank=self.local_rank,runner=type(self.model_runner).__module__+'.'+type(self.model_runner).__name__)
        import torch
        fraction=torch.cuda.memory.get_per_process_memory_fraction()
        getter=getattr(torch._C,'_accelerator_getAllocatorSettings',None)
        settings=getter() if getter else torch.cuda.memory._snapshot()['allocator_settings']
        emit('allocator_configuration',fraction=fraction,settings=settings,torch_version=torch.__version__)
        assert abs(fraction-.92)<1e-5, 'Allocator fraction did not activate; stop before loading'
        if RANK==4:
            try:
                torch.cuda.memory._record_memory_history(max_entries=50000,stacks='python');emit('history_enabled')
            except Exception as exc:emit('history_unavailable',error=repr(exc))
        threading.Thread(target=sampler,daemon=True,name='load-memory-sampler').start()
        try:original(self,*args,**kwargs)
        except BaseException as exc:
            emit('worker_load_error',error=repr(exc))
            if RANK==4:
                try:torch.cuda.memory._dump_snapshot(str(ROOT/'rank4-error.pickle'))
                except Exception as exc:emit('snapshot_unavailable',error=repr(exc))
            raise
        emit('worker_load_complete');STOP.set()
        if RANK==4:
            try:torch.cuda.memory._dump_snapshot(str(ROOT/'rank4-complete.pickle'))
            except Exception as exc:emit('snapshot_unavailable',error=repr(exc))
        (ROOT/f'rank{RANK}.loaded').write_text(str(os.getpid()))
        # Preserve native init/load; intentionally stop before profiling/KV init/graphs.
        while True:time.sleep(1)
    module.Worker.load_model=load

def iterator_hook(module):
    original=module.fastsafetensors_weights_iterator
    @functools.wraps(original)
    def iterator(files,*args,**kwargs):
        import torch
        getter=getattr(torch._C,'_accelerator_getAllocatorSettings',None)
        emit('iterator_start',files=describe(files),allocator_settings=getter() if getter else None)
        count=0;last=0
        try:
            for name,tensor in original(files,*args,**kwargs):
                count+=1;now=time.monotonic()
                if now-last>=.5:
                    emit('tensor_yield',name=name,count=count,shape=list(tensor.shape),dtype=str(tensor.dtype),tensor_bytes=tensor.numel()*tensor.element_size());last=now
                yield name,tensor
                del tensor
            emit('iterator_complete',count=count)
        except BaseException as exc:
            emit('iterator_error',error=repr(exc),count=count);raise
    module.fastsafetensors_weights_iterator=iterator

def patch(module):
    data=Path(module.__file__).read_bytes()
    emit('module_source',memory=False,module=module.__name__,path=module.__file__,sha256=hashlib.sha256(data).hexdigest())
    # Per-PID source evidence avoids concurrent writers; no package edits.
    (ROOT/(module.__name__.replace('.','_')+f'.pid{os.getpid()}.source.py')).write_bytes(data)
    n=module.__name__
    if n=='fastsafetensors.loader':
        for relative in ['frameworks/_torch.py','tensor_factory.py','copier/nogds.py']:
            source=Path(module.__file__).parent/relative
            if source.exists():
                (ROOT/(relative.replace('/','_')+f'.pid{os.getpid()}.source.py')).write_bytes(source.read_bytes())
    if n.endswith('gpu_worker'):worker_hook(module)
    elif n.endswith('weight_utils'):iterator_hook(module)
    elif n=='fastsafetensors.loader':
        for method in ('add_filenames','copy_files_to_device','reset','close'):wrap(module.BaseSafeTensorsFileLoader,method,details=method=='add_filenames')
    elif n=='fastsafetensors.parallel_loader':
        wrap(module.ParallelLoader,'__init__',True);wrap(module.PipelineParallel,'_load_single_batch',True)
    elif n=='fastsafetensors.file_buffer':wrap(module.FilesBufferOnDevice,'close')

TARGETS={'vllm.v1.worker.gpu_worker','vllm.model_executor.model_loader.weight_utils','fastsafetensors.loader','fastsafetensors.parallel_loader','fastsafetensors.file_buffer'}
class Loader(importlib.abc.Loader):
    def __init__(self,base):self.base=base
    def create_module(self,spec):return self.base.create_module(spec) if hasattr(self.base,'create_module') else None
    def exec_module(self,module):self.base.exec_module(module);patch(module)
    def __getattr__(self,name):return getattr(self.base,name)
class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname not in TARGETS:return None
        spec=importlib.machinery.PathFinder.find_spec(fullname,path,target)
        if spec and spec.loader:spec.loader=Loader(spec.loader)
        return spec

def install():
    sys.meta_path.insert(0,Finder());emit('observer_installed',memory=False,python=sys.version)
