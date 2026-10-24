"""Observe only completion of native Worker.load_model; perform no CUDA queries."""
import functools
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(os.environ['SA_LOAD_PROBE_DIR'])

class Loader(importlib.abc.Loader):
    def __init__(self,base):self.base=base
    def create_module(self,spec):return self.base.create_module(spec) if hasattr(self.base,'create_module') else None
    def exec_module(self,module):
        self.base.exec_module(module)
        original=module.Worker.load_model
        source=Path(module.__file__).read_bytes()
        @functools.wraps(original)
        def load(worker,*args,**kwargs):
            start=time.time()
            original(worker,*args,**kwargs)
            row=dict(event='worker_load_complete',time=time.time(),started=start,pid=os.getpid(),rank=worker.rank,source_sha256=hashlib.sha256(source).hexdigest(),mode='boundary_only_no_cuda_queries')
            ROOT.mkdir(exist_ok=True)
            (ROOT/f'completion-rank{worker.rank}.json').write_text(json.dumps(row,indent=2))
            (ROOT/f'rank{worker.rank}.loaded').write_text(str(os.getpid()))
            # The sole behavioral change is this successful-load boundary.
            while True:time.sleep(1)
        module.Worker.load_model=load
    def __getattr__(self,name):return getattr(self.base,name)

class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname!='vllm.v1.worker.gpu_worker':return None
        spec=importlib.machinery.PathFinder.find_spec(fullname,path,target)
        if spec and spec.loader:spec.loader=Loader(spec.loader)
        return spec

def install():sys.meta_path.insert(0,Finder())
