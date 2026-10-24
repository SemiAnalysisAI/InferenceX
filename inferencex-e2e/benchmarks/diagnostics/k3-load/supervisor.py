"""Bounded native D-worker loading repetitions in one exclusive allocation."""
import json,os,signal,subprocess,time,hashlib,threading
from pathlib import Path

ROOT=Path('/logs');ROOT.mkdir(exist_ok=True)
SETTINGS=json.loads(Path('/probe/settings.json').read_text())

def write(event,**kw):
    with (ROOT/'supervisor.jsonl').open('a') as f:
        f.write(json.dumps(dict(time=time.time(),event=event,**kw))+'\n')

def vram():
    result={}
    for p in Path('/sys/class/drm').glob('card[0-9]*/device/mem_info_vram_used'):
        if p.parent.parent.name.removeprefix('card').isdigit():
            result[str(p.parent.resolve())]=int(p.read_text())
    return result

def terminate(p):
    # The process group belongs exclusively to this diagnostic server launch.
    try:os.killpg(p.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:p.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:os.killpg(p.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        p.wait(timeout=15)

SAMPLER_STOP=threading.Event()
def sysfs_sampler():
    totals={k:int((Path(k)/'mem_info_vram_total').read_text()) for k in vram()}
    with (ROOT/'sysfs-vram.jsonl').open('a',buffering=1) as stream:
        while not SAMPLER_STOP.is_set():
            used=vram()
            stream.write(json.dumps(dict(time=time.time(),used=used,free={k:totals[k]-v for k,v in used.items()}))+'\n')
            SAMPLER_STOP.wait(.1)
sampler_thread=threading.Thread(target=sysfs_sampler,daemon=True);sampler_thread.start()
start=time.monotonic(); overall='no_reproduction'
before=vram();time.sleep(10);baseline=vram();write('allocation_start',job=os.environ.get('SLURM_JOB_ID'),host=os.uname().nodename,baseline=baseline,command=SETTINGS['command'],configuration=SETTINGS)
assert len(baseline)==8,baseline
limits={k:int((Path(k)/'mem_info_vram_total').read_text())*.10 for k in baseline}
write('preflight',before=before,after=baseline,limits=limits,policy='Original formal host gate: max VRAM10%, after10s')
if any(baseline[k]>limits[k] for k in baseline):
    write('preexisting_vram_abort',vram=baseline)
    (ROOT/'result.json').write_text(json.dumps(dict(result='preflight_declined',native_loads=0)))
    raise SystemExit(2)
for number in range(1,int(SETTINGS.get('attempts_max',4))+1):
    is_stress=(number==SETTINGS.get('timing_stress_attempt',4))
    if time.monotonic()-start>1260:
        write('budget_stop');break
    out=ROOT/f'attempt{number}';out.mkdir(exist_ok=False)
    env=dict(os.environ,**SETTINGS['env']);env.update(SA_LOAD_PROBE_DIR=str(out),PYTHONPATH='/probe'+(':'+env['PYTHONPATH'] if env.get('PYTHONPATH') else ''))
    if is_stress:env['SA_LOAD_PROBE_CLOSE_DELAY_S']='0.25'
    env['SA_LOAD_PROBE_MODE']=SETTINGS.get('observation_mode','full')
    env['HEAD_NODE_IP']=subprocess.check_output(['hostname','-I'],text=True).split()[0]
    assert 'VLLM_FASTSAFETENSORS_QUEUE_SIZE' not in env
    with (out/'server.out').open('w') as f:
        p=subprocess.Popen(SETTINGS['command'],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        write('attempt_started',attempt=number,pid=p.pid,timing_perturbation=is_stress)
        deadline=time.monotonic()+360;status='timeout'
        try:
            while time.monotonic()<deadline:
                if p.poll() is not None:
                    status='server_exited';break
                if len(list(out.glob('rank*.loaded')))==8:
                    status='all_eight_loaded'
                    # Allow native watchdogs to report an asynchronous load fault.
                    for _ in range(20):
                        time.sleep(.5)
                        if p.poll() is not None:
                            status='server_exited';break
                    break
                time.sleep(.5)
        finally:
            terminate(p)
    text=(out/'server.out').read_text(errors='replace')
    hsa='HSA_STATUS_ERROR_OUT_OF_RESOURCES' in text
    copy_error='cudaMemcpy(' in text and 'failed, err=' in text
    write('attempt_finished',attempt=number,status=status,returncode=p.returncode,hsa=hsa,copy_error=copy_error,loaded_ranks=[p.name for p in out.glob('rank*.loaded')])
    released=False
    for _ in range(30):
        current=vram()
        if all(current[k]<=baseline[k]+512*1024**2 for k in baseline):
            released=True;break
        time.sleep(1)
    write('release_checked',attempt=number,released=released,vram=current)
    if hsa or copy_error:
        overall='reproduced_native_loading_failure' if not is_stress else 'reproduced_timing_stress_loading_failure'
        break
    if not released:
        overall='own_server_vram_not_released';break
    if status!='all_eight_loaded':
        overall='diagnostic_incomplete';break
write('finished',result=overall,elapsed=time.monotonic()-start,final_vram=vram())
(ROOT/'result.json').write_text(json.dumps(dict(result=overall,elapsed=time.monotonic()-start),indent=2))
# Neither unreproduced nor reproduced is a serving/performance PASS.

SAMPLER_STOP.set();sampler_thread.join(timeout=2)
