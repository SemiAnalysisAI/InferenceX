#!/usr/bin/env python3
"""Prepare/build the reviewed candidate in a supplied existing Linux build runtime.
No download, image creation, installation, GPU access or runtime mutation.
"""
import argparse, difflib, hashlib, json, os, platform, re, shutil, subprocess, sys, tarfile, zipfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
SEALED=HERE/'dynamo-c3e05f-full'
SUFFIX='.dev20260909'
FEATURES='kv-indexer,slot-tracker,select-service,mm-routing,aic-forward-pass,request-trace-s3'
ARCHIVE_SHA='20f4b3e05e8b0bc56095471960b5f7490736465243dea00471e31d1985d008fc'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise SystemExit(message)
def run(args,**kw):subprocess.run(args,check=True,**kw)

def verify():
    records=json.loads((HERE/'integrated-candidate-file-hashes.json').read_text())
    require(len(records)==15,'Expected the 15 reviewed source/lock identities')
    for item in records:
        root=SEALED if item['tree']=='Dynamo' else HERE/'sglang-008403017-candidate'
        require(sha(root/item['path'])==item.get('candidate_sha256',item.get('sha256')),f"Candidate mismatch: {item['path']}")
    require(sha(HERE/'dynamo-c3e05f-full.tar.gz')==ARCHIVE_SHA,'Original source archive mismatch')
    overlays={x['path'] for x in records if x['tree']=='Dynamo' and 'candidate_sha256' in x}
    # Every untouched archived input must remain exact, not only the changed files.
    with tarfile.open(HERE/'dynamo-c3e05f-full.tar.gz') as archive:
        for item in archive:
            if not item.isfile():continue
            rel='/'.join(item.name.split('/')[1:])
            if rel in overlays:continue
            require((SEALED/rel).read_bytes()==archive.extractfile(item).read(),f'Unreviewed source modification: {rel}')
    return records

def prepare(work):
    records=verify()
    require(not work.exists(),'Preparation destination must be absent; reuse an existing prepared tree with build')
    shutil.copytree(SEALED,work,ignore=shutil.ignore_patterns('target','__pycache__','.pytest_cache'))
    # Reproduce nightly packaging only on this derived copy. Original15 hashes stay sealed.
    before={str(f.relative_to(work)):f.read_bytes() for f in work.rglob('*.toml')}
    before.update({str(f.relative_to(work)):f.read_bytes() for f in [work/'Cargo.lock',work/'lib/bindings/python/Cargo.lock']})
    run([sys.executable,str(work/'.github/scripts/apply_dev_version.py'),SUFFIX,str(work)])
    for rel in ['Cargo.lock','lib/bindings/python/Cargo.lock']:
        path=work/rel;text=path.read_text();blocks=text.split('[[package]]');names=[]
        for i,block in enumerate(blocks[1:],1):
            name=re.search(r'^name = "([^"]+)"',block,re.M)
            version=re.search(r'^version = "([^"]+)"',block,re.M)
            if name and version and version.group(1)=='1.5.0' and not re.search(r'^source = ',block,re.M):
                require((name.group(1).startswith(('dynamo-','kvbm-')) or name.group(1)=='libdynamo_llm'),'Unexpected local package version')
                names.append(name.group(1));blocks[i]=block.replace('version = "1.5.0"','version = "1.5.0-dev20260909"',1)
        derived='[[package]]'.join(blocks)
        for name in names:derived=derived.replace(f'"{name} 1.5.0"',f'"{name} 1.5.0-dev20260909"')
        # Only local package versions/dependency references are transformed; no resolution/update.
        path.write_text(derived)
    changes=[];patch=[]
    for rel,old in before.items():
        new=(work/rel).read_bytes()
        if old==new:continue
        changes.append({'path':rel,'base_sha256':hashlib.sha256(old).hexdigest(),'derived_sha256':hashlib.sha256(new).hexdigest()})
        patch.extend(difflib.unified_diff(old.decode().splitlines(True),new.decode().splitlines(True),fromfile='a/'+rel,tofile='b/'+rel))
    (work/'nightly-version-only.patch').write_text(''.join(patch))
    identity={'source':'c3e05f0244ae6264d7953f68e2499c6dc2f54723','python_version':'1.5.0.dev20260909','cargo_version':'1.5.0-dev20260909','reviewed_files':records,'derived_packaging':changes}
    (work/'containment-build-inputs.json').write_text(json.dumps(identity,indent=2)+'\n')
    verify()
    print(json.dumps({'prepared':str(work),'version_changes':len(changes),'sealed_originals_unchanged':True}))

def build(work,out,cargo_cache,target_cache):
    verify()
    require(platform.system()=='Linux' and platform.machine()=='x86_64','Use existing Linux x86_64 build runtime; no cross-platform substitution')
    require(sys.version_info[:3]==(3,12,3),'Expected original Python3.12.3 interpreter')
    require(cargo_cache.is_dir() and target_cache.is_dir(),'Supply existing compatible Cargo and target caches')
    for tool in ['cargo','rustc','maturin','uv','protoc','patchelf','readelf','cc','cmake','pkg-config']:
        require(shutil.which(tool),f'Missing existing build prerequisite: {tool}; do not auto-install')
    require(subprocess.check_output(['rustc','--version'],text=True).split()[1]=='1.96.1','Exact release Rust1.96.1 required')
    require(subprocess.check_output(['maturin','--version'],text=True).split()[1]=='1.15.0','Exact original wheel builder maturin1.15.0 required')
    require(not os.environ.get('RUSTFLAGS') and not os.environ.get('CARGO_ENCODED_RUSTFLAGS'),'Remove ambient Rust flag overrides; exact repository config owns compiler flags')
    identity=json.loads((work/'containment-build-inputs.json').read_text())
    derived={x['path']:x['derived_sha256'] for x in identity['derived_packaging']}
    for item in identity['reviewed_files']:
        if item['tree']!='Dynamo':continue
        require(sha(work/item['path'])==derived.get(item['path'],item.get('candidate_sha256',item.get('sha256'))),f"Prepared input drift: {item['path']}")
    for rel,digest in derived.items():require(sha(work/rel)==digest,f'Packaging drift: {rel}')
    with tarfile.open(HERE/'dynamo-c3e05f-full.tar.gz') as archive:
        for item in archive:
            if not item.isfile():continue
            rel='/'.join(item.name.split('/')[1:])
            require(sha(work/rel)==derived.get(rel,sha(SEALED/rel)),f'Prepared source drift: {rel}')
    require(not out.exists() or not any(out.iterdir()),'Wheel output must be empty to avoid stale-artifact selection')
    out.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,CARGO_HOME=str(cargo_cache),CARGO_TARGET_DIR=str(target_cache),CARGO_NET_OFFLINE='true',UV_OFFLINE='true')
    run(['uv','build','--offline','--no-build-isolation','--wheel','--out-dir',str(out)],cwd=work,env=env)
    run(['maturin','build','--release','--locked','--offline','--target','x86_64-unknown-linux-gnu','--manylinux','2_39','--features',FEATURES,'--interpreter',sys.executable,'--out',str(out)],cwd=work/'lib/bindings/python',env=env)
    wheels=sorted(out.glob('*.whl'));require(len(wheels)==2,'Expected exact Python and runtime wheel pair')
    outputs=[]
    for wheel in wheels:
        require('1.5.0.dev20260909-' in wheel.name,'Wrong nightly metadata version')
        with zipfile.ZipFile(wheel) as archive:
            metadata=archive.read(next(x for x in archive.namelist() if x.endswith('.dist-info/METADATA'))).decode()
            require('\nVersion: 1.5.0.dev20260909\n' in metadata,'Wrong wheel metadata')
            if wheel.name.startswith('ai_dynamo_runtime-'):
                require('cp310-abi3-' in wheel.name and 'manylinux_2_39_x86_64' in wheel.name,'Wrong ABI/platform tag')
                member='dynamo/_core.abi3.so';binary=out/'_core.abi3.so';binary.write_bytes(archive.read(member))
                needed=subprocess.check_output(['readelf','-d',str(binary)],text=True)
                require(not re.search(r'lib(?:avcodec|avformat|avutil|swscale|swresample|avfilter|avdevice)',needed),'Unexpected media dependency')
                (out/'elf-dynamic.txt').write_text(needed)
        outputs.append({'wheel':wheel.name,'sha256':sha(wheel)})
    # --locked must leave every derived lock byte intact.
    for rel in ['Cargo.lock','lib/bindings/python/Cargo.lock']:require(sha(work/rel)==derived[rel],f'Lockfile changed during build: {rel}')
    (out/'build-output-receipt.json').write_text(json.dumps({'inputs':identity,'features':FEATURES,'manylinux_policy':'manylinux_2_39','portability_limit':'Exact Ubuntu24.04 runtime image; not original manylinux2_28 portability','wheels':outputs,'runtime_installed':False},indent=2)+'\n')
    print(json.dumps({'wheels':outputs,'runtime_installed':False}))

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('mode',choices=['verify','prepare','build']);ap.add_argument('--work-dir',type=Path);ap.add_argument('--out',type=Path);ap.add_argument('--cargo-cache',type=Path);ap.add_argument('--target-cache',type=Path);a=ap.parse_args()
    if a.mode=='verify':verify();print('15 reviewed identities and all original archived inputs verified')
    else:
        require(a.work_dir is not None,'--work-dir required');work=a.work_dir.resolve()
        if a.mode=='prepare':prepare(work)
        else:
            require(all([a.out,a.cargo_cache,a.target_cache]),'--out, --cargo-cache and --target-cache required')
            build(work,a.out.resolve(),a.cargo_cache.resolve(),a.target_cache.resolve())
