import argparse
from pathlib import Path
import shutil
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('recipe', type=Path)
parser.add_argument('--workspace', type=Path, required=True)
parser.add_argument('--expected-mib', type=int, required=True)
parser.add_argument('--delay-seconds', type=int, required=True)
args = parser.parse_args()
workspace = args.workspace.resolve()
recipe = args.recipe.resolve()
recipe.relative_to(workspace)
probe = workspace / '.dram-oom-probe'
probe.mkdir(exist_ok=True)
shutil.copyfile(Path(__file__).with_name('sitecustomize.py'), probe / 'sitecustomize.py')
original = recipe.read_bytes()
recipe.with_suffix(recipe.suffix + '.oom-original').write_bytes(original)
item = yaml.safe_load(original)
base = item.get('base', item)
engine = base.get('engine')
if (engine.get('type') if isinstance(engine, dict) else engine) != 'sglang':
    raise ValueError('OOM overlay supports only SGLang')
role = base['roles']['agg']
if set(base['roles']) != {'agg'}:
    raise ValueError('OOM overlay requires one aggregate worker role')
env = role.setdefault('env', {})
prior = env.get('PYTHONPATH', '')
env['PYTHONPATH'] = '/infmax-workspace/.dram-oom-probe:/infmax-workspace' + (':' + prior if prior else '')
env.update(INFX_OOM_PROBE='1', INFX_OOM_EXPECTED_MIB=str(args.expected_mib), INFX_OOM_DELAY_S=str(args.delay_seconds), INFX_OOM_OUTPUT_DIR='/logs/oom-proof')
recipe.write_text(yaml.safe_dump(item, sort_keys=False))
print(f'Applied role-only OOM overlay to {recipe.relative_to(workspace)}; expected cap {args.expected_mib}MiB, delay {args.delay_seconds}s')
