"""Which replica groups does the runtime build for an SBUF all_gather? One compile + one launch each."""
import os, sys, json, subprocess
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_c2c as R
import kernels as K
out, cores = sys.argv[1], sys.argv[2]
specs = {}
for a in range(8):
    for b in range(a + 1, 8):
        specs[f"pair{a}{b}"] = [[a, b]] + [[k] for k in range(8) if k not in (a, b)]
specs["all8"] = [list(range(8))]
specs["q0123"] = [[0, 1, 2, 3], [4, 5, 6, 7]]
specs["q0145"] = [[0, 1, 4, 5], [2, 3, 6, 7]]
specs["q0246"] = [[0, 2, 4, 6], [1, 3, 5, 7]]
specs["m1"] = R.pairs_for(1)
res = {}
for name, rg in specs.items():
    d = f"{out}/{name}"
    os.makedirs(d, exist_ok=True)
    K.REPLICA = rg
    R.pairs_for_orig = R.pairs_for
    from nki.compiler.driver import compile_to_bir
    from nki.compiler.frontend import ParserFrontend
    from nki.compiler.ncc_driver import CompileOptions, compile_bir_to_neff
    K.NSTEP = 2
    x = R.rank_input(0, 1)
    try:
        opts = CompileOptions(target="trn3", lnc=1, artifacts_dir=d + "/build", output_path=d + "/kernel.neff")
        bir, r_ = compile_to_bir(K.xchg_ag, frontend=ParserFrontend(), inputs={"in_tensor": x}, compile_opts=opts)
        compile_bir_to_neff(opts, bir, input_arrays=[x], argument_names=["in_tensor"],
                            output_arg_names=getattr(r_, "output_names", []))
    except Exception as e:
        res[name] = "compile: " + str(e).strip().splitlines()[-1][:150]; print(name, res[name], flush=True); continue
    env = dict(os.environ, NEURON_LOGICAL_NC_CONFIG="1", NEURON_RT_VISIBLE_CORES=cores)
    p = subprocess.run(["neuron-explorer", "capture", "-n", d + "/kernel.neff", "-s", d + "/p.ntff", "-r", "8",
                        "--num-exec=1", "in_tensor", "/dev/null"][:-2], capture_output=True, text=True, env=env, cwd=d, timeout=300)
    txt = p.stdout + p.stderr
    reason = [l.split("reason:")[1].strip()[:80] for l in txt.splitlines() if "reason:" in l]
    res[name] = "ok" if p.returncode == 0 else ("fail: " + (reason[0] if reason else txt.strip().splitlines()[-1][:120]))
    print(name, rg if len(rg) < 3 else rg[0], res[name], flush=True)
json.dump(res, open(out + "/groups.json", "w"), indent=1)
