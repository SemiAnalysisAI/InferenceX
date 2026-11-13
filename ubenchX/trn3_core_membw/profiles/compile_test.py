import sys, os, shutil, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kernels import KERNELS
from nki.compiler.driver import compile_to_bir
from nki.compiler.frontend import ParserFrontend
from nki.compiler.ncc_driver import CompileOptions, compile_bir_to_neff
N, W = 4096, 16
rng = np.random.default_rng(0); perm = rng.permutation(N)
chain = np.zeros((N, W), np.int32); chain[perm, 0] = np.roll(perm, -1)
for name in sys.argv[1:]:
    out = f"/w/runs/proto/chase_{name}"; shutil.rmtree(out, ignore_errors=True); os.makedirs(out + "/build")
    opts = CompileOptions(target="trn3", lnc=1, artifacts_dir=out + "/build", output_path=out + "/kernel.neff")
    try:
        bir, res = compile_to_bir(KERNELS[name], frontend=ParserFrontend(), inputs={"chain": chain}, compile_opts=opts)
        compile_bir_to_neff(opts, bir, input_arrays=[chain], argument_names=["chain"], output_arg_names=getattr(res, "output_names", []))
        np.save(out + "/chain.npy", chain); print(name, "OK", os.path.getsize(out + "/kernel.neff"))
    except Exception as e:
        print(name, "FAIL", str(e)[-1500:])
