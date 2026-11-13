import sys, os, shutil, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kernels import KERNELS
from nki.compiler.driver import compile_to_bir
from nki.compiler.frontend import ParserFrontend
from nki.compiler.ncc_driver import CompileOptions, compile_bir_to_neff
x = np.arange(2 * 4 * 128 * 8192, dtype=np.float32).reshape(8, 128, 8192)
out = "/w/runs/proto/lnc2"; shutil.rmtree(out, ignore_errors=True); os.makedirs(out + "/build")
opts = CompileOptions(target="trn3", lnc=2, artifacts_dir=out + "/build", output_path=out + "/kernel.neff")
k = KERNELS["copy_h2h_static_spmd"]
try:
    bir, res = compile_to_bir(k, frontend=ParserFrontend(), inputs={"in_tensor": x}, compile_opts=opts)
    compile_bir_to_neff(opts, bir, input_arrays=[x], argument_names=["in_tensor"], output_arg_names=getattr(res, "output_names", []))
    np.save(out + "/input.npy", x); print("OK", os.path.getsize(out + "/kernel.neff"))
except Exception as e:
    print("FAIL", str(e)[-2500:])
