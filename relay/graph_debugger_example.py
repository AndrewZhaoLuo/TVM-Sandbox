import json

import tvm
from tvm import relay
from tvm.contrib.debugger import debug_executor

target = "llvm"

# Assume you have mod and params
mod = None
params = None


def run_tvm(lib):
    dev = tvm.device(target, 0)
    rt_mod = debug_executor.GraphModuleDebug(
        lib["debug_create"]("default", dev),
        [dev],
        lib.get_graph_json(),
        "./profiling_data/",
    )

    # Set inputs here
    # rt_mod.set_input("input_name", numpy_arr)

    rt_mod.run()
    tvm_res = rt_mod.get_output(0)
    return tvm_res, rt_mod


# Build graph executor library, don't fuse operators!
with tvm.transform.PassContext(opt_level=3, config={"relay.FuseOps.max_depth": 1}):
    lib = relay.build(mod, target=target, params=params)

tvm_res, rt_mod = run_tvm(lib)

print(tvm_res)

# Load graph trace + intermediate tensors
with open("profiling_data/_tvmdbg_device_CPU_0/output_tensors.params", "rb") as f:
    intermediate_tensors = dict(tvm.relay.load_param_dict(f.read()))

with open("profiling_data/_tvmdbg_device_CPU_0/_tvmdbg_graph_dump.json", "r") as f:
    graph = json.load(f)
