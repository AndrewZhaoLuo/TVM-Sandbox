"""Should work after https://github.com/apache/tvm/pull/11000 merges. 

Benchmarks individual nodes in the graph runtime.
"""

import numpy as np
import onnx
import tvm
from tvm import relay
from tvm.contrib.debugger import debug_executor

onnx_model = onnx.load("models/shufflenet-9.onnx")

# Build the model, get a VMExecutor.
mod, params = relay.frontend.from_onnx(
    onnx_model,
    freeze_params=True,
)
target = "llvm"

with tvm.transform.PassContext(opt_level=3, config={"relay.FuseOps.max_depth": 1}):
    lib = relay.build(mod, target=target, params=params)

dev = tvm.device(target, 0)
rt_mod = debug_executor.GraphModuleDebug(
    lib["debug_create"]("default", dev),
    [dev],
    lib.get_graph_json(),
    dump_root="profiling_data/",
)

# Set inputs here, examine model/look at info in config
input_dict = {"gpu_0/data_0": np.zeros((1, 3, 224, 224), dtype="float32")}
rt_mod.set_input(**input_dict)

node_name_to_out_shapes = {}
for node in rt_mod.debug_datum.get_graph_nodes():
    node_name_to_out_shapes[node["name"]] = node["shape"]

for i, node in enumerate(rt_mod.debug_datum.get_graph_nodes()):
    result = rt_mod.run_individual_node(i, 10, 10, min_repeat_ms=100)

    node_name = node["name"]
    output_shape = str(node["shape"])

    input_shapes = []
    for input_node_name in node.get("inputs", []):
        input_shapes.append(node_name_to_out_shapes[input_node_name])
    input_shapes = str(input_shapes)

    print(
        f"{i: <4} {node_name: <60} out: {output_shape: <20} in: {input_shapes: <60}"
        f"-- {result.mean:10.9f} ({result.std:10.9f}). Median: ({result.median:10.9f}), "
        f"Range: [{result.min:10.9f}, {result.max:10.9f}]"
    )
