from enum import auto

import tvm
import tvm.relay.testing
from tvm import auto_scheduler, relay
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner


def get_network(name, batch_size, layout="NCHW", dtype="float32"):
    """Get the symbol definition and random weight of a network"""

    # auto-scheduler prefers NHWC layout
    if layout == "NHWC":
        image_shape = (224, 224, 3)
    elif layout == "NCHW":
        image_shape = (3, 224, 224)
    else:
        raise ValueError("Invalid layout: " + layout)

    input_shape = (batch_size,) + image_shape
    output_shape = (batch_size, 1000)

    if name.startswith("resnet-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name.startswith("resnet3d-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "mobilenet":
        mod, params = relay.testing.mobilenet.get_workload(
            batch_size=batch_size, layout=layout, dtype=dtype, image_shape=image_shape
        )
    elif name == "squeezenet_v1.1":
        assert layout == "NCHW", "squeezenet_v1.1 only supports NCHW layout"
        mod, params = relay.testing.squeezenet.get_workload(
            version="1.1",
            batch_size=batch_size,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "inception_v3":
        input_shape = (
            (batch_size, 3, 299, 299) if layout == "NCHW" else (batch_size, 299, 299, 3)
        )
        mod, params = relay.testing.inception_v3.get_workload(
            batch_size=batch_size, dtype=dtype
        )
    elif name == "mxnet":
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model

        assert layout == "NCHW"

        block = get_model("resnet18_v1", pretrained=True)
        mod, params = relay.frontend.from_mxnet(
            block, shape={"data": input_shape}, dtype=dtype
        )
        net = mod["main"]
        net = relay.Function(
            net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
        )
        mod = tvm.IRModule.from_expr(net)

    return mod, params, input_shape, output_shape


# Define the neural network and compilation target
network = "resnet-18"
batch_size = 1
layout = "NCHW"
target = tvm.target.Target("llvm")
dtype = "float32"
use_dp_tuner = True

# Extract tasks from the network
print("Extract tasks...")
mod, params, input_shape, output_shape = get_network(
    network, batch_size, layout, dtype=dtype
)

with tvm.transform.PassContext(
    opt_level=3, config={"relay.backend.use_auto_scheduler": True}
):
    LOG_FILE = "resnet-18-NCHW-B1-llvm.json"
    BEST_LOG_FILE = "resnet-18-NCHW-B1-llvm-graph-tuned.json"
    target_op = [
        relay.op.get("nn.conv2d"),
    ]
    Tuner = DPTuner if use_dp_tuner else PBQPTuner

    executor = Tuner(mod, {"data": input_shape}, LOG_FILE, target_op, target)
    executor.benchmark_layout_transform(
        min_exec_num=1, runner=auto_scheduler.LocalRunner()
    )
    executor.run()
    executor.write_opt_sch2record_file(BEST_LOG_FILE)
