"""
Minimum example showing pass contexts are not preserved in tuning.
"""

import numpy as np
import tvm
import tvm.relay.testing
from tvm import auto_scheduler, relay
from tvm.contrib import graph_executor


def get_network(name, batch_size, layout="NHWC", dtype="float32"):
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
log_file = "%s-%s-B%d-%s.json" % (network, layout, batch_size, target.kind.name)

# Extract tasks from the network
print("Extract tasks...")
mod, params, input_shape, output_shape = get_network(
    network, batch_size, layout, dtype=dtype
)
tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

from tvm.ir.instrument import pass_instrument


@pass_instrument
class PrintStuff:
    def run_after_pass(self, mod, info):
        print("*" * 50)


def run_tuning():
    print("Begin tuning...")
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, timeout=10)

    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=30,  # change this to 20000 to achieve the best performance
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )

    with tvm.transform.PassContext(opt_level=3, instruments=[PrintStuff()]):
        tuner.tune(tune_option)


run_tuning()

# If use_vm is true, we should get some errors with the thing not be able to find things!
use_vm = True
print("Compile...")

with auto_scheduler.ApplyHistoryBest(log_file):
    with tvm.transform.PassContext(
        opt_level=3,
        config={"relay.backend.use_auto_scheduler": True},
        instruments=[PrintStuff()],
    ):
        if use_vm:
            exe = relay.vm.compile(mod, target=target, params=params)
        else:
            lib = relay.build(mod, target=target, params=params)