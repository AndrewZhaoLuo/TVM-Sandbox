import json
from black import tempfile
import tvm
import tvm.auto_scheduler

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


def from_extract_task():
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
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
    for idx, task in enumerate(tasks):
        print(
            "========== Task %d  (workload key: %s) =========="
            % (idx, task.workload_key)
        )
        print(task.compute_dag)

    # Note this depends on tvm.auto_scheduler.workload_registry.WORKLOAD_FUNC_REGISTRY
    # which is populated in extract_tasks
    LOG_FILE = "resnet-18-NCHW-B1-llvm.json"
    BEST_LOG_FILE = "resnet-18-NCHW-B1-llvm-graph-tuned.json"
    with auto_scheduler.ApplyHistoryBest(LOG_FILE):
        with tvm.transform.PassContext(
            opt_level=3, config={"relay.backend.use_auto_scheduler": True}
        ):
            all_results = list(tvm.auto_scheduler.load_records(LOG_FILE))
            print(tvm.auto_scheduler.workload_registry.WORKLOAD_FUNC_REGISTRY)
            inp = tvm.auto_scheduler.measure.recover_measure_input(
                all_results[0][0], rebuild_state=True
            )
            sch, args = inp.task.compute_dag.apply_steps_from_state(inp.state)
            print(tvm.lower(sch, args))

            # Can serialize individual keys in workload registry
            with tempfile.NamedTemporaryFile() as f:
                data = tvm.auto_scheduler.workload_registry.serialize_workload_registry_entry(
                    inp.workload
                )
                json_data = {"k": data[0], "v": data[1]}
                json.dumps(json_data, sort_keys=True)
            breakpoint()
            print("done!")
