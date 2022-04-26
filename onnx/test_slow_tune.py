import tvm
from tvm import auto_scheduler, relay

import onnx


def get_network(weight, batch_size):
    """Get the symbol definition and random weight of a network"""

    input_shape = (batch_size, 3, 224, 224)
    onnx_model = onnx.load(weight)
    input_name = "data"
    shape_dict = {input_name: input_shape}
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    desired_layouts = {
        "nn.conv2d": ["NHWC", "default"],
        "image.resize2d": ["NHWC"],
        "nn.upsampling": ["NHWC"],
    }
    seq = tvm.transform.Sequential(
        [
            relay.transform.RemoveUnusedFunctions(),
            relay.transform.ConvertLayout(desired_layouts),
        ]
    )
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)

    mod = tvm.IRModule.from_expr(mod["main"])
    mod = tvm.relay.transform.FastMath()(mod)
    mod = tvm.relay.transform.EliminateCommonSubexpr()(mod)
    BindPass = tvm.relay.transform.function_pass(
        lambda fn, new_mod, ctx: tvm.relay.build_module.bind_params_by_name(fn, params),
        opt_level=1,
    )
    mod = BindPass(mod)
    mod = tvm.relay.transform.FoldConstant()(mod)
    mod = tvm.relay.transform.CombineParallelBatchMatmul()(mod)
    mod = tvm.relay.transform.FoldConstant()(mod)

    mod = tvm.relay.transform.InferType()(mod)
    mod = tvm.relay.transform.ToMixedPrecision()(mod)
    mod = tvm.relay.transform.EliminateCommonSubexpr()(mod)
    mod = tvm.relay.transform.FoldConstant()(mod)
    mod = tvm.relay.transform.CombineParallelBatchMatmul()(mod)
    mod = tvm.relay.transform.FoldConstant()(mod)

    return mod, params, input_shape


# RPC and cross compile
target = tvm.target.Target("cuda -arch=sm_53", host="llvm -mtriple=aarch64-linux-gnu")
mod, params, input_shape = get_network("models/vgg16-12.onnx", 1)
breakpoint()
tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target, hardware_params=auto_scheduler.HardwareParams)


def run_tuning():
    print("Begin tuning...")
    #  measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=10)

    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=20800,  # change this to 20000 to achieve the best performance
        builder=auto_scheduler.LocalBuilder(timeout=1000),
        #  runner=measure_ctx.runner,
        runner=auto_scheduler.RPCRunner(
            "nano", "10.36.172.151", 9190, min_repeat_ms=300, timeout=1000, repeat=1
        ),
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )

    tuner.tune(tune_option)


run_tuning()
