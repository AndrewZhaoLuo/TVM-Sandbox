"""Compare perf of dense packed vs dense not packed."""
import json
import os
import tempfile
from typing import Dict, List

import numpy as np
import torch._C as _C
import torch.functional as F
import tvm
import tvm.meta_schedule.measure_callback as measure_callback
from tvm import meta_schedule, nd, relay
from tvm.contrib import graph_executor
from tvm.meta_schedule import database as ms_database
from tvm.relay import vm
from tvm.runtime import profiler_vm
from tvm.runtime import vm as vm_rt
from tvm.utils import roofline


def tune(mod, params, target: tvm.target.Target, max_trials_global=128, work_dir=None):
    if work_dir is None:
        raise ValueError("ERROR")

    if work_dir is not None:
        os.makedirs(work_dir, exist_ok=True)

    print(f"Tuning mod:\n{mod}")

    database = ms_database.JSONDatabase(work_dir=work_dir)
    with meta_schedule.Profiler() as profiler:
        with target:
            meta_schedule.relay_integration.tune_relay(
                mod,
                params,
                target,
                work_dir=work_dir,
                max_trials_global=max_trials_global,
            )

    print("Tuning Time:")
    print(profiler.table())
    return database


def get_runtime(mod, target, params, database, use_graph=False):
    with database, tvm.transform.PassContext(
        config={
            "relay.backend.use_meta_schedule": True,
            "relay.backend.use_meta_schedule_dispatch": target.kind.name != "cuda",
            "relay.FuseOps.max_depth": 30,
        },
        opt_level=3,
    ):
        if use_graph:
            lib = relay.build(mod, target)
            runtime = graph_executor.GraphModule(lib["default"](dev))
        else:
            exe = vm.compile(
                mod,
                target,
                params=params,
            )
            dev = tvm.device("cuda" if "cuda" in str(target) else "cpu", 0)
            runtime = vm_rt.VirtualMachine(exe, dev)
    return runtime, exe, dev


def benchmark(
    mod,
    params,
    database,
    target: tvm.target.Target,
    input_dict: Dict[str, np.ndarray],
    # These are for printing only
    use_graph: bool = False,
):
    saved_tir = roofline.SaveLoweredTIR()
    # TVM takes nd arrays
    input_dict = {k: nd.array(v) for k, v in input_dict.items()}

    disabled_passes_tir_read = ["tir.CommonSubexprElimTir", "tir.UnrollLoop"]
    # Build once to get TIR (with simplifications if needed)
    with database, target, tvm.transform.PassContext(
        config={
            "relay.backend.use_meta_schedule": True,
            "relay.backend.use_meta_schedule_dispatch": target.kind.name != "cuda",
            "relay.FuseOps.max_depth": 30,
        },
        instruments=[saved_tir],
        disabled_pass=disabled_passes_tir_read,
        opt_level=3,
    ):
        vm.compile(
            mod,
            target,
            params=params,
        )

    runtime, exe, dev = get_runtime(mod, target, params, database, use_graph=use_graph)

    results = runtime.benchmark(
        tvm.cpu(),
        func_name="main",
        number=1,
        repeat=100,
        end_to_end=True,
        **input_dict,
    )  # End to end for being fair vs. onnxrt

    if not use_graph:
        # only vm profiler right now
        vm_profiler = profiler_vm.VirtualMachineProfiler(exe, dev)
        results_profiler = vm_profiler.profile(**input_dict)

    for global_var, tir in saved_tir.functions.items():
        print(global_var, "*" * 50)
        print(tir.script())
        print()
    print(results)
    print(results_profiler)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, required=True)

    np_input = np.random.randint(-100, 100, size=(512, 768))
    np_weights = np.random.randn(3072, 768)

    args = parser.parse_args()
    target = tvm.target.Target(args.target)
    input_var = relay.var("input", shape=[512, 768])
    weight_const = relay.const(np_weights)

    results = relay.nn.dense(input_var, weight_const)

    mod = tvm.IRModule.from_expr(results)

    print(mod)

    dir = "test"
    db = tune(mod, {}, target, work_dir=dir)

    input_dict = {"input": np.random.randn(512, 768).astype("float32")}
    benchmark(mod, {}, db, target, input_dict)

    runtime, exe, dev = get_runtime(mod, target, {}, db, use_graph=False)

    breakpoint()

    print("yay!")
