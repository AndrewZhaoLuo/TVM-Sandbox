"""Script meant to be baseline for compiling and running autotvm tuned model."""
import multiprocessing as mp
from os import path

import numpy as np
import tvm
from fp16_pass.benchmark_fp16 import benchmark_model, get_yolo2
from tvm import relay
from tvm.driver import tvmc
from tvm.driver.tvmc.model import TVMCModel
from tvm.relay.transform import InferType, ToMixedPrecision


def benchmark_model(
    model_func,
    tuning_records,
    run_fp16_pass=True,
    run_other_opts=True,
    target="llvm",
    target_host="llvm",
    measure_number=100,
    measure_repeats=10,
):
    print("FP16 pass" if run_fp16_pass else "FP32 pass")
    """Get Module"""
    tvmc_model = model_func(run_pass=run_fp16_pass, run_opts=run_other_opts)
    # Create package artifacts

    print("Compiling:")
    package = tvmc.compile(
        tvmc_model,
        target=target,
        tuning_records=tuning_records,
        target_host=target_host,
    )

    print("Benchmarking:")
    result = tvmc.run(
        package,
        device="cpu" if target == "llvm" else target,
        repeat=measure_number,
        number=measure_repeats,
    )
    print(result)
    print()


if __name__ == "__main__":
    print("FP32:")
    benchmark_model(
        get_yolo2,
        "./codegen/tuning_records/tuning_records_fp32",
        run_fp16_pass=False,
        target_host="llvm -mcpu=apple-latest -mtriple=arm64-apple-macos --ffast-math=true",
    )
    print()

    print("FP16:")
    benchmark_model(
        get_yolo2,
        "./codegen/tuning_records/tuning_records_fp16",
        run_fp16_pass=True,
        target_host="llvm -mcpu=apple-latest -mtriple=arm64-apple-macos --ffast-math=true",
    )
