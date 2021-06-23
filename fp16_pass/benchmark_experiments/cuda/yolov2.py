import multiprocessing as mp

from fp16_pass.benchmark_fp16 import benchmark_model, get_yolo2

if __name__ == "__main__":
    # macOS has 'spawn' as default which doesn't work for tvm
    mp.set_start_method("fork")

    benchmark_model(
        get_yolo2,
        run_fp16_pass=True,
        run_other_opts=True,
        target="cuda",
        target_host="llvm",
    )
    benchmark_model(
        get_yolo2,
        run_fp16_pass=False,
        run_other_opts=True,
        target="cuda",
        target_host="llvm",
    )
