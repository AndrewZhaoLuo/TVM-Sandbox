import multiprocessing as mp

from fp16_pass.benchmark_fp16 import benchmark_model, get_yolo2

if __name__ == "__main__":
    # Assume using m1 mac
    benchmark_model(
        get_yolo2,
        run_fp16_pass=True,
        run_other_opts=True,
        target="llvm",
        target_host="llvm -mcpu=apple-latest -mtriple=arm64-apple-macos",
    )
    benchmark_model(
        get_yolo2,
        run_fp16_pass=False,
        run_other_opts=True,
        target="llvm",
        target_host="llvm -mcpu=apple-latest -mtriple=arm64-apple-macos",
    )
