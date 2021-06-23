import multiprocessing as mp

from fp16_pass.benchmark_fp16 import benchmark_model, get_bert

if __name__ == "__main__":
    # macOS has 'spawn' as default which doesn't work for tvm
    mp.set_start_method("fork")

    # Assume using m1 mac
    benchmark_model(
        get_bert,
        run_fp16_pass=True,
        run_other_opts=True,
        target="llvm",
        target_host="llvm -mcpu=apple-latest -mtriple=arm64-apple-macos",
    )
    benchmark_model(
        get_bert,
        run_fp16_pass=False,
        run_other_opts=True,
        target="llvm",
        target_host="llvm -mcpu=apple-latest -mtriple=arm64-apple-macos",
    )
