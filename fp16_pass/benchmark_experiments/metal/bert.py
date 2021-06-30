import multiprocessing as mp

from fp16_pass.benchmark_fp16 import benchmark_model, get_bert

if __name__ == "__main__":
    # For whatever reason, the first tuning metal jobs emits some errors so run a short job
    # Assume using m1 mac
    benchmark_model(
        get_bert,
        run_fp16_pass=True,
        run_other_opts=True,
        target="metal",
        target_host="llvm -mcpu=apple-latest -mtriple=arm64-apple-macos",
        tuning_trials=100,
    )

    benchmark_model(
        get_bert,
        run_fp16_pass=True,
        run_other_opts=True,
        target="metal",
        target_host="llvm -mcpu=apple-latest -mtriple=arm64-apple-macos",
    )
    benchmark_model(
        get_bert,
        run_fp16_pass=False,
        run_other_opts=True,
        target="metal",
        target_host="llvm -mcpu=apple-latest -mtriple=arm64-apple-macos",
    )
