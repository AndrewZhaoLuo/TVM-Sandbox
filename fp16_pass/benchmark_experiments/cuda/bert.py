import multiprocessing as mp

from fp16_pass.benchmark_fp16 import benchmark_model, get_bert

if __name__ == "__main__":
    benchmark_model(
        get_bert,
        run_fp16_pass=True,
        run_other_opts=True,
        target="cuda",
        target_host="llvm",
    )
    benchmark_model(
        get_bert,
        run_fp16_pass=False,
        run_other_opts=True,
        target="cuda",
        target_host="llvm",
    )
