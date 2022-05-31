import tvm
import tvm.autotvm

AUTOTVM_LOG_FILE = "codegen/tuning_records/tuning_records_fp32"
all_results = list(tvm.autotvm.record.load_from_file(AUTOTVM_LOG_FILE))

# type is MeausreInput in tvm/autotvm/measure/measure.py
measure_input = all_results[0][0]

target = measure_input.target
task = measure_input.task
cfg = measure_input.config
with target:
    with tvm.autotvm.apply_history_best(AUTOTVM_LOG_FILE):
        result = task.func(*task.args, **task.kwargs)
        print(tvm.lower(result[0], result[1]))
