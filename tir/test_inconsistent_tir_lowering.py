"""Demonstrate inconsistent tir lowering"""
import hashlib

import tvm
from tvm import auto_scheduler, te, topi
from tvm.ir.base import save_json

LOG_LINE = '{"i": [["[\\"conv2d_layer\\", 1, 7, 7, 512, 512, 3, 3, [1, 1], [1, 1]]", "llvm -keys=cpu -link-params=0 -mcpu=broadwell -num-cores=2", [8, 64, 64, 0, 0, 0, 0, 0], "", 1, []], [[], [["CI", 5], ["SP", 3, 0, 1, [1, 1, 1], 1], ["SP", 3, 4, 512, [2, 1, 1], 1], ["SP", 3, 8, 7, [1, 1, 1], 1], ["SP", 3, 12, 7, [1, 7, 1], 1], ["SP", 3, 16, 512, [32], 1], ["SP", 3, 18, 3, [1], 1], ["SP", 3, 20, 3, [3], 1], ["RE", 3, [0, 4, 8, 12, 1, 5, 9, 13, 16, 18, 20, 2, 6, 10, 14, 17, 19, 21, 3, 7, 11, 15]], ["FSP", 6, 0, 1, 1], ["FSP", 6, 2, 2, 1], ["FSP", 6, 4, 3, 1], ["FSP", 6, 6, 4, 1], ["RE", 6, [0, 2, 4, 6, 1, 3, 5, 7]], ["CA", 3, 6, 3], ["CR", 1], ["FU", 1, [0, 1]], ["AN", 1, 0, 3], ["FU", 6, [0, 1]], ["AN", 6, 0, 3], ["PR", 3, 0, "auto_unroll_max_step$16"], ["AN", 1, 2, 2], ["AN", 3, 21, 2], ["AN", 6, 6, 2]]]], "r": [[0.0279606], 0, 0.848875, 1647464340], "v": "v0.6"}\n'

# The workload associated with the log
@auto_scheduler.register_workload
def conv2d_layer(N, H, W, CO, CI, KH, KW, stride, padding):
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(
        data, kernel, stride, padding, dilation=1, out_dtype="float32"
    )
    out = topi.nn.relu(conv + bias)
    return [data, kernel, bias, out]


if __name__ == "__main__":
    inp, inr = auto_scheduler.measure_record.load_record_from_string(LOG_LINE)
    inp = auto_scheduler.measure.recover_measure_input(inp, rebuild_state=True)
    sch, args = inp.task.compute_dag.apply_steps_from_state(inp.state)

    for _ in range(10):
        ir_module = tvm.lower(sch, args)
        primfunc = ir_module["main"]
        json_str = save_json(primfunc)
        print(hashlib.sha256(json_str.encode("utf-8")).hexdigest())
