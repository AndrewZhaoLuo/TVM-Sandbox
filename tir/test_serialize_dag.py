"""Test serializing dag"""
import hashlib

import tvm
from tvm import auto_scheduler, te, topi
from tvm.ir.base import load_json, save_json

LOG_LINE = '{"i": [["[\\"conv2d_layer\\", 1, 7, 7, 512, 512, 3, 3, [1, 1], [1, 1]]", "llvm -keys=cpu -link-params=0 -mcpu=broadwell -num-cores=2", [8, 64, 64, 0, 0, 0, 0, 0], "", 1, []], [[], [["CI", 5], ["SP", 3, 0, 1, [1, 1, 1], 1], ["SP", 3, 4, 512, [1, 32, 16], 1], ["SP", 3, 8, 7, [7, 1, 1], 1], ["SP", 3, 12, 7, [1, 1, 1], 1], ["SP", 3, 16, 512, [1], 1], ["SP", 3, 18, 3, [1], 1], ["SP", 3, 20, 3, [3], 1], ["RE", 3, [0, 4, 8, 12, 1, 5, 9, 13, 16, 18, 20, 2, 6, 10, 14, 17, 19, 21, 3, 7, 11, 15]], ["FSP", 6, 0, 1, 2], ["FSP", 6, 3, 2, 2], ["FSP", 6, 6, 3, 2], ["FSP", 6, 9, 4, 2], ["RE", 6, [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]], ["CA", 3, 6, 7], ["CA", 1, 6, 5], ["FU", 6, [0, 1, 2, 3, 4, 5]], ["AN", 6, 0, 3], ["PR", 3, 0, "auto_unroll_max_step$512"], ["AN", 1, 3, 2], ["AN", 3, 21, 2], ["AN", 6, 6, 2]]]], "r": [[0.0331129], 0, 0.900362, 1647464342], "v": "v0.6"}\n'

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

    for _ in range(10):
        compute_dag = inp.task.compute_dag
        print("Dag:", compute_dag)

        json_str = save_json(compute_dag)
        print("Json\n", json_str)

        # Can't recreate workload
        print(compute_dag)
        print("Json\n", load_json(json_str))
