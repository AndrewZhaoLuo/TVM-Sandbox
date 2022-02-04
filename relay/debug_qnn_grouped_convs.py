"""
Debugging https://github.com/apache/tvm/issues/10088
"""

import onnx
import tvm
from tvm import relay


def build_tvm_model(mod):
  with tvm.transform.PassContext(opt_level=3):
     lib = relay.build(mod, target="llvm")
  return tvm.contrib.graph_executor.GraphModule(lib["default"](tvm.device('llvm', 0)))

model = onnx.load("models/dw_conv2d_h6_w10_c8_cm5_k3_s1_l1_fp32_onnxqt.pt_quant.onnx")
mod, params = relay.frontend.from_onnx(model)
build_tvm_model(mod)
