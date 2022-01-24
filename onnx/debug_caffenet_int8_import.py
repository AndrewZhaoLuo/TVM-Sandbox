"""
Debugging 
https://discuss.tvm.apache.org/t/onnx-relay-type-checker-error-when-calling-relay-frontend-from-onnx-on-a-quantized-model/11941
"""

import onnx
from tvm import relay

model_path = "models/caffenet-12-int8.onnx"
onnx_model = onnx.load(model_path)
shape_dict = {'data_0': (1, 3, 224, 224)}

# should be successful but fails with:
"""
The Relay type checker is unable to show the following types match:
  Tensor[(1), float32]
  Tensor[(96), float32]
In particular:
  dimension 0 conflicts: 1 does not match 96.
The Relay type checker is unable to show the following types match.
In particular `Tensor[(96), float32]` does not match `Tensor[(1), float32]`
note: run with `TVM_BACKTRACE=1` environment variable to display a backtrace.
"""
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)