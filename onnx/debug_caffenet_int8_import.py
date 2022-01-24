"""
Debugging 
https://discuss.tvm.apache.org/t/onnx-relay-type-checker-error-when-calling-relay-frontend-from-onnx-on-a-quantized-model/11941
"""

import onnx
from tvm import relay

model_path = "models/caffenet-12-int8.onnx"
onnx_model = onnx.load(model_path)
shape_dict = {'data_0': (1, 3, 224, 224)}

mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)