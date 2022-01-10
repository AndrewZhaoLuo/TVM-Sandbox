"""
An example of using the dynamic to static pass on a large model (BERT): 

https://zenodo.org/record/3733910/files/model.onnx 
"""

import time

import onnx
import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relay.transform.transform import DynamicToStatic

if __name__ == "__main__":
    onnx_model = onnx.load("./models/big_bert.onnx")
    mod, params = relay.frontend.from_onnx(
        onnx_model,
        freeze_params=True,
        # shape={"input_ids": [1, 384], "input_mask": [1, 384], "segment_ids": [1, 384]},
    )
    d2s_pass = relay.transform.DynamicToStatic()
    print(mod)
    start = time.time()
    mod = d2s_pass(mod)
    print(mod)
    print(time.time() - start)
