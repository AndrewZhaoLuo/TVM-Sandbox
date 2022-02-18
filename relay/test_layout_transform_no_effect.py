"""Shows layout transform affects models without any layout dependent ops."""

import onnx
import tvm
from tvm import relay

MODEL = "models/gpt2-10.onnx"
SHAPE_DICT = {"input1": [1, 1, 8]}

"""
MODEL = "models/bertsquad-8.onnx"
SHAPE_DICT = {
    "unique_ids_raw_output___9:0": [1],
    "segment_ids:0": [1, 256],
    "input_mask:0": [1, 256],
    "input_ids:0": [1, 256],
}
"""
onnx_model = onnx.load(MODEL)

mod, param = relay.frontend.from_onnx(onnx_model, freeze_params=True, shape=SHAPE_DICT)
mod = tvm.transform.Sequential(
    [
        # These are pre-reqs for layout transform pass
        relay.transform.InferType(),
        relay.transform.EliminateCommonSubexpr(),
        relay.transform.CanonicalizeOps(),
        relay.transform.FoldConstant(),
    ]
)(mod)
print("Before:")
print(mod)

print("After:")
mod, param = relay.frontend.from_onnx(onnx_model, freeze_params=True, shape=SHAPE_DICT)
"""Trys to convert model to NHWC layout, this mutates the model."""
desired_layouts = {
    "nn.conv2d": ["NHWC", "default"],
    "nn.conv2d_transpose": ["NHWC", "default"],
    "nn.upsampling": ["NHWC", "default"],
    # "image.resize2d": ["NHWC", "default"], // TODO(mbrookhart): fix this in TVM
    "vision.roi_align": ["NHWC", "default"],
}

# We must use tvm.transform.Sequential since these passes have pre-req Passes
# and Sequential guarantees we run those dependencies
seq = tvm.transform.Sequential(
    [
        relay.transform.InferType(),
        relay.transform.ConvertLayout(desired_layouts),
        relay.transform.EliminateCommonSubexpr(),
        relay.transform.FoldConstant(),
    ]
)
with tvm.transform.PassContext(
    opt_level=3, config={"relay.backend.use_auto_scheduler": True}
):
    mod = seq(mod)

print(mod)
