import onnx
import tvm
from tvm import relay

onnx_model = onnx.load("models/arcfaceresnet100-8.onnx")
mod, params = relay.frontend.from_onnx(onnx_model, freeze_params=True)

mod = relay.transform.InferType()(mod)

# Example of transforming layout to NHWC
"""Trys to convert model to NHWC layout, this mutates the model."""
desired_layouts = {
    "nn.conv2d": ["NHWC", "default"],
    "nn.conv2d_transpose": ["NHWC", "default"],
    "nn.upsampling": ["NHWC", "default"],
    "image.resize2d": ["NHWC", "default"],
    "vision.roi_align": ["NHWC", "default"],
}
with tvm.transform.PassContext(
    opt_level=3, config={"relay.backend.use_auto_scheduler": True}
):
    print("Rewriting layout to NHWC")
    model_nhwc = relay.transform.ConvertLayout(desired_layouts)(mod)

    print("Cleaning up NHWC graph")
    model_nhwc = tvm.relay.transform.EliminateCommonSubexpr()(model_nhwc)
    model_nhwc = tvm.relay.transform.FoldConstant()(model_nhwc)

print(f"{'*' * 10} Old mod {'*' * 10}")
print(mod)

print(f"{'*' * 10} New mod {'*' * 10}")
print(model_nhwc)
