import tvm
from tvm import relay
import onnx
from tvm import auto_scheduler

"""Examine task extraction failures for NHWC"""

MODEL = "./models/shufflenet-9.onnx"

# Grouped convs are not supported in NHWC
onnx_model = onnx.load(MODEL)
mod, params = relay.frontend.from_onnx(onnx_model, freeze_params=True)

# Convert convs to NHWC Layout
desired_layouts = {
    "nn.conv2d": ["NHWC", "default"],
    "nn.conv2d_transpose": ["NHWC", "default"],
    "nn.upsampling": ["NHWC", "default"],
    "image.resize2d": ["NHWC", "default"],
    "vision.roi_align": ["NHWC", "default"],
}
seq = tvm.transform.Sequential([relay.transform.ConvertLayout(desired_layouts)])
with tvm.transform.PassContext(
    opt_level=3, config={"relay.backend.use_auto_scheduler": True}
):
    mod = seq(mod)
    mod = tvm.relay.transform.EliminateCommonSubexpr()(mod)
    mod = tvm.relay.transform.FoldConstant()(mod)

try:
    tasks, task_weights = auto_scheduler.extract_tasks(
        mod["main"], params, target="cuda"
    )
except Exception:
    print("yay!")
