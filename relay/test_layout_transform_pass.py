from tvm import relay
import onnx
import onnxoptimizer
import tvm

onnx_model = onnx.load("./models/resnet50-v1-7.onnx")

# Testing code for running simplification passes for onnx
passes = onnxoptimizer.get_fuse_and_elimination_passes()

# all passes performs some funky passes which not only generate invalid onnx,
# but maybe split up a graph into multiple subgraphs
# onnx_model = onnxoptimizer.optimize(onnx_model, passes)

mod, params = relay.frontend.from_onnx(onnx_model)

mod = relay.transform.InferType()(mod)

# Convert model from NCHW --> NHWC
transform = relay.transform.ConvertLayout(
    {"nn.conv2d": ["NHWC", "default"], "vision.roi_align": ["NHWC", "default"]}
)

with tvm.transform.PassContext(opt_level=3):
    mod = transform(mod)

opt_mod, opt_params = relay.optimize(mod, params=params, target="cuda")

breakpoint()