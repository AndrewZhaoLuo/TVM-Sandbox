"""Code snippet for using onnx optimizer tools"""

import onnxoptimizer
import onnx
from onnxsim import simplify

# Run onnx simplification tool first
onnx_model = onnx.load_model("./models/yolov2-coco-9.onnx")
simplified_model, check = simplify(onnx_model)

if check:
    onnx_model = simplified_model
else:
    print("Simplification tool failed")

# Testing code for running simplification passes for onnx
passes = onnxoptimizer.get_fuse_and_elimination_passes()

# all passes performs some funky passes which not only generate invalid onnx,
# but maybe split up a graph into multiple subgraphs
optimized_model = onnxoptimizer.optimize(onnx_model, passes)

onnx.save_model(optimized_model, "./models/yolov2-coco-9-optimized.onnx")
