import subprocess
import tempfile

import onnx
import tensorflow as tf
import tf2onnx
import tflite
import tvm
from tensorflow.python import tf2
from tvm import relay

# Tests the fq2i pass for completeness on some .tflite models
MODEL_PATH = "./models/tflite_model_zoo/mobilenet_v1_1-tflite-int8.tflite"

interpreter = tf.lite.Interpreter(MODEL_PATH)

with tempfile.NamedTemporaryFile() as f:
    subprocess.run(
        [
            "python",
            "-m",
            "tf2onnx.convert",
            "--tflite",
            MODEL_PATH,
            "--output",
            f.name,
            "--opset",
            "13",
        ]
    )
    onnx_model = onnx.load(f.name)
    mod, params = relay.frontend.from_onnx(onnx_model, freeze_params=True)

    breakpoint()

    # Transform fake quantized sub-graphs to actual integer ops.
    # Should have no effect on graphs without the relevant patterns.
    mod = relay.transform.InferType()(mod)
    fq2i_mod = relay.transform.FakeQuantizationToInteger()(mod)

    breakpoint()
