import subprocess
import tempfile

import onnx
from tvm import relay
import os

# Tests the fq2i pass for completeness on some .tflite models
MODEL_PATHS = "./models/tflite_model_zoo/"

for model_name in os.listdir(MODEL_PATHS):
    model_path = os.path.join(MODEL_PATHS, model_name)

    with tempfile.NamedTemporaryFile() as f:
        print(f"Processing model {model_path}")
        subprocess.run(
            [
                "python",
                "-m",
                "tf2onnx.convert",
                "--tflite",
                model_path,
                "--output",
                f.name,
                "--opset",
                "13",
            ]
        )

        with open(
            f"./models/tflite_model_zoo_txt/{model_name}.txt", "w"
        ) as output_file:
            onnx_model = onnx.load(f.name)
            mod, params = relay.frontend.from_onnx(onnx_model, freeze_params=True)

            print("Original:", file=output_file)
            print(mod, file=output_file)
            print()

            # Transform fake quantized sub-graphs to actual integer ops.
            # Should have no effect on graphs without the relevant patterns.
            mod = relay.transform.InferType()(mod)
            fq2i_mod = relay.transform.FakeQuantizationToInteger()(mod)
            print("Quantized:", file=output_file)
            print(fq2i_mod, file=output_file)
            print()