from os import path

import onnx
from tvm import relay

# SRC: https://github.com/onnx/models/blob/main/vision/classification/densenet-121/README.md
MODEL_PATH = "models/densenet-3.onnx"

if __name__ == "__main__":
    onnx_model = onnx.load(MODEL_PATH)
    input_shapes = {}
    input_dtypes = {}
    initializer_names = [n.name for n in onnx_model.graph.initializer]

    # The inputs contains both the inputs and parameters. We are just interested in the
    # inputs so skip all parameters listed in graph.initializer
    for input_info in onnx_model.graph.input:
        if input_info.name not in initializer_names:
            name, shape, dtype, _ = relay.frontend.onnx.get_info(input_info)
            if dtype is None:
                raise ValueError(
                    f"Unknown dtype on input '{input_info.name}' is not supported."
                )
            input_shapes.update({input_info.name: shape})
            input_dtypes.update({input_info.name: dtype})

    mod, params = relay.frontend.from_onnx(
        onnx_model, shape=input_shapes, freeze_params=True
    )

    print(mod)
