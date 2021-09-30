from os import path

import onnx
from tvm import relay

MODEL_PATH = "~/Downloads/pytorch_lstm.onnx"

if __name__ == "__main__":
    model_path = path.join(path.expanduser(MODEL_PATH))

    onnx_model = onnx.load(model_path)
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
    mod = relay.transform.DynamicToStatic()(mod)
    breakpoint()
