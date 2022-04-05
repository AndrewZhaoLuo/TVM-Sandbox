from typing import List

import torch
import torch._C as _C
import torch.onnx
from torch import nn

TrainingMode = _C._onnx.TrainingMode

import os
from os import path


def export_model(
    torch_model,
    x,
    name,
    dir="export/",
    constant_fold=True,
    input_names=["input"],
    output_names=["output"],
):

    if torch.cuda.is_available():
        cuda_device = torch.device("cuda")
        x.to(device=cuda_device)

    # Get trace
    _ = torch_model(x)

    if not os.path.exists(dir):
        os.makedirs(dir)

    # Export the model
    torch.onnx.export(
        torch_model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        path.join(
            dir, f"{name}.onnx"
        ),  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=12,  # the ONNX version to export the model to
        do_constant_folding=constant_fold,  # whether to execute constant folding for optimization
        input_names=input_names,  # the model's input names
        output_names=output_names,  # the model's output names
        dynamic_axes={},  # variable name axis
        training=TrainingMode.EVAL,
    )


# Like multihead attention from pytorch, but derives weight, key, and query matrix seperately
# Also fixes them to be same size
class BigModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: List[torch.Tensor]):
        results = []
        for input in inputs:
            value, weight = torch.split(input, 1)
            value = value.squeeze()
            weight = weight.squeeze()
            result = torch.bmm(value, weight)
            results.append(result)

        big_result = torch.cat(results)

        results = [big_result] + results[1:]

        results = [tensor.float() for tensor in results]
        return results


if __name__ == "__main__":
    num_inputs = 8
    input_shape = [2, 4, 16, 16]
    torch_module = BigModel()

    input_tensors = [
        torch.randn(input_shape, dtype=torch.float16) for _ in range(num_inputs)
    ]
    results = torch_module(input_tensors)

    export_model(torch_module, input_tensors, "big_model", dir=".")
