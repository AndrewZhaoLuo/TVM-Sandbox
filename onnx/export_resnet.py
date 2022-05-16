"""
Example of exporting different resnets and showing that task + logs still work between models
with different weights.
"""

import tempfile

import torch
import torchvision
import tvm
from tvm import auto_scheduler, relay

import onnx


def export_to_onnx(torch_model, batch_size, output_file):
    x = torch.randn(batch_size, 3, 224, 224, requires_grad=True)
    _ = torch_model(x)

    # Export the model
    torch.onnx.export(
        torch_model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        output_file,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=10,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
    )


def extract_tasks(torch_model):
    with tempfile.NamedTemporaryFile() as f:
        export_to_onnx(torch_model, 1, f.name)
        onnx_model = onnx.load(f)
        relay_mod, params = relay.frontend.from_onnx(onnx_model)
        tasks, _ = auto_scheduler.extract_tasks(
            relay_mod["main"], params, target="llvm"
        )
        return tasks

if __name__ == "__main__":
    pretrained_model = torchvision.models.resnet50(pretrained=True)
    randomly_init_model = torchvision.models.resnet50(pretrained=False)

    # Assert the weights are different between model
    assert (
        pretrained_model.state_dict()["conv1.weight"]
        != randomly_init_model.state_dict()["conv1.weight"]
    ).any()

    tasks_pretrained = extract_tasks(pretrained_model)
    tasks_randomly_init = extract_tasks(randomly_init_model)

    assert len(tasks_pretrained) == len(tasks_randomly_init)
    for pretrained_task, randomly_init_task in zip(tasks_pretrained, tasks_randomly_init):
        assert pretrained_task.workload_key == randomly_init_task.workload_key
    print("Weight changing works as long as network topology is the same!")
