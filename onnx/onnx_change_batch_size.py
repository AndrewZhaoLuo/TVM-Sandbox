import onnx
from onnx.tools import update_model_dims
import struct
import onnx.checker

model = onnx.load("./models/ssd-10.onnx")

# Grab the names of the input/output nodes in protobuf
output_names = [node.name for node in model.graph.output]
all_names_input = [node.name for node in model.graph.input]
all_names_initializer = [node.name for node in model.graph.initializer]
user_provided_input_names = set(all_names_input) - set(all_names_initializer)


def grab_node_shape(node):
    """Each dim in the node can be a number or a name (dynamic)"""
    shape = []
    for dim in node.type.tensor_type.shape.dim:
        if dim.dim_value == 0:
            # dim param, dynamic dim
            assert dim.dim_param != "", f"unknown dim {dim}"
            shape.append(dim.dim_param)
        else:
            shape.append(dim.dim_value)
    return shape


input_shapes = {}
for input in model.graph.input:
    if input.name not in user_provided_input_names:
        continue

    input_shapes[input.name] = grab_node_shape(input)

output_shapes = {}
for output in model.graph.output:
    if output.name not in output_names:
        continue

    output_shapes[output.name] = grab_node_shape(output)

print(input_shapes)
print(output_shapes)

# Method 1 (only works for dynamic batch sizes)
# Now change the batch size, assume every input and output tensor dim 0 is the batch size
# and we can change it
"""
new_input_shapes = dict(input_shapes)
new_output_shapes = dict(output_shapes)

for k in new_input_shapes.keys():
    new_input_shapes[k][0] = 16

for k in new_output_shapes.keys():
    new_output_shapes[k][0] = 16

model_result = update_model_dims.update_inputs_outputs_dims(
    model, new_input_shapes, new_output_shapes
)

onnx.save(model_result, "./models/ssd-10-bs10.onnx")
"""

# Method 2: from https://github.com/onnx/onnx/issues/2182
# When the batch size is fixed
def rebatch(model, batch_size: int, names=set()):
    graph = model.graph

    # Change batch size in input, output and value_info
    for tensor in list(graph.input) + list(graph.value_info) + list(graph.output):
        if tensor.name not in names:
            continue
        tensor.type.tensor_type.shape.dim[0].dim_value = batch_size

    # Set dynamic batch size in reshapes (-1)
    # Perhaps not needed for us?
    for node in graph.node:
        if node.op_type != "Reshape":
            continue
        for init in graph.initializer:
            # node.input[1] is expected to be a reshape
            if init.name != node.input[1]:
                continue
            # Shape is stored as a list of ints
            if len(init.int64_data) > 0:
                # This overwrites bias nodes' reshape shape but should be fine
                init.int64_data[0] = -1
            # Shape is stored as bytes
            elif len(init.raw_data) > 0:
                shape = bytearray(init.raw_data)
                struct.pack_into("q", shape, 0, -1)
                init.raw_data = bytes(shape)

    return model


model = rebatch(
    model, batch_size=8, names=set(output_names + list(user_provided_input_names))
)
# Make sure model looks good, run full_check to run shape inference and make sure it looks good
onnx.checker.check_model(model, full_check=True)
onnx.save(model, "./models/bs10.onnx")