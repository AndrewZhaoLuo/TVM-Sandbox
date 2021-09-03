"""Debugging https://github.com/apache/tvm/issues/8918 which test sometimes produce a NaN by multiplying by 0"""

from numbers import Integral

import numpy as np
import tvm
from numpy import dtype, random
from numpy.core.fromnumeric import shape
from tvm import relay
from tvm.ir import IRModule
from tvm.relay import function as _function
from tvm.relay import transform as _transform


def get_const_int(expr):
    """Verifies expr is integer and get the constant value.

    Parameters
    ----------
    expr : tvm.Expr or int
        The input expression.

    Returns
    -------
    out_value : int
        The output.
    """
    if isinstance(expr, Integral):
        return expr
    if not isinstance(expr, tvm.tir.IntImm):
        ana = tvm.arith.Analyzer()
        expr = ana.simplify(expr)
    if not isinstance(expr, tvm.tir.IntImm):
        raise ValueError("Expect value to be constant int")
    return int(expr.value)


def get_const_tuple(in_tuple):
    """Verifies input tuple is IntImm or Var, returns tuple of int or Var.

    Parameters
    ----------
    in_tuple : tuple of Expr
        The input.

    Returns
    -------
    out_tuple : tuple of int
        The output.
    """
    ret = []
    ana = None
    for elem in in_tuple:
        if isinstance(elem, (tvm.tir.Var, tvm.tir.expr.Any)):
            ret.append(elem)
        elif not isinstance(elem, (tvm.tir.IntImm, int)):
            ana = tvm.arith.Analyzer() if ana is None else ana
            elem = ana.simplify(elem)
            if not isinstance(elem, tvm.tir.IntImm):
                ret.append(elem)
            else:
                ret.append(get_const_int(elem))
        else:
            ret.append(get_const_int(elem))
    return tuple(ret)


def infer_shape(inputs, mod=None):
    """A method to get the output type of an intermediate node in the graph."""
    out_type = infer_type(inputs, mod=mod)
    checked_type = out_type.checked_type
    if hasattr(checked_type, "shape"):
        # Regular operator that outputs tensors
        return get_const_tuple(checked_type.shape)
    # The return type is not a tensor, for example List
    return checked_type


def infer_type(node, mod=None):
    """A method to infer the type of an intermediate node in the relay graph."""
    if isinstance(mod, IRModule):
        mod["main"] = _function.Function(tvm.relay.analysis.free_vars(node), node)
        mod = _transform.InferType()(mod)
        entry = mod["main"]
        ret = entry.body
    else:
        new_mod = IRModule.from_expr(node)
        if mod is not None:
            new_mod.update(mod)

        new_mod = _transform.InferType()(new_mod)
        entry = new_mod["main"]
        ret = entry if isinstance(node, _function.Function) else entry.body

    return ret


def get_input_data_shape_dict(graph_def, input_data):
    if isinstance(input_data, list):
        input_names = {}
        shape_dict = {}
        for i, _ in enumerate(input_data):
            input_names[i] = graph_def.graph.input[i].name
            shape_dict[input_names[i]] = input_data[i].shape
    else:
        input_names = graph_def.graph.input[0].name
        shape_dict = {input_names: input_data.shape}

    return input_names, shape_dict


def get_tvm_output_with_vm(
    graph_def,
    input_data,
    target,
    dev,
    opset=None,
    freeze_params=False,
    convert_to_static=False,
    convert_config=None,
):
    """Generic function to execute and get tvm output with vm executor"""
    if not isinstance(input_data, list):
        input_data = [input_data]
    _, shape_dict = get_input_data_shape_dict(graph_def, input_data)

    mod, params = relay.frontend.from_onnx(
        graph_def,
        shape_dict,
        opset=opset,
        freeze_params=freeze_params,
        convert_config=convert_config,
    )

    if convert_to_static:
        mod = relay.transform.DynamicToStatic()(mod)

    result = relay.create_executor("vm", mod=mod, device=dev, target=target).evaluate()(
        *input_data, **params
    )
    if isinstance(result, tvm.runtime.NDArray):
        return result.numpy()
    return [r.numpy() for r in result]


def get_nllloss(input_tensor, target_tensor, weight_tensor=None, ignore_index=0):
    channels = infer_shape(input_tensor)[1]
    weight_tensor = relay.ones(
        [channels],
        dtype=input_tensor.type_annotation.dtype,
    )

    loss = -relay.gather(
        input_tensor, axis=1, indices=relay.expand_dims(target_tensor, 1)
    )
    loss = relay.squeeze(loss, axis=[1])

    expanded_target_tensor = relay.expand_dims(target_tensor, 0)
    expanded_target_tensor = relay.nn.batch_flatten(expanded_target_tensor)
    flattened_weights = relay.gather_nd(weight_tensor, expanded_target_tensor)
    select_weights = relay.reshape_like(flattened_weights, loss)
    loss *= select_weights

    if ignore_index is not None:
        # "Ignore" values whose target is the ignore_index
        mask_tensor = relay.equal(
            target_tensor,
            relay.const(ignore_index, dtype=target_tensor.type_annotation.dtype),
        )
        mask_tensor = relay.const(1, dtype="int8") - relay.cast(mask_tensor, "int8")
        loss *= relay.cast_like(mask_tensor, loss)

        # This is not explained super clearly in the onnx spec, but masked values don't
        # contribute toward the final value in reduction
        select_weights *= relay.cast_like(mask_tensor, select_weights)

    weight_total = relay.sum(select_weights)

    return loss


if __name__ == "__main__":
    for i in range(100):
        shape_input = [3, 5, 6, 6, 5]
        shape_target = shape_input[:1] + shape_input[2:]
        result = get_nllloss(
            relay.var("input", shape=shape_input),
            relay.var("target", shape=shape_target, dtype="int32"),
            ignore_index=0,
        )

        input_np = random.uniform(-10, 10, size=shape_input).astype("float32")
        target_np = np.ones(shape_target)
        target_np = target_np.flatten()
        target_np[0] = 0
        target_np = target_np.reshape(shape_target).astype("int32")
        mod = IRModule.from_expr(result)
        result = (
            relay.create_executor("vm", mod=mod)
            .evaluate()(
                input_np,
                target_np,
            )
            .asnumpy()
        )
        print(result.flatten()[:10])
