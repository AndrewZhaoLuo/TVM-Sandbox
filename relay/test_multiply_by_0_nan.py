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


def get_nllloss(
    input_tensor,
    target_tensor,
    weight_tensor=None,
    ignore_index=0,
    get_mask_tensor=False,
    get_weights=False,
):
    channels = infer_shape(input_tensor)[1]
    weight_tensor = relay.ones(
        [channels],
        dtype=input_tensor.type_annotation.dtype,
    )
    relay.gather_nd

    loss = -relay.gather(
        input_tensor,
        axis=1,
        indices=relay.expand_dims(target_tensor, 1),
        support_negative_indices=True,
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

    if get_mask_tensor:
        return mask_tensor
    if get_weights:
        return select_weights

    return loss


def test_failing_onnx_test_case():
    import pickle

    inputs = pickle.load(open("./relay/inputs.pkl", "rb"))
    input_np = inputs[0]
    target_np = inputs[1].astype("int32")

    for i in range(100):
        shape_input = input_np.shape
        shape_target = target_np.shape

        result = get_nllloss(
            relay.var("input", shape=shape_input),
            relay.var("target", shape=shape_target, dtype="int32"),
            ignore_index=0,
        )
        mod = IRModule.from_expr(result)
        result_masked = (
            relay.create_executor("vm", mod=mod)
            .evaluate()(
                input_np,
                target_np,
            )
            .asnumpy()
        )

        result = get_nllloss(
            relay.var("input", shape=shape_input),
            relay.var("target", shape=shape_target, dtype="int32"),
            ignore_index=0,
            get_mask_tensor=True,
        )
        mod = IRModule.from_expr(result)
        result_mask = (
            relay.create_executor("vm", mod=mod)
            .evaluate()(
                # input_np,
                target_np,
            )
            .asnumpy()
        )

        result = get_nllloss(
            relay.var("input", shape=shape_input),
            relay.var("target", shape=shape_target, dtype="int32"),
            ignore_index=None,
        )
        mod = IRModule.from_expr(result)
        result_unmasked = (
            relay.create_executor("vm", mod=mod)
            .evaluate()(
                input_np,
                target_np,
            )
            .asnumpy()
        )

        result = get_nllloss(
            relay.var("input", shape=shape_input),
            relay.var("target", shape=shape_target, dtype="int32"),
            ignore_index=None,
            get_weights=True,
        )
        mod = IRModule.from_expr(result)
        result_weights = (
            relay.create_executor("vm", mod=mod)
            .evaluate()(
                target_np,
                input_np,
            )
            .asnumpy()
        )

        print("RESULT   :", list(result_masked.flatten()[:10]))
        print("UNMASKED :", list(result_unmasked.flatten()[:10]))
        print("MASK     :", list(result_mask.flatten()[:10]))
        print("WEIGHTS  :", list(result_weights.flatten()[:10]))
        print()


def show_gather_negative_indices_fail(negative_indices=False, axis=1):
    # Should give same result with and without negative_indices
    input_shape = [3, 3, 3]
    target_tensor_shape = [3, 3]
    target_tensor_shape.insert(axis, 1)
    input_tensor = relay.var("input_shape", shape=input_shape, dtype="float32")
    target_tensor = relay.var("input_shape", shape=target_tensor_shape, dtype="int32")

    result = relay.gather(
        input_tensor, axis, target_tensor, support_negative_indices=negative_indices
    )
    mod = IRModule.from_expr(result)
    vm_exe = relay.create_executor("vm", mod=mod)

    numpy_input = np.arange(27).reshape(input_shape).astype("float32")

    target_tensor = np.ones(target_tensor_shape).astype("int32")
    if negative_indices:
        target_tensor *= -1
        target_tensor -= 1

    result_unmasked = vm_exe.evaluate()(
        numpy_input,
        target_tensor,
    ).asnumpy()

    print(target_tensor)
    print(result_unmasked)


show_gather_negative_indices_fail(negative_indices=True, axis=0)
show_gather_negative_indices_fail(negative_indices=False, axis=0)
test_failing_onnx_test_case()
