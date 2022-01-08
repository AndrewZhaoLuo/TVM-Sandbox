from typing import Callable

import numpy as np
import tvm
from tvm import relay
from tvm.relay.op.transform import gather_nd

# Example of using gather as a lookup table


def run_const_expr(expr):
    mod = tvm.IRModule.from_expr(expr)
    vm_exe = relay.create_executor("vm", mod=mod)
    return vm_exe.evaluate()().asnumpy()


def create_integer_lookup_table(
    floating_point_func: Callable[[np.ndarray], np.ndarray],
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    axis=-1,
    dtype="uint8",
):
    if not np.issubdtype(np.dtype(dtype), np.integer):
        raise ValueError(f"Only integer dtypes allowed got {dtype}")

    dtype_info = np.iinfo(dtype)

    # Use TVMs quantization methods via relay to be consistent
    inputs_quantized = np.array(range(dtype_info.min, dtype_info.max + 1)).astype(dtype)
    inputs_quantized = relay.const(inputs_quantized, dtype=dtype)
    inputs_dequantized = run_const_expr(
        relay.qnn.op.dequantize(
            inputs_quantized,
            input_scale=input_scale,
            input_zero_point=input_zero_point,
            axis=axis,
        )
    )

    output_dequantized = relay.const(floating_point_func(inputs_dequantized))
    output_quantized = run_const_expr(
        relay.qnn.op.quantize(
            output_dequantized, output_scale, output_zero_point, axis, dtype
        )
    )

    return output_quantized


def lookup_quantized(
    input_tensor,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    func,
    axis=-1,
    dtype="uint8",
):
    input_scale = relay.const(input_scale)
    input_zero_point = relay.const(input_zero_point, dtype="int32")
    output_scale = relay.const(output_scale)
    output_zero_point = relay.const(output_zero_point, dtype="int32")

    lookup_table = create_integer_lookup_table(
        func,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        axis=axis,
        dtype=dtype,
    )
    lookup_table = relay.const(lookup_table)

    index_tensor = relay.reshape(input_tensor, [-1])
    result = relay.gather(lookup_table, -1, index_tensor)
    return relay.reshape_like(result, input_tensor)


if __name__ == "__main__":
    # func = lambda x: 1 / (1 + np.exp(-x))
    func = np.tanh

    # Tanh saturates pretty quickly, |tanh(+-2)| ~= 0.96
    orig_input_tensor = np.random.uniform(low=-2, high=2, size=(1, 3, 50, 50))
    input_scale = 4 / 256
    input_zero_point = 128  # for symmetric quantization

    input_tensor = relay.const(
        (orig_input_tensor / input_scale).astype("uint8") + input_zero_point,
        dtype="uint8",
    )

    output_scale = 2 / 256
    output_zero_point = 128

    result = lookup_quantized(
        input_tensor,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        func,
        dtype="uint8",
    )

    output_tensor = run_const_expr(result).astype("int32")
    output_tensor_float = (output_tensor - output_zero_point) * output_scale
    print(
        f"Is close: {np.allclose(output_tensor_float, func(orig_input_tensor), rtol=1e-2, atol=5e-2)}"
    )

    max_difference = np.max(np.abs((output_tensor_float - func(orig_input_tensor))))
    print(max_difference)
