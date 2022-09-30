"""Demonstrates issue with FQ2I pass which duplicates some operations."""

from tvm import relay
from tvm import IRModule
import tvm

# scale and zero point for hte layer
SF = relay.const(0.025, dtype="float32")
ZP = relay.const(1, dtype="int32")


def relay_special_residual_layer_fake_quantized(
    input_var: relay.Var,
):
    qdq_input = relay.qnn.op.dequantize(input_var, SF, ZP)

    # Potentially expensive op!
    expensive_mean = relay.mean(qdq_input, axis=-1, keepdims=True)

    # To preserve shape
    diamond_mul = relay.subtract(expensive_mean, qdq_input)
    sqrt = relay.sqrt(diamond_mul)

    q_diamond = relay.qnn.op.quantize(sqrt, SF, ZP)
    qdq_diamond = relay.qnn.op.dequantize(q_diamond, SF, ZP)

    # This is the problematic operation, in subgraph extraction, we don't have
    # a handle into the quantized version of the previous nodes...
    # This eventually duplicates the expensive operations in the diamond
    evil_add = relay.add(qdq_diamond, sqrt)
    evil_add = relay.qnn.op.quantize(evil_add, SF, ZP)
    return evil_add


def expected_result_quantized(input_var: relay.Var):
    # input_var shoudl already be quantized
    cast_int32 = relay.cast(input_var, "int32")
    mean_int32 = relay.mean(cast_int32, axis=-1, keepdims=True)
    mean_result = relay.cast(mean_int32, "int8")

    mul_result = relay.qnn.op.mul(mean_result, input_var, SF, ZP, SF, ZP, SF, ZP)
    sqrt_result = relay.qnn.op.sqrt(mul_result, SF, ZP, SF, ZP)
    add = relay.qnn.op.add(mul_result, sqrt_result, SF, ZP, SF, ZP, SF, ZP)
    return add


def generate_evil_model(shape=[1, 512, 512], layers=10) -> IRModule:
    input_var = relay.var("input_var", shape=shape)
    input_var_quantized = relay.qnn.op.quantize(input_var, SF, ZP)

    cur_last_expr = input_var_quantized
    for _ in range(layers):
        cur_last_expr = relay_special_residual_layer_fake_quantized(cur_last_expr)

    mod = IRModule.from_expr(cur_last_expr)
    mod = relay.transform.InferType()(mod)
    return mod


def generated_expected_model(shape=[1, 512, 512], layers=10) -> IRModule:
    input_var = relay.var("input_var", shape=shape)
    input_var_quantized = relay.qnn.op.quantize(input_var, SF, ZP)

    cur_last_expr = input_var_quantized
    for _ in range(layers):
        cur_last_expr = expected_result_quantized(cur_last_expr)

    mod = IRModule.from_expr(cur_last_expr)
    mod = relay.transform.InferType()(mod)
    return mod


def run_fq2i(mod: IRModule) -> IRModule:
    passes = []

    # Infer types prior to the quantization pass below as some
    # transforms might need them.
    passes.append(relay.transform.InferType())

    # Transform fake quantized sub-graphs to actual integer ops.
    # Should have no effect on graphs without the relevant patterns.
    passes.append(
        relay.transform.FakeQuantizationToInteger(hard_fail=False, use_qat=False)
    )

    # Fold constants after FQ2I becuase some weights are stored in FP32.
    passes.append(relay.transform.FoldConstant())
    passes.append(relay.transform.EliminateCommonSubexpr())
    seq = tvm.ir.transform.Sequential(passes)
    return seq(mod)


if __name__ == "__main__":
    mod = generate_evil_model(layers=3)
    expected_mod = generated_expected_model(layers=3)

    # This should duplicate the diamond branch :'(
    print("BEFORE")
    print(mod)
    print()
    print("AFTER")
    print(run_fq2i(mod))
    print("Expected")
    print(expected_mod)