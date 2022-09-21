"""Demonstrates issue with FQ2I pass which duplicates some operations."""

from tvm import relay
from tvm import IRModule
import tvm


def relay_special_residual_layer_fake_quantized(
    input_var: relay.Var,
):
    # scale and zero point for hte layer
    sf = relay.const(0.025, dtype="float32")
    zp = relay.const(1, dtype="int32")

    q_input = relay.qnn.op.quantize(input_var, sf, zp)
    qdq_input = relay.qnn.op.dequantize(q_input, sf, zp)

    # Potentially expensive op!
    expensive_mean = relay.mean(qdq_input, axis=-1, keepdims=True)

    # To preserve shape
    diamond_mul = relay.multiply(expensive_mean, qdq_input)

    q_diamond = relay.qnn.op.quantize(diamond_mul, sf, zp)
    qdq_diamond = relay.qnn.op.dequantize(q_diamond, sf, zp)

    # This is the problematic operation, in subgraph extraction, we don't have
    # a handle into the quantized version of the previous nodes...
    # This eventually duplicates the expensive operations in the diamond
    evil_add = relay.add(qdq_diamond, diamond_mul)
    return evil_add


def generate_evil_model(shape=[1, 512, 512], layers=10) -> IRModule:
    input_var = relay.var("input_var", shape=shape)

    cur_last_expr = input_var
    for _ in range(layers):
        cur_last_expr = relay_special_residual_layer_fake_quantized(cur_last_expr)

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
    seq = tvm.ir.transform.Sequential(passes)
    return seq(mod)


if __name__ == "__main__":
    mod = generate_evil_model(layers=1)

    # This should duplicate the diamond branch :'(
    print("BEFORE")
    print(mod)
    print()
    print("AFTER")
    print(run_fq2i(mod))