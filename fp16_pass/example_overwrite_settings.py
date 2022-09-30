"""Example of overwriting FQ2I settings for an operation."""

import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relay.op import register_mixed_precision_conversion


def custom_op_rewrite(call_node: relay.Call, dtype: str):
    return [
        transform.mixed_precision.MIXED_PRECISION_FOLLOW,
        dtype,  # accumulation dtype
        dtype,  # output dtype
    ]


def example_model():
    input_var = relay.var("input_var", dtype="float32", shape=[512, 512])
    matmul_with_self = relay.nn.dense(input_var, input_var)
    result = relay.mean(matmul_with_self, -1, keepdims=True) + input_var
    return tvm.IRModule.from_expr(result)


if __name__ == "__main__":
    mod = example_model()
    mod = relay.transform.InferType()(mod)

    print("FP32 models")
    print(mod)

    print("FP16 models default")
    print(transform.ToMixedPrecision()(mod))

    # Set the level to anything higher than the default (10) to be used
    register_mixed_precision_conversion("mean", custom_op_rewrite, level=11)

    print("FP16 models with modified lists")
    print(transform.ToMixedPrecision()(mod))
