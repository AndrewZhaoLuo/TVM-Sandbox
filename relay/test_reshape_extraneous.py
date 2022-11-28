"""Testing shape inference with reshape"""

import tvm
from tvm import relay


def view_shape_inference():
    input_var = relay.var("input", shape=[128, 2, 2, 2])

    # If shape inference works for reshape, resulting mod should print following shapes:
    # Standard case
    reshaped0 = relay.reshape(input_var, [2, 2, 256])  # 2, 2, 256

    # 0 --> copy dimension from original shape
    reshaped1 = relay.reshape(input_var, [0, 4, 0])  # 128, 4, 2

    # -1 --> infer the rest of dimensions from given info
    reshaped2 = relay.reshape(input_var, [2, 2, -1])  # 2, 2, 256

    # -2 --> copy remainder dimensions from original shape
    reshaped3 = relay.reshape(input_var, [64, 4, -2])  # 64, 4, 2, 2

    # -3 --> use product of next dimensions as output
    reshaped4 = relay.reshape(input_var, [-3, 2, 2])  # 256, 2, 2

    # -4 --> split dimension to next two dimensions
    reshaped5 = relay.reshape(input_var, [-4, 32, 4, 8])  # 32, 4, 8

    result = relay.Tuple(
        [reshaped0, reshaped1, reshaped2, reshaped3, reshaped4, reshaped5]
    )

    mod = tvm.IRModule.from_expr(result)
    mod = relay.transform.InferType()(mod)

    print(mod)


if __name__ == "__main__":
    input_var = relay.var("input", shape=[128, 2, 2, 2])

    # Standard case
    reshaped0 = relay.reshape(input_var, [128, 2, 2, 2])

    # 0 --> copy dimension from original shape
    reshaped1 = relay.reshape(input_var, [0, 2, 0, 0])

    # -1 --> infer the rest of dimensions from given info
    reshaped2 = relay.reshape(input_var, [-1, 2, 2, 2])

    # -2 --> copy remainder dimensions from original shape
    reshaped3 = relay.reshape(input_var, [128, 2, -2])

    result = relay.Tuple([reshaped0, reshaped1, reshaped2, reshaped3])

    mod = tvm.IRModule.from_expr(result)
    mod = relay.transform.InferType()(mod)

    """
    Before:
    def @main(%input: Tensor[(128, 2, 2, 2), float32] /* ty=Tensor[(128, 2, 2, 2), float32] */) -> (Tensor[(128, 2, 2, 2), float32], Tensor[(128, 2, 2, 2), float32], Tensor[(128, 2, 2, 2), float32], Tensor[(128, 2, 2, 2), float32]) {
        %0 = reshape(%input, newshape=[128, 2, 2, 2]) /* ty=Tensor[(128, 2, 2, 2), float32] */;
        %1 = reshape(%input, newshape=[0, 2, 0, 0]) /* ty=Tensor[(128, 2, 2, 2), float32] */;
        %2 = reshape(%input, newshape=[-1, 2, 2, 2]) /* ty=Tensor[(128, 2, 2, 2), float32] */;
        %3 = reshape(%input, newshape=[128, 2, -2]) /* ty=Tensor[(128, 2, 2, 2), float32] */;
        (%0, %1, %2, %3) /* ty=(Tensor[(128, 2, 2, 2), float32], Tensor[(128, 2, 2, 2), float32], Tensor[(128, 2, 2, 2), float32], Tensor[(128, 2, 2, 2), float32]) */
    }
    """
    print(mod)

    mod = relay.transform.SimplifyExpr()(mod)

    """
    After:
    def @main(%input: Tensor[(128, 2, 2, 2), float32] /* ty=Tensor[(128, 2, 2, 2), float32] */) -> (Tensor[(128, 2, 2, 2), float32], Tensor[(128, 2, 2, 2), float32], Tensor[(128, 2, 2, 2), float32], Tensor[(128, 2, 2, 2), float32]) {
        (%input, %input, %input, %input) /* ty=(Tensor[(128, 2, 2, 2), float32], Tensor[(128, 2, 2, 2), float32], Tensor[(128, 2, 2, 2), float32], Tensor[(128, 2, 2, 2), float32]) */
    }
    """
    print(mod)
