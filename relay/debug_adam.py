import math
import pickle

import numpy as np
import tvm
from tvm import relay

"""Based off test_adam_multiple here:
https://github.com/onnx/onnx/blob/07c494bf077e9e4a7898119f28a50585469ad4cd/onnx/backend/test/case/node/adam.py#L16
"""
inputs = pickle.load(open("./relay/inputs3.pkl", "rb"))


def apply_adam(r, t, x, g, v, h, norm_coefficient, norm_coefficient_post, alpha, beta, epsilon):  # type: ignore
    # Add gradient of regularization term.
    g_regularized = norm_coefficient * x + g
    # Update momentum.
    v_new = alpha * v + (1 - alpha) * g_regularized
    # Update second-order momentum.
    h_new = beta * h + (1 - beta) * (g_regularized * g_regularized)
    # Compute element-wise square root.
    h_sqrt = np.sqrt(h_new) + epsilon
    # Adjust learning rate.
    r_adjusted = None
    if t > 0:
        # Consider bias correction on momentums.
        r_adjusted = r * np.sqrt(1 - beta ** t) / (1 - alpha ** t)
    else:
        # No bias correction on momentums.
        r_adjusted = r
    # Apply Adam update rule.
    x_new = x - r_adjusted * (v_new / h_sqrt)
    # It's possible to apply regularization in the end.
    x_final = (1 - norm_coefficient_post) * x_new
    return (np.sqrt(h_new)), v_new, h_new


def get_numbers():
    r = inputs[0]
    t = inputs[1]

    first_batch = inputs[2::2]
    second_batch = inputs[3::2]
    norm_coefficient = 0.001
    norm_coefficient_post = 0.0
    alpha = 0.95
    beta = 0.85
    epsilon = 1e-2

    args1 = (
        [r]
        + [t]
        + first_batch
        + [norm_coefficient, norm_coefficient_post, alpha, beta, epsilon]
    )
    args2 = (
        [r]
        + [t]
        + second_batch
        + [norm_coefficient, norm_coefficient_post, alpha, beta, epsilon]
    )

    print(apply_adam(*args1))
    print(apply_adam(*args2))


if __name__ == "__main__":
    x = relay.var("x", shape=[])
    result = relay.sqrt(x)
    mod = tvm.IRModule.from_expr(result)
    vm_exe = relay.create_executor("vm", mod=mod)

    for i in range(100):
        import numpy.random

        numpy_input = np.random.uniform(0, 1)

        result = vm_exe.evaluate()(
            numpy_input,
        ).asnumpy()
        print(result)
        print(np.sqrt(numpy_input))

        get_numbers()
