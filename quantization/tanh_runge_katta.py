# Using runge katta to approx tanh(x)
import math


def one_iteration(y_n, step_size):
    k1 = 1 - y_n ** 2
    k2 = 1 - (y_n + k1 / 2 * step_size) ** 2
    k3 = 1 - (y_n + k2 / 2 * step_size) ** 2
    k4 = 1 - (y_n + k3 * step_size) ** 2
    y_np1 = y_n + (k1 + k2 * 2 + k3 * 2 + k4) / 6 * step_size
    return y_np1


x = 0
y = 0
step = 1 / 128
for i in range(1000):
    y = one_iteration(y, step)
    print(x, ":", y, math.tanh(x))
    x += step
