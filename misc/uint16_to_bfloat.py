import numpy as np
from rsa import sign

"""
Anatomy of bfloat 16:

SEEE EEEE EMMM MMMM

Anatomy of float 16

SEEE EEMM MMMM MMMM

S = sign bit
E = exponent bits
M = mantissa bits 
"""


def manual_reinterpret_float16(arr: np.ndarray):
    # Does not handle denormals or special floating point values (like 0 lol)
    sign_bits = (arr >> 15).astype("int16")
    exponent_bits = ((arr >> 10) & 0x1F).astype("int16")
    mantissa_bits = (arr & 0x3FF).astype("uint16")

    exponent = (exponent_bits - 15).astype("float64")
    mantissa = 1 + mantissa_bits.astype("float64") / (2 ** 10)
    result = (2 ** exponent) * mantissa

    signs = (sign_bits + (sign_bits - 1)) * -1
    result *= signs
    return result


def manual_reinterpret_bfloat16(arr: np.ndarray):
    # Does not handle denormals or special floating point values (like 0 lol)
    sign_bits = (arr >> 15).astype("int16")
    exponent_bits = ((arr >> 7) & 0xFF).astype("int16")
    mantissa_bits = (arr & 0x7F).astype("uint16")

    exponent = (exponent_bits - 127).astype("float64")
    mantissa = 1 + mantissa_bits.astype("float64") / (2 ** 10)
    result = (2 ** exponent) * mantissa

    signs = (sign_bits + (sign_bits - 1)) * -1
    result *= signs
    return result

if __name__ == "__main__":
    r = np.array([0x0400, 0x3555, 0x3BFF, 0x3C00, 0x3C01, 0x7BFF, 0xC000]).astype("uint16")

    # Matches values here https://en.wikipedia.org/wiki/Half-precision_floating-point_format
    print("uint16:", r)
    print("float16:", r.view("float16"))
    print("float16:", manual_reinterpret_float16(r))

    # Matches values here: https://en.wikipedia.org/wiki/Bfloat16_floating-point_format
    r = np.array([0x7F7F, 0x0080, 0x3F80, 0xC000]).astype("uint16")
    print("bfloat16:", manual_reinterpret_bfloat16(r))
