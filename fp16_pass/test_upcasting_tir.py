"""
This file tests how upcasting works in TVM
"""
import tvm
from tvm import tir
import numpy as np

def try_to_add(lhs_dtype, rhs_dtype):
    lhs = tir.Var('lhs', lhs_dtype)
    rhs = tir.Var('rhs', rhs_dtype)
    return lhs + rhs 

if __name__ == "__main__":
    # Mixing integral and float is ok!
    try_to_add('float32', 'float32')
    try_to_add('float32', 'int8')
    try_to_add('float32', 'int16')
    try_to_add('float32', 'int32')
    try_to_add('float32', 'int64')
    try_to_add('float16', 'float16')
    try_to_add('float16', 'int8')
    try_to_add('float16', 'int16')
    try_to_add('float16', 'int32')
    try_to_add('float16', 'int64')

    # Oh no!
    try_to_add('float32', 'float16')
    print('All pass!')