# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import multiprocessing as mp
import os
from os import path

import numpy as np
import pytest
import tvm
from tvm import relay
from tvm.driver import tvmc
from tvm.driver.tvmc.model import TVMCModel, TVMCPackage, TVMCResult
from tvm.relay.transform import AMPRewrite
from tvm.relay.transform.transform import InferType


def get_one_conv_model(N=4, C_I=16, C_O=64, H=224, W=224, K=3, dtype="float32"):
    data_shape = [N, C_I, H, W]
    kernel_shape = [C_O, C_I, K, K]
    data = relay.var("data", dtype=dtype, shape=data_shape)
    weight = relay.var("weight", dtype=dtype, shape=kernel_shape)
    conv = relay.nn.conv2d(
        data=data,
        weight=weight,
        padding=(1, 1),
        data_layout="NCHW",
        kernel_layout="OIHW",
        kernel_size=K,
        channels=C_O,
    )
    mod = tvm.IRModule.from_expr(conv)
    params = {
        # "data": np.random.uniform(-1, 1, size=data_shape),
        "weight": np.random.uniform(-1, 1, size=kernel_shape).astype(dtype),
    }
    return tvmc.TVMCModel(mod, params)


def get_distillbert(run_pass=True):
    tvmc_model = tvmc.load("/Users/andrewzhaoluo/Downloads/distilbert.onnx")
    if run_pass:
        mod, params = tvmc_model.mod, tvmc_model.params
        fp16_mod = AMPRewrite()(mod)
        return TVMCModel(fp16_mod, params)
    return tvmc_model


def get_bert(run_pass=True):
    tvmc_model = tvmc.load("./models/bert-base-uncased.pb")
    mod, params = tvmc_model.mod, tvmc_model.params
    # Weird functions we don't use are in there it's weird
    mod = tvm.IRModule.from_expr(mod["main"])
    if run_pass:
        mod = InferType()(mod)
        fp16_mod = AMPRewrite()(mod)
        return TVMCModel(fp16_mod, params)
    return TVMCModel(mod, params)


if __name__ == "__main__":
    # macOS has 'spawn' as default which doesn't work for tvm
    mp.set_start_method("fork")

    """Get Module"""
    # tvmc_model = get_one_conv_model(dtype="float16")
    # tvmc_model = get_one_conv_model(dtype="float32")
    # tvmc_model = get_distillbert()
    tvmc_model = get_bert(run_pass=False)
    tvmc_model.summary()

    # Create tuning artifacts
    target = "llvm -mcpu=apple-latest -mtriple=arm64-apple-macos"
    tuning_records = tvmc.tune(tvmc_model, target=target, trials=1000)

    # Create package artifacts
    package = tvmc.compile(tvmc_model, target=target, tuning_records=tuning_records)
    result = tvmc.run(package, device="cpu", repeat=100, number=10)
    print(result)
