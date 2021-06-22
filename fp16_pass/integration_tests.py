import tempfile
from collections import defaultdict
from os import path
from typing import *

import numpy as np
import onnx
import tensorflow as tf
import torch.onnx
import torchvision
import tvm
import tvm.relay.testing.tf as tf_testing
from tvm import relay
from tvm.driver import tvmc
from tvm.relay.testing import densenet, lstm, mobilenet, resnet, resnet_3d, squeezenet
from tvm.relay.transform import InferType, ToMixedPrecision, mixed_precision

MODELS_DIR = "./models/"


def run_module(mod, mod_params, target="llvm"):
    dev = tvm.device(target, 0)
    intrp = relay.create_executor("debug", mod, device=dev, target=target)
    result = intrp.evaluate()(**mod_params)
    if isinstance(result, tvm.runtime.container.ADT):
        result = [r.asnumpy() for r in result]
        return result
    else:
        return [result.asnumpy()]


def convert_to_list(x):
    if not isinstance(x, list):
        x = [x]
    return x


def run_tf_graph(sess, input_data, input_node, output_node):
    """Generic function to execute tensorflow"""
    input_data = convert_to_list(input_data)
    input_node = convert_to_list(input_node)
    output_node = convert_to_list(output_node)

    tensor = [sess.graph.get_tensor_by_name(output_name) for output_name in output_node]

    input_dict = {e: input_data[i] for i, e in enumerate(input_node)}
    if len(input_node) == 1 and input_node[0] == "":
        output_data = sess.run(tensor)
    else:
        output_data = sess.run(tensor, input_dict)
    return output_data


def verify_fp32_fp16_output_close(mod, mod_params, rtol=1e-3, atol=0, run_opt=True):
    mod = InferType()(mod)
    result_fp32 = run_module(mod, mod_params)

    if run_opt:
        # mod = tvm.relay.transform.FastMath()(mod)
        mod = tvm.relay.transform.EliminateCommonSubexpr()(mod)
        mod = tvm.relay.transform.FoldConstant()(mod)
        mod = tvm.relay.transform.CombineParallelBatchMatmul()(mod)
        mod = tvm.relay.transform.FoldConstant()(mod)

    mod = ToMixedPrecision()(mod)

    if run_opt:
        # run one more pass to clean up new subgraph
        mod = tvm.relay.transform.FastMath()(mod)
        mod = tvm.relay.transform.EliminateCommonSubexpr()(mod)
        mod = tvm.relay.transform.FoldConstant()(mod)
        mod = tvm.relay.transform.CombineParallelBatchMatmul()(mod)
        mod = tvm.relay.transform.FoldConstant()(mod)
    result_fp16 = run_module(mod, mod_params)

    # Ensure the results are close
    for fp32, fp16 in zip(result_fp32, result_fp16):
        np.testing.assert_allclose(fp32, fp16, rtol=rtol, atol=atol)

    return mod


# Native relay models
def test_resnet18():
    np.random.seed(4321)
    mod, mod_params = resnet.get_workload(1, 5, num_layers=18, image_shape=(1, 32, 32))
    mod_params["data"] = np.random.uniform(-10, 10, (1, 1, 32, 32)).astype("float32")

    verify_fp32_fp16_output_close(mod, mod_params)


def test_resnet18_3d():
    np.random.seed(3215)
    mod, mod_params = resnet_3d.get_workload(
        1, 5, num_layers=18, image_shape=(1, 3, 32, 32)
    )
    mod_params["data"] = np.random.uniform(-10, 10, (1, 1, 3, 32, 32)).astype("float32")

    verify_fp32_fp16_output_close(mod, mod_params)


def test_mobilenet():
    np.random.seed(4615)

    mod, mod_params = mobilenet.get_workload(1, 5, image_shape=(1, 32, 32))
    mod_params["data"] = np.random.uniform(-10, 10, (1, 1, 32, 32)).astype("float32")

    verify_fp32_fp16_output_close(mod, mod_params)


def test_densenet():
    np.random.seed(3222)
    mod, mod_params = densenet.get_workload(
        classes=5, batch_size=1, image_shape=(1, 224, 224)
    )
    mod_params["data"] = np.random.uniform(-10, 10, (1, 1, 224, 224)).astype("float32")

    verify_fp32_fp16_output_close(mod, mod_params)


def test_squeezenet():
    np.random.seed(5628)
    mod, mod_params = squeezenet.get_workload(1, 5, image_shape=(1, 32, 32))
    mod_params["data"] = np.random.uniform(-10, 10, (1, 1, 32, 32)).astype("float32")
    verify_fp32_fp16_output_close(mod, mod_params)


# Straight image classification models
def test_onnx_resnet18():
    model_path = path.join(MODELS_DIR, "resnet18-v1-7.onnx")
    # now you have super_resolution.onnx on disk
    onnx_model = onnx.load(model_path)
    mod, mod_params = relay.frontend.from_onnx(onnx_model)
    mod_params["data"] = np.random.uniform(0, 1, size=[1, 3, 224, 224]).astype(
        "float32"
    )
    output_mod = verify_fp32_fp16_output_close(mod, mod_params, atol=0.05, rtol=0.01)
    assert not tvm.ir.structural_equal(mod, output_mod)


def test_onnx_efficientnet():
    model_path = path.join(MODELS_DIR, "efficientnet-lite4-11.onnx")
    # now you have super_resolution.onnx on disk
    onnx_model = onnx.load(model_path)
    mod, mod_params = relay.frontend.from_onnx(onnx_model)
    mod_params["images:0"] = np.random.uniform(0, 1, size=[1, 224, 224, 3]).astype(
        "float32"
    )
    output_mod = verify_fp32_fp16_output_close(mod, mod_params, atol=0.05, rtol=0.01)
    assert not tvm.ir.structural_equal(mod, output_mod)


def test_onnx_densenet():
    model_path = path.join(MODELS_DIR, "densenet-3.onnx")
    # now you have super_resolution.onnx on disk
    onnx_model = onnx.load(model_path)
    mod, mod_params = relay.frontend.from_onnx(onnx_model)
    mod_params["data_0"] = np.random.uniform(0, 1, size=[1, 3, 224, 224]).astype(
        "float32"
    )
    output_mod = verify_fp32_fp16_output_close(mod, mod_params, atol=0.05, rtol=0.01)
    assert not tvm.ir.structural_equal(mod, output_mod)


def test_onnx_inceptionv3():
    model_path = path.join(MODELS_DIR, "inceptionv3.onnx")
    onnx_model = onnx.load(model_path)
    mod, mod_params = relay.frontend.from_onnx(
        onnx_model, shape={"input.1": [1, 3, 299, 299]}
    )
    mod_params["input.1"] = np.random.uniform(0, 1, size=[1, 3, 299, 299]).astype(
        "float32"
    )
    output_mod = verify_fp32_fp16_output_close(mod, mod_params, atol=0.05, rtol=0.01)
    assert not tvm.ir.structural_equal(mod, output_mod)


# Object detection models
def test_onnx_tinyyolo2():
    model_path = path.join(MODELS_DIR, "tinyyolov2-7.onnx")
    onnx_model = onnx.load(model_path)
    mod, mod_params = relay.frontend.from_onnx(
        onnx_model, shape={"image": [1, 3, 416, 416]}
    )
    mod_params["image"] = np.random.uniform(0, 1, size=[1, 3, 416, 416]).astype(
        "float32"
    )
    output_mod = verify_fp32_fp16_output_close(mod, mod_params, atol=0.05, rtol=0.01)
    assert not tvm.ir.structural_equal(mod, output_mod)


def test_onnx_yolo2():
    model_path = path.join(MODELS_DIR, "yolov2-coco-9.onnx")
    onnx_model = onnx.load(model_path)
    mod, mod_params = relay.frontend.from_onnx(
        onnx_model, shape={"input.1": [1, 3, 416, 416]}
    )
    mod_params["input.1"] = np.random.uniform(0, 1, size=[1, 3, 416, 416]).astype(
        "float32"
    )
    output_mod = verify_fp32_fp16_output_close(mod, mod_params, atol=0.05, rtol=0.01)
    assert not tvm.ir.structural_equal(mod, output_mod)


def test_onnx_yolo4():
    model_path = path.join(MODELS_DIR, "yolov4.onnx")
    onnx_model = onnx.load(model_path)
    mod, mod_params = relay.frontend.from_onnx(
        onnx_model, shape={"input_1:0": [1, 416, 416, 3]}, freeze_params=True
    )
    mod_params["input_1:0"] = np.random.uniform(0, 1, size=[1, 416, 416, 3]).astype(
        "float32"
    )
    output_mod = verify_fp32_fp16_output_close(mod, mod_params, atol=0.05, rtol=0.01)
    assert not tvm.ir.structural_equal(mod, output_mod)


# Face recognition / embedding
def test_onnx_arcfaceresnet():
    model_path = path.join(MODELS_DIR, "arcfaceresnet100-8.onnx")
    onnx_model = onnx.load(model_path)
    mod, mod_params = relay.frontend.from_onnx(onnx_model)
    mod_params["data"] = np.random.uniform(0, 1, size=[1, 3, 112, 112]).astype(
        "float32"
    )
    output_mod = verify_fp32_fp16_output_close(mod, mod_params, atol=0.05, rtol=0.01)
    assert not tvm.ir.structural_equal(mod, output_mod)


def test_onnx_rfb():
    model_path = path.join(MODELS_DIR, "version-RFB-320.onnx")
    onnx_model = onnx.load(model_path)
    mod, mod_params = relay.frontend.from_onnx(onnx_model)
    mod_params["input"] = np.random.uniform(0, 1, size=[1, 3, 240, 320]).astype(
        "float32"
    )
    output_mod = verify_fp32_fp16_output_close(mod, mod_params, atol=0.05, rtol=0.01)
    assert not tvm.ir.structural_equal(mod, output_mod)


# Super resolution
def test_onnx_superresolution():
    model_path = path.join(MODELS_DIR, "super-resolution-10.onnx")
    onnx_model = onnx.load(model_path)
    mod, mod_params = relay.frontend.from_onnx(
        onnx_model, shape={"input": [1, 1, 224, 224]}
    )
    mod_params["input"] = np.random.uniform(0, 1, size=[1, 1, 224, 224]).astype(
        "float32"
    )
    output_mod = verify_fp32_fp16_output_close(mod, mod_params, atol=0.05, rtol=0.01)
    assert not tvm.ir.structural_equal(mod, output_mod)


# NLP models (ruh roh!)
def test_onnx_gpt2():
    model_path = path.join(MODELS_DIR, "gpt2-10.onnx")
    onnx_model = onnx.load(model_path)

    mod, mod_params = relay.frontend.from_onnx(onnx_model, shape={"input1": [1, 1, 1]})
    mod_params["input1"] = np.random.randint(0, 100, size=[1, 1, 1])
    output_mod = verify_fp32_fp16_output_close(mod, mod_params, atol=0.05, rtol=0.01)
    assert not tvm.ir.structural_equal(mod, output_mod)


def test_onnx_distillbert():
    model_path = path.join(MODELS_DIR, "distilbert.onnx")
    onnx_model = onnx.load(model_path)

    mod, mod_params = relay.frontend.from_onnx(onnx_model, shape={"input.1": [10, 100]})
    mod_params["input.1"] = np.random.randint(0, 100, size=[10, 100])
    output_mod = verify_fp32_fp16_output_close(mod, mod_params, atol=0.05, rtol=0.01)
    assert not tvm.ir.structural_equal(mod, output_mod)


def test_pb_bert():
    tvmc_model = tvmc.load(path.join(MODELS_DIR, "bert-base-uncased.pb"))
    mod, mod_params = tvmc_model.mod, tvmc_model.params
    # Weird functions we don't use are in there it's weird
    mod = tvm.IRModule.from_expr(mod["main"])
    mod_params["x"] = np.random.randint(0, 100, size=[1, 128]).astype("int32")
    output_mod = verify_fp32_fp16_output_close(
        mod, mod_params, atol=0.05, rtol=0.01, run_opt=True
    )
    assert not tvm.ir.structural_equal(mod, output_mod)


def test_onnx_ssd():
    tvmc_model = tvmc.load(path.join(MODELS_DIR, "ssd-10.onnx"))
    mod, mod_params = tvmc_model.mod, tvmc_model.params
    # Weird functions we don't use are in there it's weird
    mod = tvm.IRModule.from_expr(mod["main"])
    mod_params["image"] = np.random.uniform(-1, 1, size=[1, 3, 1200, 1200]).astype(
        "float32"
    )
    # TODO: this works but the threshold for scores is too low where there are errors
    output_mod = verify_fp32_fp16_output_close(mod, mod_params, atol=0.05, rtol=0.01)
    assert not tvm.ir.structural_equal(mod, output_mod)


def test_tf_ssd_impl():
    """Test SSD with backbone MobileNet V1"""
    # SHOULD FAIL UNTIL ADT types are supported
    with tf.Graph().as_default():
        graph_def = tf_testing.get_workload(
            "object_detection/ssd_mobilenet_v1_ppn_shared_"
            "box_predictor_300x300_coco14_sync_2018_07_03.pb"
        )
        # Call the utility to import the graph definition into default graph.
        graph_def = tf_testing.ProcessGraphDefParam(graph_def)

    data = [np.random.uniform(0.0, 255.0, size=(1, 512, 512, 3)).astype("uint8")]
    in_node = ["image_tensor"]
    out_node = ["detection_boxes", "detection_scores", "detection_classes"]

    shape_dict = {
        e: i.shape if hasattr(i, "shape") else () for e, i in zip(in_node, data)
    }
    mod, mod_params = relay.frontend.from_tensorflow(
        graph_def, layout="NCHW", shape=shape_dict, outputs=out_node
    )
    mod_params[in_node[0]] = data[0]
    output_mod = verify_fp32_fp16_output_close(mod, mod_params, atol=0.05, rtol=0.01)
    assert not tvm.ir.structural_equal(mod, output_mod)
