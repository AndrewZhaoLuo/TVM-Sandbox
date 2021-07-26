import os
import random
import tensorflow as tf
import tensorflow.keras as keras

SEED = 42
random.seed(SEED)
tf.random.set_seed(SEED)

model = keras.models.Sequential()
model.add(
    keras.layers.Conv2D(
        filters=1,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        input_shape=(3, 3, 1),
    )
)
kmodel_fn = "/tmp/c1mdl.h5"
model.save(kmodel_fn)

import coremltools as ct

mdl = ct.convert(kmodel_fn)
model_file = "/tmp/c1.mlmodel"
mdl.save(model_file)

mdl = ct.models.MLModel(model_file)
desc = mdl.get_spec().description
iname = desc.input[0].name
ishape = desc.input[0].type.multiArrayType.shape
shape_dict = {}
for i in mdl.get_spec().description.input:
    iname = i.name
    ishape = i.type.multiArrayType.shape
    shape_dict[iname] = ishape
print("shape dictionary:", shape_dict)

import tvm
from tvm import te
import tvm.micro as micro
import tvm.relay as relay

mod, params = relay.frontend.from_coreml(mdl, shape_dict)

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, "llvm", params=params)
