"""
Dumps BERT pb used for tuning.
"""

import os
import time

import click
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)


def measure(func, args, repeats=50):
    res = []
    for _ in range(repeats):
        start = time.time()
        func(args)
        end = time.time()
        res.append((end - start) * 1000.0)
    return np.mean(res), np.std(res)


def _load_keras_model(module, name, seq_len, batch_size):
    model = module.from_pretrained(name)
    dummy_input = tf.keras.Input(shape=[seq_len], batch_size=batch_size, dtype="int32")
    model(dummy_input)  # Propagate shapes through the keras model.
    return model


def keras_to_graphdef(model, batch_size, seq_len):
    model_func = tf.function(lambda x: model(x))
    input_dict = model._saved_model_inputs_spec
    input_spec = input_dict[list(input_dict.keys())[0]]
    model_func = model_func.get_concrete_function(
        tf.TensorSpec([batch_size, seq_len], input_spec.dtype)
    )
    frozen_func = convert_variables_to_constants_v2(model_func)
    return frozen_func.graph.as_graph_def()


def get_huggingface_model(name, batch_size, seq_len):
    import transformers

    module = getattr(transformers, "TFBertForSequenceClassification")
    model = _load_keras_model(module, name=name, batch_size=batch_size, seq_len=seq_len)
    return model


def save_model_pb(graphdef, name, prefix="./models"):
    tf.io.write_graph(graphdef, prefix, name.replace("/", "_") + ".pb", False)


@click.command()
@click.option("--model-name", default="bert-base-uncased", help="name of model")
@click.option("--save-prefix", default="./models")
def main(model_name, save_prefix):
    batch_size = 1
    seq_len = 128
    model = get_huggingface_model(model_name, batch_size, seq_len)
    graphdef = keras_to_graphdef(model, batch_size, seq_len)
    save_model_pb(graphdef, model_name, save_prefix)


if __name__ == "__main__":
    main()
