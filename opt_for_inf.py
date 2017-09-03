"""Removes parts of a graph that are only needed for training.

May significantly reduces size of trained model.

Basically this is reconstruction for in-python usage of:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/optimize_for_inference.py

Example 1 (input=frozen_graph):

python OptimizeForInference.py \
--input=/home/nauris/Dropbox/coding/squalio/car_license_plate/Recognizer/models/freeze_fcn8s_vgg.pb \
--output=/home/nauris/Dropbox/coding/squalio/car_license_plate/Recognizer/models/optimized_fcn8s_vgg.pb \
--frozen_graph=True \
--intput_names=lock/ExpandDims \
--output_names=lock/ArgMax

Example 2 (input=text GraphDef proto file):

python OptimizeForInference.py \
--input=/home/nauris/Downloads/ModelsArchive/CAR_TYPES.pbtxt \
--output=/home/nauris/Downloads/ModelsArchive/optimized_CAR_TYPES.pbtxt \
--frozen_graph=False \
--intput_names=lock/ExpandDims \
--output_names=lock/ArgMax

Run directly from Tensorflow repository:

python ./tensorflow/tensorflow/python/tools/optimize_for_inference.py \
--input=/home/nauris/Dropbox/coding/squalio/car_license_plate/Recognizer/models/freeze_fcn8s_vgg.pb \
--output=/home/nauris/Dropbox/coding/squalio/car_license_plate/Recognizer/models/optimized_fcn8s_vgg.pb \
--frozen_graph=True \
--input_names=lock/ExpandDims \
--output_names=lock/ArgMax
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Standard imports
import argparse
import os
import sys

# Dependecy Imports
from google.protobuf import text_format

from tensorflow.core.framework import graph_pb2 # pylint: disable=E0611
from tensorflow.python.framework import dtypes # pylint: disable=E0611
from tensorflow.python.framework import graph_io # pylint: disable=E0611
from tensorflow.python.platform import app # pylint: disable=E0611
from tensorflow.python.platform import gfile # pylint: disable=E0611
from tensorflow.python.tools import optimize_for_inference_lib # pylint: disable=E0611

FLAGS = None

def main(unused_args):
    """Run."""

    if not gfile.Exists(FLAGS.input):
        print("Input graph file '" + FLAGS.input + "' does not exist!")
        return -1

    input_graph_def = graph_pb2.GraphDef()
    with gfile.Open(FLAGS.input, "rb") as _file:
        data = _file.read()
        if FLAGS.frozen_graph:
            input_graph_def.ParseFromString(data)
        else:
            text_format.Merge(data.decode("utf-8"), input_graph_def)

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def,
        FLAGS.input_names.split(","),
        FLAGS.output_names.split(","), FLAGS.placeholder_type_enum
    )

    if FLAGS.frozen_graph:
        _file = gfile.FastGFile(FLAGS.output, "w")
        _file.write(output_graph_def.SerializeToString())
    else:
        graph_io.write_graph(output_graph_def,
                             os.path.dirname(FLAGS.output),
                             os.path.basename(FLAGS.output))

    return 0


def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="TensorFlow \'GraphDef\' file to load.")
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="File to save the output graph to.")
    parser.add_argument(
        "--input_names",
        type=str,
        default="",
        help="Input node names, comma separated.")
    parser.add_argument(
        "--output_names",
        type=str,
        default="",
        help="Output node names, comma separated.")
    parser.add_argument(
        "--frozen_graph",
        nargs="?",
        const=True,
        type="bool",
        default=True,
        help="""\
        If true, the input graph is a binary frozen GraphDef
        file; if false, it is a text GraphDef proto file.\
        """)
    parser.add_argument(
        "--placeholder_type_enum",
        type=int,
        default=dtypes.float32.as_datatype_enum,
        help="The AttrValue enum to use for placeholders.")

    return parser.parse_known_args()


if __name__ == "__main__":
    FLAGS, UNPARSED = parse_args()
    app.run(main=main, argv=[sys.argv[0]] + UNPARSED)
