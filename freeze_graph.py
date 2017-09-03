"""We save out the graph to disk, and then call the const conversion routine

Before creating FreezeGraph you need to create graph description file (pbtxt). This
can be done right after restoring symbolic graph with TF and calling following cmd:

tf.train.write_graph(
    sess.graph_def,
    '/path/to/model/dir/',
    'model_txt_description.pbtxt'
)

More info about this (Step 3) and more related to Freezing Graph:

https://medium.com/@hamedmp/exporting-trained-tensorflow-models-to-c-the-right-way-cf24b609d183

If you don't know what is Tensorflow node name (output_node_names) then you can check
this in newly created model_txt_description.pbtxt which is txt file. You need to search
for node which output you need for inference.

Basic usage:

python FreezeGraph.py \
--input_graph=/home/nauris/Dropbox/coding/squalio/car_license_plate/Recognizer/models/fcn8s_vgg.pbtxt \
--output_graph=/home/nauris/Dropbox/coding/squalio/car_license_plate/Recognizer/models/freeze_fcn8s_vgg.pb \
--input_checkpoint=/home/nauris/Dropbox/coding/squalio/car_license_plate/Recognizer/models/CAR_TYPES.chk \
--output_node_names=lock/ArgMax

"""

# Standard imports
import argparse

# Dependecy imports
from tensorflow.python.tools import freeze_graph  # pylint: disable=E0611


if __name__ == "__main__":

    PARSER = argparse.ArgumentParser()
    PARSER.register("type", "bool", lambda v: v.lower() == "true")
    PARSER.add_argument(
        "--input_graph",
        type=str,
        default="",
        help="TensorFlow \'GraphDef\' file to load.")
    PARSER.add_argument(
        "--input_saver",
        type=str,
        default="",
        help="TensorFlow saver file to load.")
    PARSER.add_argument(
        "--input_checkpoint",
        type=str,
        default="",
        help="TensorFlow variables file to load.")
    PARSER.add_argument(
        "--output_graph",
        type=str,
        default="",
        help="Output \'GraphDef\' file name.")
    PARSER.add_argument(
        "--input_binary",
        nargs="?",
        const=True,
        type="bool",
        default=False,
        help="Whether the input files are in binary format.")
    PARSER.add_argument(
        "--output_node_names",
        type=str,
        default="",
        help="The name of the output nodes, comma separated.")
    PARSER.add_argument(
        "--restore_op_name",
        type=str,
        default="save/restore_all",
        help="The name of the master restore operator.")
    PARSER.add_argument(
        "--filename_tensor_name",
        type=str,
        default="save/Const:0",
        help="The name of the tensor holding the save path.")
    PARSER.add_argument(
        "--clear_devices",
        nargs="?",
        const=True,
        type="bool",
        default=True,
        help="Whether to remove device specifications.")
    PARSER.add_argument(
        "--initializer_nodes",
        type=str,
        default="",
        help="comma separated list of initializer nodes to run before freezing.")
    PARSER.add_argument(
        "--variable_names_blacklist",
        type=str,
        default="",
        help="""\
        comma separated list of variables to skip converting to constants\
        """)
    FLAGS, _ = PARSER.parse_known_args()

    freeze_graph.freeze_graph(
        FLAGS.input_graph,
        FLAGS.input_saver,
        FLAGS.input_binary,
        FLAGS.input_checkpoint,
        FLAGS.output_node_names,
        FLAGS.restore_op_name,
        FLAGS.filename_tensor_name,
        FLAGS.output_graph,
        FLAGS.clear_devices,
        FLAGS.initializer_nodes,
        FLAGS.variable_names_blacklist
    )
