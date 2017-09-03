"""Simple python wrapper for inspecting checkpoint from cli."""

# Standard imports
import sys

# Dependecy imports
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

CHK_NAME = sys.argv[1]

print_tensors_in_checkpoint_file(CHK_NAME, "", False)
