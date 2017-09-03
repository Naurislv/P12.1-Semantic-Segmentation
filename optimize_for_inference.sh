#!/bin/bash

python freeze_graph.py \
--input_graph=data/vgg_fcn/inference_fcn8s_vgg_2017-09-02-21-24.pbtxt \
--output_graph=data/vgg_fcn/freezed_fcn8s_vgg_2017-09-02-21-24.pb \
--input_checkpoint=data/vgg_fcn/inference_fcn8s_vgg_2017-09-02-21-24.chk \
--output_node_names=outputs/logits_softmax

python opt_for_inf.py \
--input=data/vgg_fcn/freezed_fcn8s_vgg_2017-09-02-21-24.pb \
--output=data/vgg_fcn/optimized_fcn8s_vgg_2017-09-02-21-24.pb \
--frozen_graph=True \
--intput_names=image_input \
--output_names=outputs/logits_softmax