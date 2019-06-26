#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : freeze_graph.py
#   Author      : YunYang1994
#   Created date: 2019-03-20 15:57:33
#   Description :
#
#================================================================


import tensorflow as tf
from core.yolov3 import YOLOV3

pb_file = "./yolov3_coco.pb"
ckpt_file = "./checkpoint/yolov3_test_loss=1.5346.ckpt-45"
output_node_names = ["input/input_data", "pred_sbbox/concat_2", "pred_mbbox/concat_2", "pred_lbbox/concat_2"]

with tf.name_scope('input'):
    input_data = tf.compat.v1.placeholder(dtype=tf.float32, name='input_data')

model = YOLOV3(input_data, trainable=False)
print(model.conv_sbbox, model.conv_mbbox, model.conv_lbbox)

sess  = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True))
saver = tf.train.Saver()
saver.restore(sess, ckpt_file)

converted_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(sess,
                            input_graph_def  = sess.graph.as_graph_def(),
                            output_node_names = output_node_names)

with tf.io.gfile.GFile(pb_file, "wb") as f:
    f.write(converted_graph_def.SerializeToString())
