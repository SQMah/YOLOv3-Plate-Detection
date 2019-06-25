#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : Atom
#   File name   : image_batch.py
#   Author      : SQMah
#   Created date: 2019-06-25 10:26:03
#   Description :
#
#================================================================

import os
import cv2
import numpy as np
import core.utils as utils
import tensorflow as tf
from PIL import Image

return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
pb_file         = "./yolov3_coco.pb"
image_dir       = "./to_crop"
output_dir      = "./cropped"
num_classes     = 1
input_size      = 960  # This HAS to be a multiple of 32
graph           = tf.Graph()

images = os.listdir(image_dir)
image_counter = 0

with tf.Session(graph = graph) as sess:
    for image_path in images:
        image_path = os.path.join(image_dir, image_path)
        is_image = False

        # Check if it is an image
        try:
            original_image = cv2.imread(image_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            is_image = True
        except:
            pass

        if is_image:
            original_image_size = original_image.shape[:2]
            image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
            image_data = image_data[np.newaxis, ...]

            return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)

            pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
                [return_tensors[1], return_tensors[2], return_tensors[3]],
                        feed_dict={ return_tensors[0]: image_data})

            pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                        np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                        np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

            bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.3, False)
            bboxes = utils.nms(bboxes, 0.45, method='nms')
            image = Image.fromarray(original_image)

            for bbox in bboxes:
                # Add some padding
                cropped = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                cropped.save(os.path.join(output_dir, str(image_counter) + '.jpg'))
                image_counter += 1
