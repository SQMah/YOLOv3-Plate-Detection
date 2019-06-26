#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : Atom
#   File name   : scraper.py
#   Author      : SQMah
#   Created date: 2019-06-25 15:36:53
#   Description :
#
#================================================================

import os
import cv2
import numpy as np
import core.utils as utils
import tensorflow as tf
from PIL import Image
import requests
import urllib.request
from bs4 import BeautifulSoup
from io import BytesIO
import json

return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
pb_file         = "./yolov3_coco.pb"
output_dir      = "./cropped"
num_classes     = 1
input_size      = 960  # This HAS to be a multiple of 32, increasing this improves accuracy, but also slows it down
graph           = tf.Graph()

# Storage variables from web scraping
image_urls = []
plates = []
makes = []

# Save progress, this is pretty prone to crashing
url_dict = {}
url_dict['image_urls'] = []
urls_exists = False
url_path = os.path.join(output_dir, 'urls.json')
index_dict = {}
index_dict['index'] = 0
index_exists = False
index_path = os.path.join(output_dir, 'index.json')

'''
Start scraping platesmania for the urls you need
Modify the range and starting URL as needed
'''

try:
    with open(url_path) as urls_file:
        url_dict = json.load(urls_file)
        urls_exists = True
except:
    urls_exists = False

if not urls_exists:
    for i in range(0, 99):
        url = 'http://platesmania.com/cn/gallery.php?&format=4032&start=' + str(i)
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.90 Safari/537.36'}
        response = requests.get(url, headers = headers)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Handle panels
        panels = soup.find_all('div', class_='panel-body')

        for panel in panels:
            # Get image:
            img_url = panel.find_all('img')[0]['src']

            # Get plate number
            plate_num = panel.find_all('img')[1]['alt']

            # Get make
            make = ""
            for a in panel.find_all('a'):
                if a.get_text().strip() != "" :
                    make = a.get_text()
                    break

            image_urls.append(img_url.replace("/m", "/o")) # Get the full sized images
            plates.append(plate_num)
            makes.append(make)

            print ("Loaded {} image urls...".format(len(image_urls)), end="\r")

    url_dict['image_urls'] = image_urls
    url_dict['plates'] = plates
    with open(url_path, 'w') as outfile:
        json.dump(url_dict, outfile)
    print("\n")
else:
    print("URLs loaded from json!")

try:
    with open(index_path) as index_file:
        index_dict = json.load(index_file)
        print("Index loaded from json!")
except:
    pass

# Create the directory to store the original assets
try:
    os.mkdir(os.path.join(output_dir, 'orig'))
    print ("Successfully created the orig directory")
except OSError:
    print ("Creation of the directory orig failed or orig already exists")


# Create the train.txt file
training = open('train.txt', 'a+')

with tf.compat.v1.Session(graph = graph) as sess:

    for url in url_dict['image_urls'][int(index_dict['index']):]:
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.90 Safari/537.36'}

        is_image = False

        # Check if it is an image
        try:
            img_stream = BytesIO(requests.get(url, headers = headers).content)
            original_image = cv2.imdecode(np.fromstring(img_stream.read(), np.uint8), 1)
            is_image = True
        except:
            pass

        if is_image:
            image_counter = 0

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

            bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.2, False)
            bboxes = utils.nms(bboxes, 0.45, method='nms')
            image = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))

            # Save originals
            orig_path = os.path.join(os.path.join(output_dir, 'orig'), str(index_dict['index']) + '.jpg')
            image.save(orig_path)
            row_string = orig_path

            for bbox in bboxes:
                """
                bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
                """

                # Add some padding
                cropped = image.crop((bbox[0] - 10, bbox[1] - 10, bbox[2] + 10, bbox[3] + 10))
                cropped_path = os.path.join(output_dir, url_dict['plates'][int(index_dict['index'])] + '_{}.jpg'.format(image_counter))
                cropped.save(cropped_path)
                image_counter += 1

                row_string += " "

                # This is to allow checking if the file still exists later to see if the detection was positive or a false positive
                row_string += '{},{},{},{},{}||{}'.format(bbox[0], bbox[1], bbox[2], bbox[3], bbox[5], cropped_path)

            # Save detected boxes to training file
            # Check if was populated with box at all, could be an image that exists without a bounding box
            if row_string != orig_path:
                training.write(row_string + "\n")

            # Save current index
            index_dict['index'] = int(index_dict['index']) + 1
            with open(index_path, 'w') as outfile:
                json.dump(index_dict, outfile)

            # Save some memory
            img_stream = None
            original_image = None
            image = None
            image_data = None
            return_tensors = None
