#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : Atom
#   File name   : prune.py
#   Author      : SQMah
#   Created date: 2019-06-26 12:52:23
#   Description : Prune the bboxes from the train.txt generated by scraper.py
#                 that no longer exist
#
#================================================================

import os

TRAIN_TXT = "train.txt"

if os.path.exists(TRAIN_TXT):
    f = open(TRAIN_TXT, "r")
    lines = f.readlines()

    # Save as backup
    os.rename(TRAIN_TXT, TRAIN_TXT + ".bak")

    f2 = open(TRAIN_TXT, "a")

    for line in lines:
        split_line = line.split()
        image, box_line = split_line[0], split_line[1:]
        box_string = ""

        for box in box_line:
            split_box = box.split("||")
            box, path = split_box[0], split_box[1]

            # Round class ids
            box_params = box.split(",")
            box_params = [str(round(float(param))) for param in box_params]
            box = ",".join(box_params)

            if len(split_box) > 1:
                if os.path.isfile(path):
                    box_string += " " + box
                else:
                    print("{} not found! Removing...".format(path))
            else:
                # This has already been pruned
                box_string += " " + box

        # Check if there were any boxes at all
        if box_string:
            f2.write(image + box_string + '\n')

    f2.close()

else:
    print("Train.txt not found at: /{}, please update the TRAIN_TXT variable.".format(TRAIN_TXT))
