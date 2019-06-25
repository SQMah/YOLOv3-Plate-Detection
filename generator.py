"""
Generates the correct text file to train the YOLOv3 model
Accepts via_region_data or flat images with the correct names
Also add the classes
"""
import os
import json
import cv2

# Directory containing training data, each subfolder is a class
DATA_DIR = "training_data"

# Create the train.txt file
training = open('train.txt', 'a+')

# Create the plate_classes.txt file
classes = open('model_data/plate_and_car_classes.txt', 'a+')

class_list = os.listdir(DATA_DIR)


def remove_ds_store(directory_list):
    if ".DS_Store" in directory_list:
        directory_list.remove(".DS_Store")

# Remove .DS_Store for macOS users
remove_ds_store(class_list)

# Write each class to the plate_classes.txt file
for index, one_class in enumerate(class_list):
    # Get all images in the class directory
    class_dir = os.path.join(DATA_DIR, one_class)
    images = os.listdir(class_dir)
    remove_ds_store(images)

    # Special handling for region.json
    if "region.json" in images:
        print("region.json found!")
        print("Loading images...")
        json_path = os.path.join(class_dir, "region.json")
        annotations = json.load(open(json_path))

        for image in annotations:
            image_exists = False
            filename = annotations[image]['filename']

            # Make sure that the name of the file has no spaces
            image_path = os.path.join(class_dir, filename)
            filename = filename.replace(" ", "_")
            new_image_path = os.path.join(class_dir, filename)

            try:
                os.rename(image_path, new_image_path)
                image_exists = True
            except:
                # Image doesn't exist
                pass

            if image_exists:
                print("Loaded " + filename)
                regions = annotations[image]['regions']
                boxes = []
                for region in regions:
                    x_coords = region['shape_attributes']['all_points_x']
                    y_coords = region['shape_attributes']['all_points_y']
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)

                    # Box format: x_min,y_min,x_max,y_max,class_id
                    boxes.append('{},{},{},{},{}'.format(x_min, y_min, x_max, y_max, class_list.index(list(region['region_attributes'].keys())[0])))

                # Row format: image_file_path box1 box2 ... boxN
                row_string = new_image_path
                for box in boxes:
                    row_string += " "
                    row_string += box

                # Check if was populated with box at all, could be an image that exists without a bounding box
                if row_string != new_image_path:
                    training.write(row_string + "\n")

    else: # Get bounding boxes from file name
        # Each box is formatted as such: x_min,y_min,x_max,y_max,NAME
        # Row format: image_file_path box1 box2 ... boxN
        for image in images:
            row_string = image
            boxes = image.split()

            # Add class_id
            for box_index, box in enumerate(boxes):
                box_params = box.split(",")
                class_name = box_params.pop()
                box_params.append(class_list.index(class_name))
                boxes[box_index] = ",".join(box_params)

            for box in boxes:
                row_string += " "
                row_string += box

            training.write(row_string + "\n")

    # Write classes
    if index == len(class_list): # Last line does not have a line escape
        classes.write(one_class)
    else:
        classes.write(one_class + "\n")

training.close()
classes.close()
