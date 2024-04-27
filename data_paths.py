from libraries import *
import os

data_dir = '/datasets/tdt4265/mic/asoca'
categories = ['Diseased', 'Normal']  # Define the categories

train_files = []  # Initialize an empty list to hold the data dictionaries

for category in categories:
    image_dir = os.path.join(data_dir, category, 'CTCA')
    annotation_dir = os.path.join(data_dir, category, 'Annotations')  # Ensure folder names are correct

    # List and sort images and annotations
    images = sorted(os.listdir(image_dir))
    annotations = sorted(os.listdir(annotation_dir))

    # Assuming filenames for annotations might require minor adjustments from image filenames
    for img in images:
        base_name = img.split('.')[0]  # Removes the file extension assuming it's '.nrrd'
        # Adjust the following line if the naming convention is different
        annotation_file = base_name + '.nrrd'
        if annotation_file in annotations:
            train_files.append({
                'img': os.path.join(image_dir, img),
                'seg': os.path.join(annotation_dir, annotation_file)
            })
        else:
            print(f"Warning: No corresponding annotation found for {os.path.join(image_dir, img)}")

print("finish")