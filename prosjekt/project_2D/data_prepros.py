import SimpleITK as sitk
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split

def load_nrrd_file(file_path):
    itkimage = sitk.ReadImage(file_path)
    array = sitk.GetArrayFromImage(itkimage)
    return array

def preprocess_image(image_array):
    image_array = image_array.astype(np.float16)
    min_value = np.min(image_array)
    if min_value < 0:
        image_array -= min_value
    max_value = np.max(image_array)
    if max_value > 0:
        image_array /= max_value
    return image_array

def preprocess_mask(mask_array):
    mask_array = mask_array.astype(np.float16)
    mask_array[mask_array > 0] = 1
    return mask_array

def process_all_files(base_path):
    categories = ['Normal', 'Diseased']
    types = ['CTCA', 'Annotations']
    data = []
    for category in categories:
        for i in range(1, 21):
            image_path = f'{base_path}/{category}/{types[0]}/{category}_{i}.nrrd'
            mask_path = f'{base_path}/{category}/{types[1]}/{category}_{i}.nrrd'
            if os.path.exists(image_path) and os.path.exists(mask_path):
                image_volume = load_nrrd_file(image_path)
                mask_volume = load_nrrd_file(mask_path)
                for slice_index in range(image_volume.shape[0]):
                    processed_image = preprocess_image(image_volume[slice_index])
                    processed_mask = preprocess_mask(mask_volume[slice_index])
                    data.append((processed_image, processed_mask))
    return data

def pad_to_shape(arr, max_shape):
    # Ensure we're padding a 2D array (one slice at a time)
    result = np.zeros(max_shape, dtype=arr.dtype)
    slices = tuple(slice(0, min(dim, max_dim)) for dim, max_dim in zip(arr.shape, max_shape))
    result[slices] = arr
    return result.reshape(result.shape + (1,))  # Add channel dimension


def generate_batches(data, batch_size):
    random.shuffle(data)
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        # Safely handle possibly empty batches if total data size isn't a multiple of batch_size
        if not batch:
            continue

        max_height = max(x[0].shape[0] for x in batch)
        max_width = max(x[0].shape[1] for x in batch)
        max_shape = (max_height, max_width)

        X_batch = np.array([pad_to_shape(x[0], max_shape) for x in batch], dtype=np.float16)
        y_batch = np.array([pad_to_shape(x[1], max_shape) for x in batch], dtype=np.float16)

        X_batch = X_batch.reshape((-1, max_height, max_width, 1))  # Ensure the correct shape
        y_batch = y_batch.reshape((-1, max_height, max_width, 1))

        yield (X_batch, y_batch)



if __name__ == "__main__":
    base_path = '/datasets/tdt4265/mic/asoca'
    data = process_all_files(base_path)
    batch_size = 5
    for X_batch, y_batch in generate_batches(data, batch_size):
        print("Batch X shape:", X_batch.shape)
        print("Batch Y shape:", y_batch.shape)
