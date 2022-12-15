import os
import tensorflow as tf
import json
import yaml
from yaml.loader import SafeLoader
DTYPE = "float32"

with open('config.yaml') as f:
    config = yaml.load(f, Loader=SafeLoader)

print("config:\n", config, '\n')


def clean_folder(root_folder):
    '''
    Delete all the corrupted pictures in root_folder. These images cannot be opened correctly and used by the model.

    Variables:
    -root_folder (str): absolute path to the folder to clean
    '''
    l_corrupted_pictures = []
    l_wrong_shapes_pictures = []
    print(f"begin checking folder {root_folder}...\n")

    l_imgs = os.listdir(root_folder)
    N_imgs = len(l_imgs)
    for i, img in enumerate(l_imgs):
        if img[-3:] == 'JPG' or img[-3:] == 'jpg':
            path_img = os.path.join(root_folder, img)
            try:
                jpg_img = tf.io.decode_jpeg(
                    tf.io.read_file(path_img), channels=3)
                jpg_img = tf.cast(jpg_img, dtype=DTYPE)

                for j in range(3):
                    if jpg_img.shape[j] < config["HIGH_RESOLUTION_SHAPE"][j]:
                        l_wrong_shapes_pictures.append(
                            (img, jpg_img.numpy().shape))
                        break
            except:
                l_corrupted_pictures.append(img)
        if i % 100 == 0:
            print("%4.0f/%4.0f images checked so far" % (i, N_imgs), end='\r')

    print('l_corrupted_pictures:', l_corrupted_pictures, '                ')

    with open(os.path.join(root_folder, 'l_corrupted_pictures.json'), 'w') as f:
        json.dump(l_corrupted_pictures, f, indent=2)

    print('l_corrupted_pictures saved at:', os.path.join(
        root_folder, 'l_corrupted_pictures.json'))

    for name in l_corrupted_pictures:
        os.remove(os.path.join(root_folder, name))

    print('elements of l_corrupted_pictures have been deleted\n')

    print('l_wrong_shapes_pictures:', l_wrong_shapes_pictures)

    with open(os.path.join(root_folder, 'l_wrong_shapes_pictures.json'), 'w') as f:
        json.dump(l_wrong_shapes_pictures, f, indent=2)

    print('l_wrong_shapes_pictures saved at:', os.path.join(
        root_folder, 'l_wrong_shapes_pictures.json'))

    for name, _ in l_wrong_shapes_pictures:
        os.remove(os.path.join(root_folder, name))

    print('elements of l_wrong_shapes_pictures have been deleted\n')


if __name__ == '__main__':
    path_to_validation_dataset = ...
    path_to_train_dataset = ...
    l_root_folders = [path_to_train_dataset, path_to_validation_dataset]
    for root_folder in l_root_folders:
        clean_folder(root_folder)
