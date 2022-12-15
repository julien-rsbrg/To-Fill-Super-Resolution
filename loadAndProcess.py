import yaml
from yaml.loader import SafeLoader

with open('config.yaml') as f:
    config = yaml.load(f, Loader=SafeLoader)


################################# LOAD DATASET #################################

# source: https://www.kaggle.com/datasets/saputrahas/dataset-image-super-resolution

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pathlib
import os

DTYPE = 'float32'
tf.keras.backend.set_floatx(DTYPE)

AUTOTUNE = tf.data.AUTOTUNE

################################# #################################
current_dir = os.getcwd()
print("current_dir:", current_dir)

train_data_dir = os.path.join(current_dir, "datasets/train/dataraw")
print("train_data_dir:", train_data_dir)

val_data_dir = os.path.join(current_dir, "introduction/specific_validation")
print("val_data_dir:", val_data_dir, '\n')


################################# #################################
print('------LOAD DATASET------')

train_ds = tf.data.Dataset.list_files(
    str(pathlib.Path(train_data_dir+'/*.JPG')), shuffle=False)
plib_train_data_dir = pathlib.Path(train_data_dir)
image_count = len(list(plib_train_data_dir.glob('*.JPG')))

train_ds = train_ds.shuffle(image_count, reshuffle_each_iteration=True)


val_ds = tf.data.Dataset.list_files(
    str(pathlib.Path(val_data_dir+'/*.JPG')), shuffle=False)
plib_val_data_dir = pathlib.Path(val_data_dir)
image_count = len(list(plib_val_data_dir.glob('*.JPG')))

val_ds = val_ds.shuffle(image_count, reshuffle_each_iteration=True)

print('training and validation datasets loaded')


print("length of each dataset:")
print("training:", tf.data.experimental.cardinality(train_ds).numpy())
print("validation:", tf.data.experimental.cardinality(val_ds).numpy())


################################# process path #################################
print('--process path--')


def decode_img(img):
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_jpeg(img, channels=3)
    return tf.cast(img, dtype=DTYPE)


def process_path(file_path):
    # Load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img


# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)


################################# Visualize #################################
if __name__ == '__main__':
    print("--visualize dataset--")

    def visualize_ds(ds):
        plt.figure(figsize=(15, 15))

        i = 0
        for img in ds.take(9):
            ax = plt.subplot(3, 3, i + 1)
            ax.imshow(img.numpy().astype("uint8"))
            ax.axis("off")
            i += 1
        plt.show()

    visualize_ds(train_ds)


################################# PROCESS DATASET #################################
print('------PROCESS DATASET------')

################################# Crop and rescale #################################
print('--Crop and rescale--')

high_resolution_shape = config["HIGH_RESOLUTION_SHAPE"]
scale_factor = config["SCALE_FACTOR"]

seed = np.random.randint(0, 101, size=2, dtype=np.int32)
print("seed for crop layer:", seed)


crop_and_rescale = tf.keras.Sequential([
    tf.keras.layers.Lambda(lambda img: tf.image.stateless_random_crop(
        img, size=high_resolution_shape, seed=seed)),
    # tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Rescaling(1./127.5, offset=-1)
])


train_ds = train_ds.map(lambda img: crop_and_rescale(img))
val_ds = val_ds.map(lambda img: crop_and_rescale(img))

################################# Add low resolution #################################
print('--Add low resolution--')

avg_pooling_2d = tf.keras.layers.AveragePooling2D((4, 4), padding="same")


def apply_avg_pooling(img):
    img = tf.expand_dims(img, axis=0)
    img_avg = avg_pooling_2d(img)
    img_avg = tf.squeeze(img_avg)
    return img_avg


train_ds = train_ds.map(lambda img: (apply_avg_pooling(img), img))
val_ds = val_ds.map(lambda img: (apply_avg_pooling(img), img))


def lr_hr_comparison(ds):
    plt.figure(figsize=(15, 15))

    i = 0
    for lr_img, hr_img in ds.take(3):
        ax = plt.subplot(3, 2, i + 1)
        plt.imshow(lr_img.numpy())
        plt.title('lr'+str(lr_img.shape))
        plt.axis("off")
        i += 1

        ax = plt.subplot(3, 2, i + 1)
        plt.imshow(hr_img.numpy())
        plt.title('hr'+str(hr_img.shape))
        plt.axis("off")
        i += 1
    plt.show()


################################# CONFIGURE FOR PERFORMANCE #################################
print('------CONFIGURE FOR PERFORMANCE------')


def configure_for_performance(ds, batch_size):
    # ds = ds.cache() # dataset too large for it
    ds = ds.shuffle(
        buffer_size=config["BUFFER_SIZE"], reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


train_ds = configure_for_performance(train_ds, config["BATCH_SIZE"])
val_ds = configure_for_performance(val_ds, config["VAL_BATCH_SIZE"])

if __name__ == "__main__":
    if config["BATCH_SIZE"] >= 9:
        for image_batch, label_batch in train_ds.take(1):
            plt.figure(figsize=(10, 10))
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(image_batch[i].numpy())
                plt.title("final dataset...")
                plt.axis("off")

            plt.show()
