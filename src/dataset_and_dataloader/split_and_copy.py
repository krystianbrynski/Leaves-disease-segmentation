import os
import shutil

from sklearn.model_selection import train_test_split


def copy_data(images: str, new_images_dir: str, new_masks_dir: str, images_dir: str, masks_dir: str):
    for i in images:
        image_file = images_dir + '/' + i
        mask_file = masks_dir + '/' + i.replace(".jpg", ".png")
        shutil.copy(image_file, new_images_dir)
        shutil.copy(mask_file, new_masks_dir)


def split_data(images_dir: str, masks_dir: str, train_masks_dir: str, test_masks_dir: str, train_images_dir: str,
               test_images_dir: str, valid_images_dir: str, valid_masks_dir: str) -> None:
    images_file = os.listdir(images_dir)  # all files in images dir
    masks_file = os.listdir(masks_dir)  # all files in masks dir

    os.makedirs(train_masks_dir, exist_ok=True)  # make new dir for masks train
    os.makedirs(test_masks_dir, exist_ok=True)  # make new dir for masks test
    os.makedirs(valid_masks_dir, exist_ok=True)  # make new dir for masks valid
    os.makedirs(train_images_dir, exist_ok=True)  # make new dir for images train
    os.makedirs(test_images_dir, exist_ok=True)  # make new dir for images test
    os.makedirs(valid_images_dir, exist_ok=True)  # make new dir for images valid

    train_images, temp_images, train_masks, temp_masks = train_test_split(images_file, masks_file, test_size=0.2,
                                                                          random_state=42)

    test_images, valid_images, test_masks, valid_masks = train_test_split(temp_images, temp_masks, test_size=0.5,
                                                                          random_state=42)

    copy_data(train_images, train_images_dir, train_masks_dir, images_dir, masks_dir)  # copy train data
    copy_data(test_images, test_images_dir, test_masks_dir, images_dir, masks_dir)  # copy test data
    copy_data(valid_images, valid_images_dir, valid_masks_dir, images_dir, masks_dir)  # copy valid data
