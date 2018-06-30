'''
Capsules for Object Segmentation (SegCaps)
Original Paper by Rodney LaLonde and Ulas Bagci (https://arxiv.org/abs/1804.04241)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This file contains the custom data augmentation functions.
'''

import numpy as np
import cv2
from scipy.ndimage.interpolation import map_coordinates

from keras.preprocessing.image import random_rotation, random_shift, random_zoom, random_shear

MASK_BACKGROUND = (0,0,0,0)    

# Function to distort image
def elastic_transform(image, alpha=2000, sigma=40, alpha_affine=40, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    for i in range(shape[2]):
        image[:,:,i] = cv2.warpAffine(image[:,:,i], M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    image = image.reshape(shape)

    blur_size = int(4*sigma) | 1

    dx = cv2.GaussianBlur((random_state.rand(*shape_size) * 2 - 1), ksize=(blur_size, blur_size), sigmaX=sigma) * alpha
    dy = cv2.GaussianBlur((random_state.rand(*shape_size) * 2 - 1), ksize=(blur_size, blur_size), sigmaX=sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    def_img = np.zeros_like(image)
    for i in range(shape[2]):
        def_img[:,:,i] = map_coordinates(image[:,:,i], indices, order=1).reshape(shape_size)

    return def_img


def salt_pepper_noise(image, salt=0.2, amount=0.004):
    row, col, chan = image.shape
    num_salt = np.ceil(amount * row * salt)
    num_pepper = np.ceil(amount * row * (1.0 - salt))

    for n in range(chan//2): # //2 so we don't augment the mask
        # Add Salt noise
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[0:2]]
        image[coords[0], coords[1], n] = 1

        # Add Pepper noise
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[0:2]]
        image[coords[0], coords[1], n] = 0

    return image


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

def image_resize2square(image, desired_size = None):
    '''
    Transform image to a square image with desired size(resolution)
    Padding image with black color which defined as MASK_BACKGROUND
    '''
    
    # initialize dimensions of the image to be resized and
    # grab the image size
    old_size = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if desired_size is None or (old_size[0]==desired_size and old_size[1]==desired_size):
        return image

    # calculate the ratio of the height and construct the
    # dimensions
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format
    resized = cv2.resize(image, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    new_image = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value = MASK_BACKGROUND)

    # return the resized image
    return new_image


def augmentImages(batch_of_images, batch_of_masks):
    for i in range(len(batch_of_images)):
        img_and_mask = np.concatenate((batch_of_images[i, ...], batch_of_masks[i,...]), axis=2)
        if img_and_mask.ndim == 4: # This assumes single channel data. For multi-channel you'll need
            # change this to put all channel in slices channel
            orig_shape = img_and_mask.shape
            img_and_mask = img_and_mask.reshape((img_and_mask.shape[0:3]))

        if np.random.randint(0,10) == 7:
            img_and_mask = random_rotation(img_and_mask, rg=45, row_axis=0, col_axis=1, channel_axis=2,
                                           fill_mode='constant', cval=0.)

        if np.random.randint(0, 5) == 3:
            img_and_mask = elastic_transform(img_and_mask, alpha=1000, sigma=80, alpha_affine=50)

        if np.random.randint(0, 10) == 7:
            img_and_mask = random_shift(img_and_mask, wrg=0.2, hrg=0.2, row_axis=0, col_axis=1, channel_axis=2,
                                        fill_mode='constant', cval=0.)

        if np.random.randint(0, 10) == 7:
            img_and_mask = random_shear(img_and_mask, intensity=16, row_axis=0, col_axis=1, channel_axis=2,
                         fill_mode='constant', cval=0.)

        if np.random.randint(0, 10) == 7:
            img_and_mask = random_zoom(img_and_mask, zoom_range=(0.75, 0.75), row_axis=0, col_axis=1, channel_axis=2,
                         fill_mode='constant', cval=0.)

        if np.random.randint(0, 10) == 7:
            img_and_mask = flip_axis(img_and_mask, axis=1)

        if np.random.randint(0, 10) == 7:
            img_and_mask = flip_axis(img_and_mask, axis=0)

        if np.random.randint(0, 10) == 7:
            salt_pepper_noise(img_and_mask, salt=0.2, amount=0.04)

        if batch_of_images.ndim == 4:
            batch_of_images[i, ...] = img_and_mask[...,0:img_and_mask.shape[2]//2]
            batch_of_masks[i,...] = img_and_mask[...,img_and_mask.shape[2]//2:]
        if batch_of_images.ndim == 5:
            img_and_mask = img_and_mask.reshape(orig_shape)
            batch_of_images[i, ...] = img_and_mask[...,0:img_and_mask.shape[2]//2, :]
            batch_of_masks[i,...] = img_and_mask[...,img_and_mask.shape[2]//2:, :]

        # Ensure the masks did not get any non-binary values.
        batch_of_masks[batch_of_masks > 0.5] = 1
        batch_of_masks[batch_of_masks <= 0.5] = 0

    return(batch_of_images, batch_of_masks)