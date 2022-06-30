##Data augmentation
img_size = 512
import tensorflow as tf


@tf.function
def resize(input_img, tar_img, img_size):
    input_img = tf.image.resize(input_img, [img_size, img_size])
    tar_img = tf.image.resize(tar_img, [img_size, img_size])
    
    return input_img, tar_img


def normalize(input_img, tar_img):
    input_img = (input_img/255.) - 1
    tar_img = (tar_img/255.) - 1
    return input_img, tar_img

def random_jitter(input_img, tar_img):
    input_img, tar_img = resize(input_img, tar_img, 572)

  
    stacked_image = tf.stack([input_img, tar_img], axis=0)
  
    cropped_image = tf.image.random_crop(stacked_image, size=[2, img_size, img_size, 3])
    
    input_img, tar_img = cropped_image[0], cropped_image[1]
    if tf.random.uniform(()) > 0.5:
        input_img = tf.image.flip_left_right(input_img)
        tar_img = tf.image.flip_left_right(tar_img)
    return input_img, tar_img