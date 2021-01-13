import tensorflow as tf
import tensorflow.keras as keras
import numpy as np


def loadimage(imagepath):
    image=keras.preprocessing.image.load_img(imagepath)

    image_array=keras.preprocessing.image.img_to_array(image)
    image_array_cropped=tf.image.crop_and_resize(image_array,)

    keras.preprocessing.image.array_to_img(image_array_cropped).show()

loadimage("t1.jpg")