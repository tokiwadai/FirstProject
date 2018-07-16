import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model

with tf.Session() as test_a:
    box_confidence = tf.random_normal([2, 2, 1, 1], mean=1, stddev=4, seed = 1)
    print("box_confidence.shape: " + str(box_confidence.shape))
    print("box_confidence1: " + str(test_a.run(box_confidence)))
    print("box_confidence2: " + str(test_a.run(box_confidence)))

    #b = box_confidence[0, :]
    #print(test_a.run(b))
    b1 = box_confidence[0, :]
    print("b1: " + str(test_a.run(b1)))
    b2 = box_confidence[1, :]
    print("b2: " + str(test_a.run(b2)))

    boxes = tf.random_normal([4, 4, 2, 4], mean=1, stddev=4, seed = 1)
    print("boxes.shape: " + str(boxes.shape))
    box_class_probs = tf.random_normal([4, 4, 2, 5], mean=1, stddev=4, seed = 1)
    print("box_class_probs.shape: " + str(box_class_probs.shape))

    box_scores = box_confidence * box_class_probs
    print("box_scores.shape: " + str(box_scores.shape))

    box_classes = K.argmax(box_scores, axis=-1)
    box_class_scores = K.max(box_classes, axis=-1, keepdims=False)
    print("box_classes.shape: " + str(box_classes.shape))
    print("box_class_scores.shape: " + str(box_class_scores.shape))