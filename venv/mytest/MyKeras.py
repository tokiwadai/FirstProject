from typing import Any, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from tensorflow import Session

sess = K.get_session()
sess.run()

with tf.Session() as test_a:
    box_confidence = tf.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1)
    print("box_confidence.shape: " + str(box_confidence.shape))
    boxes = tf.random_normal([19, 19, 5, 4], mean=1, stddev=4, seed = 1)
    print("boxes.shape: " + str(boxes.shape))
    box_class_probs = tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1)
    print("box_class_probs.shape: " + str(box_class_probs.shape))

    box_scores = box_confidence * box_class_probs
    print("box_scores.shape: " + str(box_scores.shape))

    box_classes = K.argmax(box_scores, axis=-1)
    box_class_scores = K.max(box_classes, axis=-1, keepdims=False)
    print("box_classes.shape: " + str(box_classes.shape))
    print("box_class_scores.shape: " + str(box_class_scores.shape))