import tensorflow as tf
from keras import backend as K

with tf.Session() as test_a:
    box_confidence = tf.constant(
        [
            [
                [
                    [10, 20]
                ], [
                    [11, 21]
                ], [
                    [12, 22]
                ]
            ], [
                [
                    [40, 50]
                ], [
                    [410, 51]
                ], [
                    [42, 52]
                ]
            ], [
                [
                    [70, 80]
                ], [
                    [71, 81]
                ], [
                    [72, 82]
                ]
            ]
        ]
    )
    print("box_confidence.shape = " + str(box_confidence.shape))
    #print("box_confidence =\n" + str(test_a.run(box_confidence)))


    box_classes = K.argmax(box_confidence, axis=-1)
    box_class_scores = K.max(box_confidence, axis=-1)
    print("box_classes.shape: " + str(box_classes.shape))
    print("box_classes[0, :] = " + str(test_a.run(box_classes)))
    print("box_classes[0, :] = " + str(test_a.run(box_classes[0, :])))
    print("box_classes[0, 0, :] = " + str(test_a.run(box_classes[0, 0, :])))
    print("box_classes[0, 0, 0] = " + str(test_a.run(box_classes[0, 0, 0])))

    print("box_class_scores.shape: " + str(box_class_scores.shape))
    print("box_class_scores[0, :] = " + str(test_a.run(box_class_scores)))
    print("box_class_scores[0, :] = " + str(test_a.run(box_class_scores[0, :])))
    print("box_class_scores[0, 0, :] = " + str(test_a.run(box_class_scores[0, 0, :])))
    print("box_class_scores[0, 0, 0] = " + str(test_a.run(box_class_scores[0, 0, 0])))

    filtering_mask = (tf.cast(box_class_scores, tf.float32) > 400)
    print("filtering_mask.shape: " + str(filtering_mask.shape))
    print("filtering_mask[0, :] = " + str(test_a.run(filtering_mask)))

    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    print("scores.shape: " + str(scores.shape))
    print("scores = " + str(test_a.run(scores)))