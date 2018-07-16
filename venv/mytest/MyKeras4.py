import tensorflow as tf
from keras import backend as K

with tf.Session() as test_a:
    box_confidence = tf.constant(
        [
            [
                [
                    [10], [20]
                ], [
                    [11], [21]
                ], [
                    [12], [22]
                ]
            ], [
                [
                    [40], [50]
                ], [
                    [41], [51]
                ], [
                    [42], [52]
                ]
            ], [
                [
                    [70], [80]
                ], [
                    [71], [81]
                ], [
                    [72], [82]
                ]
            ]
        ]
    )
    print("box_confidence.shape = " + str(box_confidence.shape))
    #print("box_confidence =\n" + str(test_a.run(box_confidence)))

    boxes = tf.constant(
        [
            [
                [
                    [100, 101, 102, 103], [201, 202, 203, 204]
                ], [
                    [110, 111, 112, 113], [210, 211, 212, 213]
                ], [
                    [120, 121, 122, 123], [220, 221, 222, 223]
                ]
            ], [
                [
                    [400, 401, 402, 403], [500, 501, 502, 503]
                ], [
                    [410, 411, 412, 413], [510, 511, 512, 513]
                ], [
                    [420, 421, 422, 423], [520, 521, 522, 523]
                ]
            ], [
                [
                    [700, 701, 702, 703], [800, 801, 802, 803]
                ], [
                    [710, 711, 712, 713], [810, 811, 812, 813]
                ], [
                    [720, 721, 722, 723], [820, 821, 822, 823]
                ]
            ]
        ]
    )
    print("boxes.shape = " + str(boxes.shape))
    #print("boxes =\n" + str(test_a.run(boxes)))

    box_class_probs = tf.constant(
        [
            [
                [
                    [100, 101, 102, 103], [201, 202, 203, 204]
                ], [
                    [110, 111, 112, 113], [210, 211, 212, 213]
                ], [
                    [120, 121, 122, 123], [220, 221, 222, 223]
                ]
            ], [
                [
                    [400, 401, 402, 403], [500, 501, 502, 503]
                ], [
                    [410, 411, 412, 413], [510, 511, 512, 513]
                ], [
                    [420, 421, 422, 423], [520, 521, 522, 523]
                ]
            ], [
                [
                    [700, 701, 702, 703], [800, 801, 802, 803]
                ], [
                    [710, 711, 712, 713], [810, 811, 812, 813]
                ], [
                    [720, 721, 722, 723], [820, 821, 822, 823]
                ]
            ]
        ]
    )

    print("box_class_probs.shape = " + str(box_class_probs.shape))
    #print("box_class_probs =\n" + str(test_a.run(boxes)))

    box_scores = box_confidence * box_class_probs

    box_classes = K.argmax(box_scores, axis=-1)
    print("box_classes.shape: " + str(box_classes.shape))
    print("box_classes[0, :] = " + str(test_a.run(box_classes[0, :])))
    print("box_classes[0, 0, :] = " + str(test_a.run(box_classes[0, 0, :])))
    print("box_classes[0, 0, 0] = " + str(test_a.run(box_classes[0, 0, 0])))

    print("box_scores.shape: " + str(box_scores.shape))
    print("box_scores: " + str(test_a.run(box_scores)))
    #print("box_classes = " + str(test_a.run(box_classes)))