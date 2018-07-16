import tensorflow as tf

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
    print("box_confidence =\n" + str(test_a.run(box_confidence)))
    print("box_confidence[0, :] =\n" + str(test_a.run(box_confidence[0, :])))
    print("box_confidence[0, 0] = " + str(test_a.run(box_confidence[0, 0])))
    print("box_confidence[0, 0, 0] = " + str(test_a.run(box_confidence[0, 0, 0])))
    print("box_confidence[0, 0, 0, 0] = " + str(test_a.run(box_confidence[0, 0, 0, 0])))

