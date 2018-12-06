import tensorlayer as tl
import logging
import numpy as np
import tensorflow as tf
import os
class VGG16BackBone(object):
    def __init__(self, input_place_holder):
        #The following code is used to offset the preprocessing layer in tl.vgg16
        mean = tf.constant([123.68, 116.779, 103.939],
                    dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        y = input_place_holder+mean
        y = y/255.
        self.vgg = tl.models.VGG16(y, end_with = 'conv5_3')

    def get_output(self):
        return self.vgg.outputs

    def restore_params(self,  sess, weights):
        from tensorlayer.files import assign_params
        logging.info("Restore pre-trained parameters")
        npz = np.load(weights)
        params = []
        for val in sorted(npz.items()):
            print("  Loading params %s" % str(val[1].shape))
            params.append(val[1])
            if len(self.vgg.all_params) == len(params):
                break

        assign_params(sess, params, self.vgg)
        del params

