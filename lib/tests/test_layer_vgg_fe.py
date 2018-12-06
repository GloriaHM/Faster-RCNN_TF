import sys, os
import unittest
import numpy as np, cv2
import logging
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import tensorlayer as tl
class Test_layer_vgg_fe(unittest.TestCase):

    def setUp(self):
        self.datadir = 'tests/test_layer_vgg_fe_data/'

    def restore_params(self, md, sess):
            # we have to customize the restore param function a bit
            from tensorlayer.files import assign_params
            logging.info("Restore pre-trained parameters")
            npz = np.load(os.path.join('../data/pretrain_model', 'vgg16_weights.npz'))

            params = []
            for val in sorted(npz.items()):
                print("  Loading params %s" % str(val[1].shape))
                params.append(val[1])
                if len(md.all_params) == len(params):
                    break

            assign_params(sess, params, md.net)
            del params

    def test_keras_vgg(self):
        tf.reset_default_graph()
        vgg16 = tf.keras.applications.VGG16(weights='imagenet')
        img_file = '../data/demo/000456.jpg'

        # keras ref model
        img = image.load_img(img_file, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        label = vgg16.predict(x)

        #tensor layer vgg16 backend, which can be reusable
        x = tf.placeholder(tf.float32, [None, 224, 224, 3])
        vgg = tl.models.VGG16(x)

        with tf.Session() as sess:
            self.restore_params(vgg, sess)
            probs = tf.nn.softmax(vgg.outputs)
            res = sess.run(probs , feed_dict = { x : np.expand_dims( image.img_to_array(img) , axis = 0) / 255 } )

        #print res
        assert( np.max( abs(label - res) ) < 1e-5 )
        assert( np.argmax(label) == np.argmax(res) )

    def test_layer_vgg_fe(self):
        import pickle
        from models.layer_vgg_fe import VGG16BackBone
        tf.reset_default_graph()

        blobs = pickle.load( open( self.datadir +'blobs0.pkl', 'rb' ))
        res_ref = pickle.load( open( self.datadir +'convout.pkl', 'rb' ))
        x = tf.placeholder(tf.float32, [None, None, None, 3])

        #mean = tf.constant([123.68, 116.779, 103.939],
        #        dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        #y = x+mean
        #y = y/255.
        #vgg = tl.models.VGG16(y, end_with = 'conv5_3')

        #with tf.Session() as sess:
        #    self.restore_params(vgg, sess)
        #    out = vgg.outputs
        #    res = sess.run(out, feed_dict = {x: blobs})

        vgg = VGG16BackBone(x)
        with tf.Session() as sess:
            vgg.restore_params(sess,
                    weights = os.path.join('../data/pretrain_model', 'vgg16_weights.npz'))
            out = vgg.get_output()
            res = sess.run(out, feed_dict = {x: blobs})

        diff = (res[:,0:-1,:,:] - res_ref).reshape(1,1,1,-1).squeeze()
        print "max error {}, min error {}".format(diff.max(), diff.min())



if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(Test_layer_vgg_fe('test_keras_vgg'))
    suite.addTest(Test_layer_vgg_fe('test_layer_vgg_fe'))

    unittest.TextTestRunner(verbosity = 2).run(suite)






