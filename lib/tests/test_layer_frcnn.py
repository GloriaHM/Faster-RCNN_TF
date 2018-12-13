import sys, os
import unittest
import numpy as np
import logging
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from models.layer_frcnn import FRCNN
import pickle
from fast_rcnn.config import cfg

class Test_layer_rpn(unittest.TestCase):

    def setUp(self):
        self.datadir = 'tests/test_layer_frcnn_data/'

    def test_frcnn_proposal_target(self):
        tf.reset_default_graph()
        with open(self.datadir+'test_proposal_target.pkl', 'rb') as fid:
            df = pickle.load(fid)

        with open(self.datadir+'train_cfg.pkl') as fid:
            cfg1 = pickle.load(fid)

        roiin,gt, \
        rois_ref,labels_ref,bbox_targets_ref,bbox_iw_ref,bbox_ow_ref = df

        frcnn = FRCNN( DEBUG = True)
        cfg.TRAIN = cfg1.TRAIN
        n_classes = 21

        rois, labels, bbox_targets, bbox_iw, bbox_ow = \
        frcnn.frcnn_proposal_target( roiin, gt, n_classes )

        with tf.Session() as sess:
            rois_r, labels_r, bbox_targets_r, bbox_iw_r, bbox_ow_r = \
                    sess.run([ rois, labels, bbox_targets, bbox_iw, bbox_ow ])

        assert( np.equal(rois_r, rois_ref).all() )
        assert( np.equal(labels_r, labels_ref).all() )
        assert( np.equal(bbox_targets_r, bbox_targets_ref).all() )
        assert( np.equal(bbox_iw_r, bbox_iw_ref).all() )
        assert( np.equal(bbox_ow_r, bbox_ow_ref).all() )

    def test_frcnn_roi_pooling(self):
        '''
        need to unzip test_roi_pooling.pkl.zip first
        '''
        tf.reset_default_graph()
        with open(self.datadir+'test_roi_pooling.pkl', 'rb') as fid:
            df = pickle.load(fid)

        reshapein,rois,pool_5_ref = df

        frcnn = FRCNN( DEBUG = True)
        input_ph = tf.placeholder( dtype = tf.float32, shape = [None,None,None,None] )
        input_ph1 = tf.placeholder( dtype = tf.float32, shape = [None,None] )
        pool5 = frcnn.frcnn_roi_pooling( input_ph, input_ph1, 7, 7, 1./16, name = 'pool_5' )

        with tf.Session() as sess:

            pool_5_r = sess.run( pool5 , feed_dict =
                    {input_ph: reshapein,
                     input_ph1 : rois[0] })

        assert( np.equal(pool_5_ref, pool_5_r).all() )


    def test_frcnn_head(self):
        frcnn = FRCNN( DEBUG = True)
        input_ph = tf.placeholder( dtype = tf.float32, shape = [256, 7, 7 ,512 ] )
        cls, clsp, bbox = frcnn.frcnn_head( input_ph, num_classes = 21, isTrain = True )

        assert(cls.shape == [256,21])
        assert(clsp.shape == [256,21])
        assert(bbox.shape == [256,4*21])

    def test_frcnn_loss(self):
        pass

    def test_frcnn_all(self):
        '''
        it is too larget to store test_all.pkl
        to regenerate:
            get input output from frcnn and all weights/bias tensors
            make sure dropout keep prob is 1
            before npr.choice in sampling_roi function, reset rand seed to 0
            store all tensor values into self.datadir+'test_all.pkl'
        '''
        tf.reset_default_graph()
        with open(self.datadir+'test_all.pkl', 'rb') as fid:
            df = pickle.load(fid)

        reshapein , rpn_rois , gt, \
        cls_prob_res,bbox_pred_res, \
        w0, b0, w1, b1, w2, b2, w3, b3 = df

        frcnn = FRCNN( DEBUG = True)
        model = frcnn.create(
                reshapein, rpn_rois, gt,
                num_classes = 21, isTrain = True
                )
        with open(self.datadir+'train_cfg.pkl') as fid:
            cfg1 = pickle.load(fid)

        cfg.TRAIN = cfg1.TRAIN
        cls_prob, bbox_pred, cross_entropy, loss_box = model.get_outputs()

        with tf.Session() as sess:
            sess.run(tf.assign(
                tf.get_default_graph().get_tensor_by_name('fc6/kernel:0'), w0
                ) )
            sess.run(tf.assign(
                tf.get_default_graph().get_tensor_by_name('fc6/bias:0'), b0
                ) )

            sess.run(tf.assign(
                tf.get_default_graph().get_tensor_by_name('fc7/kernel:0'), w1
                ) )
            sess.run(tf.assign(
                tf.get_default_graph().get_tensor_by_name('fc7/bias:0'), b1
                ) )

            sess.run(tf.assign(
                tf.get_default_graph().get_tensor_by_name('cls_score/kernel:0'), w2
                ))
            sess.run(tf.assign(
                tf.get_default_graph().get_tensor_by_name('cls_score/bias:0'), b2
                ))

            sess.run(tf.assign(
                tf.get_default_graph().get_tensor_by_name('bbox_pred/kernel:0'), w3
                ))
            sess.run(tf.assign(
                tf.get_default_graph().get_tensor_by_name('bbox_pred/bias:0'), b3
                ))

            cls_prob_r, bbox_pred_r, \
                ce_r, lb_r = sess.run([ cls_prob, bbox_pred, cross_entropy, loss_box ])

        assert( np.equal(cls_prob_res, cls_prob_r).all() )
        assert( np.equal(bbox_pred_res, bbox_pred_r).all() )

    def test_frcnn_testmode(self):
        tf.reset_default_graph()

        reshapein = tf.placeholder(dtype = tf.float32, shape = (None, None, None, 512))
        rpn_rois = tf.placeholder( dtype = tf.float32, shape = ( None, 5 )  )

        frcnn = FRCNN( DEBUG = True)
        model = frcnn.create(
                reshapein, rpn_rois,
                num_classes = 21,
                isTrain = False
                )
        cls_prob, bbox_pred, cross_entropy, loss_box = model.get_outputs()

if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(Test_layer_rpn('test_frcnn_proposal_target'))
    #suite.addTest(Test_layer_rpn('test_frcnn_roi_pooling'))
    suite.addTest(Test_layer_rpn('test_frcnn_head'))
    #suite.addTest(Test_layer_rpn('test_frcnn_all'))
    suite.addTest(Test_layer_rpn('test_frcnn_testmode'))

    unittest.TextTestRunner(verbosity = 2).run(suite)

