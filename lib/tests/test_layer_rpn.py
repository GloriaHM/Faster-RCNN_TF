import sys, os
import unittest
import numpy as np, cv2
import logging
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import tensorlayer as tl
from models.layer_rpn import RPN
import pickle

class Test_layer_rpn(unittest.TestCase):

    def setUp(self):
        self.datadir = 'tests/test_layer_rpn_data/'


    def test_rpn_anchor_target_layer(self):
        tf.reset_default_graph()
        with open(self.datadir+'test_anchor_target.pkl', 'rb') as fid:
            df = pickle.load(fid)

        with open(self.datadir+'/test_anchor_target_refout.pkl', 'rb') as fid:
            res_ref = pickle.load(fid)

        gt_boxes = df['gt_boxes']
        im_info = df['im_info']
        rpn_cls_score = df['rpn_cls_score']

        rpn_labels_ref = res_ref['rpn_labels']
        rpn_bbox_targets_ref = res_ref['rpn_bbox_targets']
        rpn_bbox_inside_weights_ref = res_ref['rpn_bbox_inside_weights']
        rpn_bbox_outside_weights_ref = res_ref['rpn_bbox_outside_weights']

        rpn = RPN(DEBUG = True)
        l,b, iw, ow = rpn.rpn_anchor_target_layer(
                tf.shape(rpn_cls_score),
                gt_boxes,
                im_info,
                _feat_stride = [16,],
                anchor_scales = [8, 16, 32],
                name = 'rpn_anchor_target_layer'
                )

        with tf.Session() as sess:
            label, boxtarget, iw, ow = sess.run([ l, b, iw, ow ])

        print label, label, iw, ow

        assert(np.array_equal(label, rpn_labels_ref) )
        assert(np.array_equal(boxtarget, rpn_bbox_targets_ref) )
        assert(np.array_equal(iw, rpn_bbox_inside_weights_ref) )
        assert(np.array_equal(ow, rpn_bbox_outside_weights_ref) )

    def test_rpn_cls_score_reshpae(self):
        tf.reset_default_graph()
        with open(self.datadir+'test_rpn_cls_reshape.pkl', 'rb') as fid:
            df = pickle.load(fid)
        rpn_cls_score = df['input']
        res_ref = df['output']

        rpn = RPN(DEBUG = True)
        input_tensor = tf.placeholder( dtype = tf.float32, shape = (None, None, None, None) )
        rpn_cls_prob_reshape = rpn._rpn_cls_score_reshape(input_tensor)

        with tf.Session() as sess:
            res = sess.run(rpn_cls_prob_reshape, feed_dict = {input_tensor:rpn_cls_score})

        assert( np.equal(res, res_ref).all() )



if __name__ == "__main__":
    suite = unittest.TestSuite()
    #suite.addTest(Test_layer_rpn('test_rpn_anchor_target_layer'))
    suite.addTest(Test_layer_rpn('test_rpn_cls_score_reshpae'))

    unittest.TextTestRunner(verbosity = 2).run(suite)


