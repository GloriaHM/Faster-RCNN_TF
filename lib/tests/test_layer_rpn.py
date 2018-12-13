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

    def test_rpn_head(self):
        tf.reset_default_graph()
        with open(self.datadir+'test_rpn_head.pkl', 'rb') as fid:
            df = pickle.load(fid)

        reshapein, out1_ref, out2_ref, w0, b0,w1,b1,w2,b2 = tuple(df)

        input_tensor = tf.placeholder(dtype = tf.float32, shape = (None,None,None,512))
        rpn = RPN(DEBUG = True)
        rpn_cls_score, rpn_bbox_pred = rpn.rpn_head(input_tensor, nfilters = 512)

        with tf.Session() as sess:
            sess.run(tf.assign(
                tf.get_default_graph().get_tensor_by_name('rpn_conv/3x3/kernel:0'),
                w0))
            sess.run(tf.assign(
                tf.get_default_graph().get_tensor_by_name('rpn_conv/3x3/bias:0'),
                b0))

            sess.run(tf.assign(
                tf.get_default_graph().get_tensor_by_name('rpn_cls_score/kernel:0'),
                w1))
            sess.run(tf.assign(
                tf.get_default_graph().get_tensor_by_name('rpn_cls_score/bias:0'),
                b1))

            sess.run(tf.assign(
                tf.get_default_graph().get_tensor_by_name('rpn_bbox_pred/kernel:0'),
                w2))
            sess.run(tf.assign(
                tf.get_default_graph().get_tensor_by_name('rpn_bbox_pred/bias:0'),
                b2))

            out1, out2 = sess.run([ rpn_cls_score , rpn_bbox_pred],
                    feed_dict = {input_tensor : reshapein})

        assert( np.equal(out1, out1_ref).all() )
        assert( np.equal(out2, out2_ref).all() )


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
        rpn_cls_prob_reshape = rpn.rpn_cls_score_reshape(input_tensor)

        with tf.Session() as sess:
            res = sess.run(rpn_cls_prob_reshape, feed_dict = {input_tensor:rpn_cls_score})
        assert( np.equal(res[1], res_ref).all() )


    def test_rpn_loss(self):
        tf.reset_default_graph()
        with open(self.datadir+'test_rpn_loss.pkl', 'rb') as fid:
            df = pickle.load(fid)
        rpn_cls_score, rpn_bbox_pred, rpn_labels,\
        rpn_bbox_targets, rpn_bbox_inside_weights, \
        rpn_bbox_outside_weights, cross_entropy_ref, bbox_loss_ref = df

        rpn = RPN(DEBUG = True)
        cross_entropy, bbox_loss = rpn.rpn_loss( rpn_cls_score,
                rpn_bbox_pred,
                rpn_labels,
                rpn_bbox_targets,
                rpn_bbox_inside_weights,
                rpn_bbox_outside_weights )

        with tf.Session() as sess:
            cross_entropy_res, bbox_loss_res = sess.run( [cross_entropy, bbox_loss] )

        assert( np.equal(cross_entropy_ref, cross_entropy_res).all() )
        assert( np.equal(bbox_loss_ref, bbox_loss_res).all() )


    def test_proposal_layer(self):
        tf.reset_default_graph()
        with open(self.datadir+'test_proposal_layer.pkl', 'rb') as fid:
            df = pickle.load(fid)
        rpn_cls_prob_reshape = df['rpn_cls_prob_reshape']
        rpn_bbox_pred = df['rpn_bbox_pred']
        im_info = df['im_info']
        cfg_key = df['cfg_key']
        res_ref = df['res']

        rpn = RPN(DEBUG = True)
        blobs = rpn.proposal_layer(
            rpn_cls_prob_reshape,
            rpn_bbox_pred,
            im_info,
            cfg_key,
            _feat_stride = [16,],
            anchor_scales = [8, 16, 32]
            )

        with tf.Session() as sess:
            res = sess.run(blobs)

        assert( np.equal(res, res_ref).all() )


    def test_create(self):

        tf.reset_default_graph()
        input_tensor = tf.placeholder(dtype = tf.float32, shape = (None, None, None, 512))
        im_info = tf.placeholder( dtype = tf.float32, shape = [None,None] )
        gt_boxes = tf.placeholder( dtype = tf.float32, shape = [None, None] )

        rpn = RPN(DEBUG = True).create( input_tensor, im_info, gt_boxes, isTrain = True )

    def test_all(self):
        tf.reset_default_graph()
        with open(self.datadir+'test_rpn_all.pkl', 'rb') as fid:
            df = pickle.load(fid)

        features, im_info, gt, \
        proposals_ref, rpn_cross_entropy_ref, rpn_loss_box_ref, \
        w0,b0,w1,b1,w2,b2 = df

        input_tensor = tf.placeholder(dtype = tf.float32, shape = (None, None, None, 512))
        im_info_tensor= tf.placeholder( dtype = tf.float32, shape = [None,None] )
        gt_boxes_tensor = tf.placeholder( dtype = tf.float32, shape = [None, None] )

        rpn = RPN(DEBUG = True).create( input_tensor, im_info_tensor,
                gt_boxes_tensor, isTrain = True )

        proposal, cross_entro, bbox_loss = rpn.get_outputs()

        with tf.Session() as sess:

            sess.run(tf.assign(
                tf.get_default_graph().get_tensor_by_name('rpn_conv/3x3/kernel:0'),
                w0))
            sess.run(tf.assign(
                tf.get_default_graph().get_tensor_by_name('rpn_conv/3x3/bias:0'),
                b0))

            sess.run(tf.assign(
                tf.get_default_graph().get_tensor_by_name('rpn_cls_score/kernel:0'),
                w1))
            sess.run(tf.assign(
                tf.get_default_graph().get_tensor_by_name('rpn_cls_score/bias:0'),
                b1))

            sess.run(tf.assign(
                tf.get_default_graph().get_tensor_by_name('rpn_bbox_pred/kernel:0'),
                w2))
            sess.run(tf.assign(
                tf.get_default_graph().get_tensor_by_name('rpn_bbox_pred/bias:0'),
                b2))

            proposal_r, cross_entro_r, bbox_loss_r = sess.run( [proposal, cross_entro,
                bbox_loss ] ,
                feed_dict = {
                    input_tensor : features,
                    im_info_tensor: im_info,
                    gt_boxes_tensor: gt
                    })

        assert( np.equal(proposals_ref, proposal_r).all() )
        assert( np.equal(cross_entro_r, rpn_cross_entropy_ref).all() )
        assert( np.equal(bbox_loss_r, rpn_loss_box_ref).all() )

    def test_rpn_testmode(self):
        tf.reset_default_graph()
        input_tensor = tf.placeholder(dtype = tf.float32, shape = (None, None, None, 512))
        im_info_tensor= tf.placeholder( dtype = tf.float32, shape = [None,None] )

        rpn = RPN(DEBUG = True).create( input_tensor, im_info_tensor,
                isTrain = False )

        proposals, rpn_cross_entropy, rpn_loss_box = rpn.get_outputs()


if __name__ == "__main__":
    suite = unittest.TestSuite()

    suite.addTest(Test_layer_rpn('test_rpn_head'))
    suite.addTest(Test_layer_rpn('test_rpn_anchor_target_layer'))
    suite.addTest(Test_layer_rpn('test_rpn_cls_score_reshpae'))
    suite.addTest(Test_layer_rpn('test_rpn_loss'))
    suite.addTest(Test_layer_rpn('test_proposal_layer'))
    suite.addTest(Test_layer_rpn('test_create'))
    suite.addTest(Test_layer_rpn('test_all'))
    suite.addTest(Test_layer_rpn('test_rpn_testmode'))

    unittest.TextTestRunner(verbosity = 2).run(suite)


