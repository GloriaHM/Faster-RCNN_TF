import numpy as np
from fast_rcnn.config import cfg
import numpy.random as npr
from generate_anchors import generate_anchors
from utils.cython_bbox import bbox_overlaps
from fast_rcnn.bbox_transform import bbox_transform, bbox_transform_inv, clip_boxes
from fast_rcnn.nms_wrapper import nms
import roi_pooling_layer.roi_pooling_op as roi_pool_op

import os
import tensorflow as tf

class FRCNN(object):

    def __init__(self, DEBUG = False):
        self.DEBUG = DEBUG

    def get_outputs(self):

        return  self.cls_prob, self.bbox_pred, self.cross_entropy, self.loss_box

    def create(self,features, rpn_rois,
            gt_boxes = None,
            num_classes = 21,
            isTrain = False):
        '''
        input:
            features : Nbatch , h, w, Nchannel
            rois : Nrois , 5
            gt_boxes : Ngt, 4
        output:
            cls_prob: Nrois, pooled_h, polled_w, Nclass
            bbox_pred: Nrois, pooled_h, polled_w, 4*Nclass
            cross_entropy,
            loss_box

        connections:
             features   rpn_rois, gt_boxes
                 |              |
                 |      [     frcnn_proposal_target  ]
                 |              |
                 |           rois,          labels, bbox_targets, bbox_iw, bbox_ow
                 |   ___________|                    |
                 |   |                               |
             [    frcnn_roi_pooling ]                |
                 |                                   |
                 pool_5                              |
                 |                                   |
             [    frcnn_head        ]                |
                 |                                   |
              cls_score, cls_prob, bbox_pred         |
                |                        |           |
             [          frcnn_loss                       ]
                            |
                cross_entropy, loss_box

        '''

        if isTrain:
            rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = \
            self.frcnn_proposal_target( rpn_rois, gt_boxes,  num_classes )
        else:
            rois = rpn_rois

        pool_5 = self.frcnn_roi_pooling(features, rois, pooled_height = 7, pooled_width = 7, \
            spatial_scale = 1./16, name = "pool_5")

        cls_score, self.cls_prob, self.bbox_pred = \
                self.frcnn_head(pool_5, num_classes, isTrain=isTrain)

        if isTrain:
            self.cross_entropy, self.loss_box = \
                self.frcnn_loss( cls_score, self.bbox_pred,
                labels ,bbox_targets,bbox_inside_weights,bbox_outside_weights  )

        else:
            self.cross_entropy = None
            self.loss_box = None

        return self

    def frcnn_proposal_target(self, rpn_rois, gt_boxes, num_classes ):
        '''
        input:
            rpn_rois: Nrois , 5 (index, 4 bbox coordinates)
            gt_boxes: Ngt , 4
        output:
            rois - Nrois, 5 sampled rois during training
            labels - Nrois, class label
            bbox_targets - Nrois , 4*numclass (1-hot encoded bbox coordinates)
            bbox_inside_weights - Nrois, 4*numclass
            bbox_outside_weights - Nrois, 4*numclass
        '''
        with tf.variable_scope("frcnn_proposal_target") as scope:
            rois,labels,bbox_targets,bbox_inside_weights,bbox_outside_weights = \
                    tf.py_func(self._proposal_target_layer,
                            [rpn_rois,gt_boxes, num_classes],
                            [tf.float32,tf.float32,tf.float32,tf.float32,tf.float32])

            rois = tf.reshape(rois,[-1,5] , name = 'rois')

            labels = tf.convert_to_tensor(tf.cast(labels,tf.int32), name = 'labels')
            labels = tf.identity(labels, name = 'labels')

            bbox_targets = tf.convert_to_tensor(bbox_targets, name = 'bbox_targets')
            bbox_targets = tf.identity(bbox_targets, name = 'bbox_targets')

            bbox_inside_weights = tf.convert_to_tensor(bbox_inside_weights,
                    name = 'bbox_inside_weights')
            bbox_inside_weights = tf.identity(bbox_inside_weights,
                    name = 'bbox_inside_weights')

            bbox_outside_weights = tf.convert_to_tensor(bbox_outside_weights,
                    name = 'bbox_outside_weights')
            bbox_outside_weights = tf.identity(bbox_outside_weights,
                    name = 'bbox_outside_weights')

            return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

    def _proposal_target_layer(self, rpn_rois, gt_boxes,_num_classes):
        """
        Assign object detection proposals to ground-truth targets. Produces proposal
        classification labels and bounding-box regression targets.
        """
        def _get_bbox_regression_labels(bbox_target_data, num_classes):
            """Bounding-box regression targets (bbox_target_data) are stored in a
            compact form N x (class, tx, ty, tw, th)

            This function expands those targets into the 4-of-4*K representation used
            by the network (i.e. only one class has non-zero targets).

            Returns:
                bbox_target (ndarray): N x 4K blob of regression targets
                bbox_inside_weights (ndarray): N x 4K blob of loss weights
            """

            clss = np.array(bbox_target_data[:, 0], dtype=np.uint16, copy=True)
            bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
            bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
            inds = np.where(clss > 0)[0]
            for ind in inds:
                cls = clss[ind]
                start = 4 * cls
                end = start + 4
                bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
                bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
            return bbox_targets, bbox_inside_weights


        def _compute_targets(ex_rois, gt_rois, labels):
            """Compute bounding-box regression targets for an image."""

            assert ex_rois.shape[0] == gt_rois.shape[0]
            assert ex_rois.shape[1] == 4
            assert gt_rois.shape[1] == 4

            targets = bbox_transform(ex_rois, gt_rois)
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
                        / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
            return np.hstack(
                    (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)

        def _sample_rois(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
            """Generate a random sample of RoIs comprising foreground and background
            examples.
            """
            # overlaps: (rois x gt_boxes)
            overlaps = bbox_overlaps(
                np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
                np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
            gt_assignment = overlaps.argmax(axis=1)
            max_overlaps = overlaps.max(axis=1)
            labels = gt_boxes[gt_assignment, 4]

            # Select foreground RoIs as those with >= FG_THRESH overlap
            fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
            # Guard against the case when an image has fewer than fg_rois_per_image
            # foreground RoIs
            fg_rois_per_this_image = int(min(fg_rois_per_image, fg_inds.size))
            # Sample foreground regions without replacement
            if fg_inds.size > 0:
                if self.DEBUG:
                    np.random.seed(0);
                fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

            # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
            bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                               (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
            # Compute number of background RoIs to take from this image (guarding
            # against there being fewer than desired)
            bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
            bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
            # Sample background regions without replacement
            if bg_inds.size > 0:
                if self.DEBUG:
                    np.random.seed(0);
                bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

            # The indices that we're selecting (both fg and bg)
            keep_inds = np.append(fg_inds, bg_inds)
            # Select sampled values from various arrays:
            labels = labels[keep_inds]
            # Clamp labels for the background RoIs to 0
            labels[fg_rois_per_this_image:] = 0
            rois = all_rois[keep_inds]

            bbox_target_data = _compute_targets(
                rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)

            bbox_targets, bbox_inside_weights = \
                _get_bbox_regression_labels(bbox_target_data, num_classes)

            return labels, rois, bbox_targets, bbox_inside_weights

        # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
        # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
        all_rois = rpn_rois
        # TODO(rbg): it's annoying that sometimes I have extra info before
        # and other times after box coordinates -- normalize to one format

        # Include ground-truth boxes in the set of candidate rois
        zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        all_rois = np.vstack(
            (all_rois, np.hstack((zeros, gt_boxes[:, :-1])))
        )

        # Sanity check: single batch only
        assert np.all(all_rois[:, 0] == 0), \
                'Only single item batches are supported'

        num_images = 1
        rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
        fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

        # Sample rois with classification labels and bounding box regression
        # targets
        labels, rois, bbox_targets, bbox_inside_weights = _sample_rois(
            all_rois, gt_boxes, fg_rois_per_image,
            rois_per_image, _num_classes)

        if self.DEBUG:
            print 'num fg: {}'.format((labels > 0).sum())
            print 'num bg: {}'.format((labels == 0).sum())
            #_count += 1
            #_fg_num += (labels > 0).sum()
            #_bg_num += (labels == 0).sum()
            #print 'num fg avg: {}'.format(_fg_num / _count)
            #print 'num bg avg: {}'.format(_bg_num / _count)
            #print 'ratio: {:.3f}'.format(float(_fg_num) / float(_bg_num))

        rois = rois.reshape(-1,5)
        labels = labels.reshape(-1,1)
        bbox_targets = bbox_targets.reshape(-1,_num_classes*4)
        bbox_inside_weights = bbox_inside_weights.reshape(-1,_num_classes*4)

        bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)

        return rois,labels,bbox_targets,bbox_inside_weights,bbox_outside_weights


    def frcnn_roi_pooling(self, features, rois, pooled_height, pooled_width, \
            spatial_scale, name = "pool_5"):
        '''
        input:
            features : Nbatch , h , w , C
            rois : NROI x 5
        output:
            pooled_features: NROI x 7 x 7 x C
        '''

        return roi_pool_op.roi_pool(features, rois,
                                    pooled_height,
                                    pooled_width,
                                    spatial_scale,
                                    name=name)[0]


    def frcnn_head(self, pooled_feature, num_classes, isTrain=False ):
        '''
        input:
            roi pooled features: Nrois , poolh, pool w, Channel
        output:
            cls_score: Nrois ,  Nlasses
            cls_prob: Nrois ,  Nclasses
            bbox_pred: Nrois , 4*Nclasses

        connections:
             .fc(4096, name='fc6')
             .dropout(0.5, name='drop6')
             .fc(4096, name='fc7')
             .dropout(0.5, name='drop7')
             .fc(n_classes, relu=False, name='cls_score')
             .softmax(name='cls_prob'))

        '''

        dim = 1
        for d in (pooled_feature).shape[1:].as_list():
            dim *= d
            pooled_feature_reshape = tf.reshape(
                    tf.transpose(pooled_feature,[0,3,1,2]), [-1, dim])

        fc6  = tf.layers.dense( pooled_feature_reshape, \
                4096, activation = tf.nn.relu, name = 'fc6',
                kernel_initializer = tf.truncated_normal_initializer(0.0, stddev=0.01))
        rate = 0 if self.DEBUG else 0.5
        dp6 = tf.layers.dropout( fc6, rate = rate, \
                training = isTrain , name = 'drop6')
        fc7  =  tf.layers.dense( dp6, 4096, activation = tf.nn.relu, name = 'fc7',
                kernel_initializer = tf.truncated_normal_initializer(0.0, stddev=0.01))
        dp7 = tf.layers.dropout( fc7, rate = rate, \
                training = isTrain , name = 'drop7')

        cls_score = tf.layers.dense( dp7, num_classes, activation = None,
                name = 'cls_score' ,
                kernel_initializer = tf.truncated_normal_initializer(0.0, stddev=0.01)
                )
        cls_prob = tf.nn.softmax( cls_score, name = 'cls_prob')
        bbox_pred = tf.layers.dense(dp7, 4*num_classes, activation = None,
                name = 'bbox_pred',
                kernel_initializer =  tf.truncated_normal_initializer(0.0, stddev=0.001)
                )

        return cls_score, cls_prob, bbox_pred


    def frcnn_loss(self, cls_score, bbox_pred,
            labels,bbox_targets,bbox_inside_weights,bbox_outside_weights
            ):

        def _modified_smooth_l1( sigma, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights):
            """
            ResultLoss = outside_weights * SmoothL1(inside_weights * (bbox_pred - bbox_targets))
            SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                          |x| - 0.5 / sigma^2,    otherwise
            """
            sigma2 = sigma * sigma

            inside_mul = tf.multiply(bbox_inside_weights, tf.subtract(bbox_pred, bbox_targets))

            smooth_l1_sign = tf.cast(tf.less(tf.abs(inside_mul), 1.0 / sigma2), tf.float32)
            smooth_l1_option1 = tf.multiply(tf.multiply(inside_mul, inside_mul), 0.5 * sigma2)
            smooth_l1_option2 = tf.subtract(tf.abs(inside_mul), 0.5 / sigma2)
            smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
                                      tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))

            outside_mul = tf.multiply(bbox_outside_weights, smooth_l1_result)

            return outside_mul

        # classification loss
        label = tf.reshape(labels,[-1])
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label), name = 'cross_entropy')

        # bounding box regression L1 loss
        smooth_l1 = _modified_smooth_l1(1.0, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)
        loss_box = tf.reduce_mean(tf.reduce_sum(smooth_l1, reduction_indices=[1]), name = 'loss_box')

        return cross_entropy, loss_box

    def restore_params(self, session, weights):
        npz = np.load(weights)

        with tf.Session() as sess:
            sess.run(tf.assign(
                tf.get_default_graph().get_tensor_by_name('fc6/kernel:0'), npz['fc6_W']
                ) )
            sess.run(tf.assign(
                tf.get_default_graph().get_tensor_by_name('fc6/bias:0'), npz['fc6_b']
                ) )

            sess.run(tf.assign(
                tf.get_default_graph().get_tensor_by_name('fc7/kernel:0'), npz['fc7_W']
                ) )
            sess.run(tf.assign(
                tf.get_default_graph().get_tensor_by_name('fc7/bias:0'), npz['fc7_b']
                ) )

