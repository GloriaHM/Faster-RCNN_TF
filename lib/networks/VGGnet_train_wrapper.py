import tensorflow as tf
from models.layer_frcnn import FRCNN
from models.layer_rpn import RPN
from models.layer_vgg_fe import VGG16BackBone

n_classes = 21
_feat_stride = [16,]
anchor_scales = [8, 16, 32]

class VGGnet_train_wrapper(object):
    def __init__(self, trainable=True):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
        self.gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'data':self.data, 'im_info':self.im_info, 'gt_boxes':self.gt_boxes})
        self.trainable = trainable
        self.setup()

        # create ops and placeholders for bbox normalization process
        with tf.variable_scope('bbox_pred', reuse=True):
            weights = tf.get_variable("kernel")
            biases = tf.get_variable("bias")

            self.bbox_weights = tf.placeholder(weights.dtype, shape=weights.get_shape())
            self.bbox_biases = tf.placeholder(biases.dtype, shape=biases.get_shape())

            self.bbox_weights_assign = weights.assign(self.bbox_weights)
            self.bbox_bias_assign = biases.assign(self.bbox_biases)


    def setup(self):
        '''
        connections:
            data
             |
        [ VGG16BackBone  ]
             |
        conv5_3, gt_boxes, im_info
             |  |      |
             |  |  [   RPN ]
             |  |      |
             |  |  rois,    rpn_cross_entropy, rpn_loss_box
             |  |   |
        [     FRCNN    ]
                 |
        cls_prob,bbox_pred, cross_entropy,loss_box

        '''
        isTrain = self.trainable

        #VGG backend
        self.vggbk = VGG16BackBone(self.data)
        conv5_3 = self.vggbk.get_output()

        #RPN
        rpn_rois, rpn_cross_entro, rpn_loss_box = \
                RPN().create(
                        conv5_3,
                        self.im_info,
                        self.gt_boxes,
                        isTrain = isTrain
                        ).get_outputs()
        #FRCNN
        self.frcnn = FRCNN().create(
                        conv5_3,
                        rpn_rois,
                        self.gt_boxes,
                        num_classes = n_classes,
                        isTrain = isTrain
                        )
        cls_prob, bbox_pred, cross_entropy, loss_box = self.frcnn.get_outputs()


        self.outputs = {}
        self.outputs['cls_prob'] = cls_prob
        self.outputs['bbox_pred'] = bbox_pred
        self.outputs['cross_entropy'] = cross_entropy
        self.outputs['loss_box'] = loss_box
        self.outputs['rpn_cross_entropy'] = rpn_cross_entro
        self.outputs['rpn_loss_box'] = rpn_loss_box

        self.outputs['rois'] = rpn_rois
        self.outputs['cls_score'] = tf.get_default_graph().get_tensor_by_name( 'cls_score/BiasAdd:0' )


    def get_output(self, key):

        try:
            res = self.outputs[key]
        except:
            raise ValueError("key does not exists")

        return res

    def load(self, data_path, session, saver, ignore_missing=False):
        if data_path.endswith('.ckpt'):
            saver.restore(session, data_path)
        else:
            self.vggbk.restore_params(session, data_path)
            self.frcnn.restore_params(session, data_path)

