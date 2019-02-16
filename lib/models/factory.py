# --------------------------------------------------------
# SubCNN_TF
# Copyright (c) 2016 CVGL Stanford
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

import models.VGGnet_train_wrapper
import pdb
import tensorflow as tf
import roi_pooling_layer.roi_pooling_op as roi_pool_op
import roi_pooling_layer.roi_pooling_op_grad

#__sets['VGGnet_train'] = networks.VGGnet_train()

#__sets['VGGnet_test'] = networks.VGGnet_test()


def get_network(name):
    """Get a network by name."""
    #if not __sets.has_key(name):
    #    raise KeyError('Unknown dataset: {}'.format(name))
    #return __sets[name]
    if name.split('_')[1] == 'test':
       return models.VGGnet_train_wrapper.VGGnet_train_wrapper(trainable = False)
    elif name.split('_')[1] == 'train':
       return models.VGGnet_train_wrapper.VGGnet_train_wrapper()
    else:
       raise KeyError('Unknown dataset: {}'.format(name))


def list_networks():
    """List all registered imdbs."""
    return __sets.keys()
