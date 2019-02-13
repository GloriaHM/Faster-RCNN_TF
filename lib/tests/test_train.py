import _init_paths
import sys, os
import unittest
import numpy as np, cv2
import logging
import tensorflow as tf
import pickle

from data_loader.data_loader import ROIDataLoader
from models.factory import get_network
from trainer_tester.train import SolverWrapper
from fast_rcnn.config import cfg,cfg_from_file, cfg_from_list, get_output_dir

class Test_train(unittest.TestCase):
    def setUp(self):
        from datasets.factory import get_imdb
        self.imdb = get_imdb('voc_2007_trainval')
        print "imdb name", self.imdb.name

    def test_preprocess_train(self):
        '''
        test on how to use imdb class
        '''
        imdb = self.imdb
        loader = ROIDataLoader()
        loader.preprocess_train(imdb)
        roidb0 = pickle.load( open('tests/test_data_loader_data/roidb0.pkl', 'rb') )

        assert( roidb0.keys() == loader._roidb[0].keys() )
        for i in ['gt_classes', 'max_classes', 'max_overlaps', 'seg_areas']:
            assert(all(roidb0[i]== loader._roidb[0][i]))
        assert( roidb0['image']== loader._roidb[0]['image'] )
        assert( loader._roidb[0]['max_overlaps'].shape[0]
                == loader._roidb[0]['gt_overlaps'].shape[0] )
        assert( np.array_equal( roidb0['gt_overlaps'].todense() ,
                loader._roidb[0]['gt_overlaps'].todense() ) )

    def test_train(self):
        from fast_rcnn.config import cfg
        cfg.TRAIN.HAS_RPN = True
        cfg.TRAIN.IMS_PER_BATCH = 1
        cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = True

        ## Template of how context should call trainer
        imdb = self.imdb

        def get_data_layer():
            """return a data layer."""
            if cfg.TRAIN.HAS_RPN:
                if cfg.IS_MULTISCALE:
                    raise
                else:
                    layer = ROIDataLoader(imdb, imdb.num_classes)
            else:
                raise

            return layer

        dataloader = get_data_layer()

        network = get_network('VGG_train')

        outputdir = get_output_dir(imdb, None)

        saver = tf.train.Saver(max_to_keep=100)
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

            dataloader.preprocess_train()

            trainer = SolverWrapper(sess, saver, network, dataloader, outputdir)


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(Test_train('test_train'))

    unittest.TextTestRunner(verbosity = 2).run(suite)

