import _init_paths
import sys, os
import unittest
import numpy as np, cv2
import logging
import tensorflow as tf
import pickle

from data_loader.data_loader import ROIDataLoader
class Test_data_loader(unittest.TestCase):
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

    def test_get_minibatch(self):
        from fast_rcnn.config import cfg
        cfg.TRAIN.HAS_RPN = True
        cfg.TRAIN.IMS_PER_BATCH = 1

        imdb = self.imdb
        loader = ROIDataLoader()
        loader.preprocess_train(imdb)
        loader.init_sampler(imdb.num_classes)
        blobs = loader.get_next_batch()
        print "blobs has the following keys"
        print blobs.keys()




if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(Test_data_loader('test_preprocess_train'))
    suite.addTest(Test_data_loader('test_get_minibatch'))

    unittest.TextTestRunner(verbosity = 2).run(suite)
