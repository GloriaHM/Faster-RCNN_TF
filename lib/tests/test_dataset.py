import _init_paths
import sys, os
import unittest
import numpy as np, cv2
import logging
import tensorflow as tf

class Test_dataset(unittest.TestCase):
    def test_get_imdb(self):
        '''
        test on how to use imdb class
        '''
        from datasets.factory import get_imdb
        imdb = get_imdb('voc_2007_trainval')
        print "imdb name", imdb.name

    def test_pascal_voc(self):
        ''' test how to use pascal voc data set
        '''
        from datasets.factory import get_imdb
        imdb = get_imdb('voc_2007_trainval')

        #To read image
        fname = imdb.image_path_from_index( imdb.image_index[0] )
        fname1 = imdb.image_path_at(0)

        assert(fname == fname1)

        img = cv2.imread( fname )
        assert(len(img.shape) == 3)

        #To read in roi db
        gt_roi = imdb.gt_roidb()

        assert(len(gt_roi) == imdb.num_images)
        print "Every gt_roi has the following entries"
        print gt_roi[0].keys()

        # seg_area are just box areas
        area = gt_roi[0]['seg_areas'][0]
        box = gt_roi[0]['boxes'][0]
        assert( area == (box[2] - box[0] +1 ) * (box[3] - box[1] + 1) )

    def test_voc_eval(self):
        '''
        understand voc eval metric
        '''
        from datasets.factory import get_imdb
        imdb = get_imdb('voc_2007_trainval')

        #sample box: [bbox 1x4, score]
        sample_box = np.array( [0, 0, 100, 100, 1] );
        #all_boxes[ pred_class ][img_index] = [ [bbox 1x4, score] ]
        all_boxes = np.tile([], (imdb.num_classes, imdb.num_images, 1)).tolist()
        all_boxes[1][0] = np.expand_dims( sample_box, 0 )

        imdb._write_voc_results_file( all_boxes )
        # when evaluating for class i
        # for a image j
        # read gt class i in image j -> gt_boxes
        # for each detected class i box, check if box has enough overlap with any of gt_boxes
        imdb.evaluate_detections(all_boxes, '/tmp');




if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(Test_dataset('test_get_imdb'))
    suite.addTest(Test_dataset('test_pascal_voc'))
    suite.addTest(Test_dataset('test_voc_eval'))

    unittest.TextTestRunner(verbosity = 2).run(suite)


