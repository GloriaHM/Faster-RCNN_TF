#raw roi in data set -> filter/preprocessing -> roi_datalayer which provides shuffle and get minibatch
import numpy as np
import PIL
from fast_rcnn.config import cfg
from minibatch import get_minibatch

class ROIDataLoader(object):

    def __init__(self):
        self._roidb = None

    def preprocess_train(self, imdb):
        '''
        augment image
        add derived roi statistics
        filter roi for training
        '''

        if cfg.TRAIN.USE_FLIPPED:
            print 'Appending horizontally-flipped training examples...'
            imdb.append_flipped_images()
            print 'done'

        #preprocessing
        def prepare_roidb(imdb):
            """Enrich the imdb's roidb by adding some derived quantities that
            are useful for training. This function precomputes the maximum
            overlap, taken over ground-truth boxes, between each ROI and
            each ground-truth box. The class with maximum overlap is also
            recorded.
            """
            sizes = [PIL.Image.open(imdb.image_path_at(i)).size
                     for i in xrange(imdb.num_images)]
            roidb = imdb.roidb
            for i in xrange(len(imdb.image_index)):
                roidb[i]['image'] = imdb.image_path_at(i)
                roidb[i]['width'] = sizes[i][0]
                roidb[i]['height'] = sizes[i][1]
                # need gt_overlaps as a dense array for argmax
                # gt_ovelaps Nobj x Nclasses matrix, records overlap of ith object
                # with jth class
                gt_overlaps = roidb[i]['gt_overlaps'].toarray()
                # max overlap with gt over classes (columns)
                max_overlaps = gt_overlaps.max(axis=1)
                # gt class that had the max overlap
                max_classes = gt_overlaps.argmax(axis=1)
                roidb[i]['max_classes'] = max_classes
                roidb[i]['max_overlaps'] = max_overlaps
                # sanity checks
                # max overlap of 0 => class should be zero (background)
                zero_inds = np.where(max_overlaps == 0)[0]
                assert all(max_classes[zero_inds] == 0)
                # max overlap > 0 => class should not be zero (must be a fg class)
                nonzero_inds = np.where(max_overlaps > 0)[0]
                assert all(max_classes[nonzero_inds] != 0)

            return imdb.roidb

        roidb = prepare_roidb(imdb)

        #filter roi db
        def filter_roidb(roidb):
            """Remove roidb entries that have no usable RoIs."""

            def is_valid(entry):
                # Valid images have:
                #   (1) At least one foreground RoI OR
                #   (2) At least one background RoI
                overlaps = entry['max_overlaps']
                # find boxes with sufficient overlap
                fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
                # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
                bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                                   (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
                # image is only valid if such boxes exist
                valid = len(fg_inds) > 0 or len(bg_inds) > 0
                return valid

            num = len(roidb)
            filtered_roidb = [entry for entry in roidb if is_valid(entry)]
            num_after = len(filtered_roidb)
            print 'Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                               num, num_after)
            return filtered_roidb

        roidb = filter_roidb(roidb)

        self._roidb = roidb

    def preprocess_test(self, imdb):
        '''
        '''
        pass


    def init_sampler(self, num_classes):
        """Set the roidb to be used by this layer during training."""
        self._num_classes = num_classes
        self._shuffle_roidb_inds()


    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if cfg.TRAIN.HAS_RPN:
            if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
                self._shuffle_roidb_inds()

            db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
            self._cur += cfg.TRAIN.IMS_PER_BATCH
        else:
            # sample images
            db_inds = np.zeros((cfg.TRAIN.IMS_PER_BATCH), dtype=np.int32)
            i = 0
            while (i < cfg.TRAIN.IMS_PER_BATCH):
                ind = self._perm[self._cur]
                num_objs = self._roidb[ind]['boxes'].shape[0]
                if num_objs != 0:
                    db_inds[i] = ind
                    i += 1

                self._cur += 1
                if self._cur >= len(self._roidb):
                    self._shuffle_roidb_inds()

        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.

        If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
        separate process and made available through self._blob_queue.
        """
        db_inds = self._get_next_minibatch_inds()
        minibatch_db = [self._roidb[i] for i in db_inds]
        return get_minibatch(minibatch_db, self._num_classes)

    def get_next_batch(self):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()
        return blobs
