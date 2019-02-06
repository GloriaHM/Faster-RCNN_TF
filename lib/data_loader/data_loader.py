#raw roi in data set -> filter/preprocessing -> roi_datalayer which provides shuffle and get minibatch
import numpy as np
import PIL
from fast_rcnn.config import cfg
from minibatch import get_minibatch
from fast_rcnn.bbox_transform import bbox_transform
from utils.cython_bbox import bbox_overlaps

class ROIDataLoader(object):

    def __init__(self, imdb, num_classes):
        self._roidb = None
        self.bbox_means = None
        self.bbox_stds = None
        self.imdb = imdb
        self._num_classes = num_classes

    def preprocess_train(self):
        '''
        augment image
        add derived roi statistics
        filter roi for training
        '''

        if cfg.TRAIN.USE_FLIPPED:
            print 'Appending horizontally-flipped training examples...'
            self.imdb.append_flipped_images()
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

            return roidb

        roidb = prepare_roidb(self.imdb)

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

        def add_bbox_regression_targets(roidb):
            """Add information needed to train bounding-box regressors."""
            assert len(roidb) > 0
            assert 'max_classes' in roidb[0], 'Did you call prepare_roidb first?'

            num_images = len(roidb)
            # Infer number of classes from the number of columns in gt_overlaps
            num_classes = roidb[0]['gt_overlaps'].shape[1]
            for im_i in xrange(num_images):
                rois = roidb[im_i]['boxes']
                max_overlaps = roidb[im_i]['max_overlaps']
                max_classes = roidb[im_i]['max_classes']
                roidb[im_i]['bbox_targets'] = \
                        _compute_targets(rois, max_overlaps, max_classes)

            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Use fixed / precomputed "means" and "stds" instead of empirical values
                means = np.tile(
                        np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (num_classes, 1))
                stds = np.tile(
                        np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (num_classes, 1))
            else:
                # Compute values needed for means and stds
                # var(x) = E(x^2) - E(x)^2
                class_counts = np.zeros((num_classes, 1)) + cfg.EPS
                sums = np.zeros((num_classes, 4))
                squared_sums = np.zeros((num_classes, 4))
                for im_i in xrange(num_images):
                    targets = roidb[im_i]['bbox_targets']
                    for cls in xrange(1, num_classes):
                        cls_inds = np.where(targets[:, 0] == cls)[0]
                        if cls_inds.size > 0:
                            class_counts[cls] += cls_inds.size
                            sums[cls, :] += targets[cls_inds, 1:].sum(axis=0)
                            squared_sums[cls, :] += \
                                    (targets[cls_inds, 1:] ** 2).sum(axis=0)

                means = sums / class_counts
                stds = np.sqrt(squared_sums / class_counts - means ** 2)

            print 'bbox target means:'
            print means
            print means[1:, :].mean(axis=0) # ignore bg class
            print 'bbox target stdevs:'
            print stds
            print stds[1:, :].mean(axis=0) # ignore bg class

            # Normalize targets
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS:
                print "Normalizing targets"
                for im_i in xrange(num_images):
                    targets = roidb[im_i]['bbox_targets']
                    for cls in xrange(1, num_classes):
                        cls_inds = np.where(targets[:, 0] == cls)[0]
                        roidb[im_i]['bbox_targets'][cls_inds, 1:] -= means[cls, :]
                        roidb[im_i]['bbox_targets'][cls_inds, 1:] /= stds[cls, :]
            else:
                print "NOT normalizing targets"

            # These values will be needed for making predictions
            # (the predicts will need to be unnormalized and uncentered)
            return means.ravel(), stds.ravel()

        def _compute_targets(rois, overlaps, labels):
            """Compute bounding-box regression targets for an image."""
            # Indices of ground-truth ROIs
            gt_inds = np.where(overlaps == 1)[0]
            if len(gt_inds) == 0:
                # Bail if the image has no ground-truth ROIs
                return np.zeros((rois.shape[0], 5), dtype=np.float32)
            # Indices of examples for which we try to make predictions
            ex_inds = np.where(overlaps >= cfg.TRAIN.BBOX_THRESH)[0]

            # Get IoU overlap between each ex ROI and gt ROI
            ex_gt_overlaps = bbox_overlaps(
                np.ascontiguousarray(rois[ex_inds, :], dtype=np.float),
                np.ascontiguousarray(rois[gt_inds, :], dtype=np.float))

            # Find which gt ROI each ex ROI has max overlap with:
            # this will be the ex ROI's gt target
            gt_assignment = ex_gt_overlaps.argmax(axis=1)
            gt_rois = rois[gt_inds[gt_assignment], :]
            ex_rois = rois[ex_inds, :]

            targets = np.zeros((rois.shape[0], 5), dtype=np.float32)
            targets[ex_inds, 0] = labels[ex_inds]
            targets[ex_inds, 1:] = bbox_transform(ex_rois, gt_rois)
            return targets

        print 'Computing bounding-box regression targets...'

        if cfg.TRAIN.BBOX_REG:
            self.bbox_means, self.bbox_stds = add_bbox_regression_targets(roidb)
        else:
            self.bbox_means, self.bbox_stds = None, None

        print 'done'

        self._roidb = roidb

    def preprocess_test(self ):
        '''
        '''
        pass


    def init_sampler(self ):
        """Set the roidb to be used by this layer during training."""
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

    def get_bbox_means_stds(self):
        return self.bbox_means, self.bbox_stds

