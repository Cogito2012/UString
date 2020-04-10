
import _init_paths
from utils.visualization import draw_bounding_boxes
from datasets.factory import get_imdb
from model.test import _get_blobs
import os, cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
try:
  import cPickle as pickle
except ImportError:
  import pickle

if __name__ == '__main__':

    output_dir = '../output/res101/voc_2007_val/default/res101_faster_rcnn_iter_200000/'
    det_file = os.path.join(output_dir, 'detections.pkl')

    with open(det_file, 'rb') as f:
        all_boxes = pickle.load(f)

    imdb = get_imdb('voc_2007_val')
    num_images = len(imdb.image_index)
    for i in range(num_images):
        im = cv2.imread(imdb.image_path_at(i))

        # get the detected boxes for each image
        boxes_vis = []
        for j in range(1, imdb.num_classes):
            boxes_scores = all_boxes[j][i]
            keep = np.where(boxes_scores[:, -1] >= 0.6)[0]
            if len(keep) > 0:
                boxes_cls = np.hstack([boxes_scores[keep, :4], j * np.ones((len(keep), 1), dtype=np.int)])
            else:
                continue
            boxes_vis.append(boxes_cls)
        boxes_vis = np.vstack(boxes_vis)

        blobs, im_scales = _get_blobs(im)
        im_blob = blobs['data']
        im_info = [im_blob.shape[1], im_blob.shape[2], 1.0]
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        image_predbox = draw_bounding_boxes(im, boxes_vis, im_info)

        gt_boxes = imdb.roidb[i]['boxes']
        gt_clses = imdb.roidb[i]['gt_classes']
        gt_boxes = np.concatenate([gt_boxes, gt_clses[:, np.newaxis]], axis=1)
        image_gtbox = draw_bounding_boxes(im, gt_boxes, im_info)
        # image_vis = np.concatenate((image_gtbox, image_predbox), axis=0)

        plt.figure(figsize=(20, 6))
        plt.imshow(image_gtbox)
        plt.title('GT Boxes')

        plt.figure(figsize=(20, 6))
        plt.imshow(image_predbox)
        plt.title('Det Boxes')
        plt.show()

    # all_sizes = np.vstack(all_sizes)
    # print('-- mean:')
    # print(np.mean(all_sizes, axis=0))
    # print('-- max:')
    # print(np.max(all_sizes, axis=0))
    # print('-- min:')
    # print(np.min(all_sizes, axis=0))
