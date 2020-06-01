import os
import numpy as np

def process(data_path, dest_path, phase):
    files_list = []
    batch_id = 1
    for filename in sorted(os.listdir(os.path.join(data_path, phase))):
        filepath = os.path.join(data_path, phase, filename)
        all_data = np.load(filepath)
        features = all_data['data']  # 10 x 100 x 20 x 4096
        labels = all_data['labels']  # 10 x 2
        detections = all_data['det']  # 10 x 100 x 19 x 6
        videos = all_data['ID']  # 10
        nid = 1
        for i, vid in enumerate(videos):
            vidname = 'b' + str(batch_id).zfill(3) + '_' + vid.decode('UTF-8')
            if vidname in files_list:
                vidname = vidname + '_' + str(nid).zfill(2)
                nid += 1
            feat_file = os.path.join(dest_path, vidname + '.npz')
            if os.path.exists(feat_file):
                continue
            np.savez_compressed(feat_file, data=features[i], labels=labels[i], det=detections[i], ID=vidname)
            print('batch: %03d, %s file: %s' % (batch_id, phase, vidname))
            files_list.append(vidname)
        batch_id += 1
    return files_list

def split_dad(data_path, dest_path):
    # prepare the result paths
    train_path = os.path.join(dest_path, 'training')
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    test_path = os.path.join(dest_path, 'testing')
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    # process training set
    train_list = process(data_path, train_path, 'training')
    print('Training samples: %d'%(len(train_list)))
    # process testing set
    test_list = process(data_path, test_path, 'testing')
    print('Testing samples: %d' % (len(test_list)))

if __name__ == '__main__':
    DAD_PATH = '/data/DAD/features'
    DEST_PATH = '/data/DAD/features_split'
    split_dad(DAD_PATH, DEST_PATH)
