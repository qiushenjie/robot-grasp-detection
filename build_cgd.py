'''Converts Cornell Grasping Dataset data into TFRecords data format using Example protos.
The raw data set resides in png and txt files located in the following structure:
    
    dataset/03/pcd0302r.png
    dataset/03/pcd0302cpos.txt
'''

import tensorflow as tf
import os
import glob
import numpy as np

dataset = 'D:\JiangShan\cornell_grasping_dataset'

def _process_image(filename,sess):
    # Decode the image
    image_raw = tf.read_file(filename)
    image_tensor = tf.image.decode_png(image_raw, channels=3)
    image_buffer = sess.run(image_tensor)
    assert len(image_buffer.shape) == 3
    height = image_buffer.shape[0]
    width = image_buffer.shape[1]
    assert image_buffer.shape[2] == 3    
    return image_buffer, height, width

def _process_bboxes(name):
    '''Create a list with the coordinates of the grasping rectangles. Every 
    element is either x or y of a vertex.'''
    with open(name, 'r') as f:
        bboxes = list(map(
              lambda coordinate: float(coordinate), f.read().strip().split()))
    return bboxes

def _int64_feature(v):
    if not isinstance(v, list):
        v = [v]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=v))

def _floats_feature(v):
    if not isinstance(v, list):
        v = [v]
    return tf.train.Feature(float_list=tf.train.FloatList(value=v))

def _bytes_feature(v):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))

def _convert_to_example(filename, bboxes, image_buffer, height, width):
    # Build an Example proto for an example
    example = tf.train.Example(features=tf.train.Features(feature={
          'image/filename': _bytes_feature(filename.encode()),
          'image/encoded': _bytes_feature(image_buffer.tobytes()),
          'image/height': _int64_feature(height),
          'image/width': _int64_feature(width),
          'bboxes': _floats_feature(bboxes)}))
    return example

def main():
    train_file = os.path.join(dataset, 'train-cgd')
    validation_file = os.path.join(dataset, 'validation-cgd')
    print(train_file)
    print(validation_file)
    writer_train = tf.python_io.TFRecordWriter(train_file)
    writer_validation = tf.python_io.TFRecordWriter(validation_file)
   
    # Creating a list with all the image paths
    folders = range(1,11)
    folders = ['0'+str(i) if i<10 else '10' for i in folders]
    filenames = []
    for i in folders:
        for name in glob.glob(os.path.join(dataset, i, 'pcd'+i+'*r.png')):
            filenames.append(name)
    
    # Shuffle the list of image paths
    np.random.shuffle(filenames)

    count = 0
    valid_img = 0
    train_img = 0

    sess = tf.Session()

    for filename in filenames:
        bbox = filename[:-5]+'cpos.txt'
        bboxes = _process_bboxes(bbox)
        image_buffer, height, width = _process_image(filename, sess)
        example = _convert_to_example(filename, bboxes, image_buffer, height, width)
        # Split the dataset in 80% for training and 20% for validation
        if count % 5 == 0:
            writer_validation.write(example.SerializeToString())
            valid_img +=1
        else:
            writer_train.write(example.SerializeToString())
            train_img +=1
        count +=1

    print('Done converting %d images in TFRecords with %d train images and %d validation images' % (count, train_img, valid_img))
    
    writer_train.close()
    writer_validation.close()

if __name__ == '__main__':
    main()