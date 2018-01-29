import tensorflow as tf
import numpy as np
import keras
from keras import backend as K
import keras_contrib
from costar_google_brainrobotdata.grasp_loss import gaussian_kernel_2D
from costar_google_brainrobotdata.inception_preprocessing import preprocess_image
from tensorflow.python.platform import flags

flags.DEFINE_integer('image_size', 224,
                     """Provide square images of this size.""")
flags.DEFINE_integer('num_preprocess_threads', 12,
                     """Number of preprocessing threads per tower. """
                     """Please make this a multiple of 4.""")
flags.DEFINE_integer('num_readers', 12,
                     """Number of parallel readers during train.""")
flags.DEFINE_integer('input_queue_memory_factor', 12,
                     """Size of the queue of preprocessed images. """
                     """Default is ideal but try smaller values, e.g. """
                     """4, 2 or 1, if host memory is constrained. See """
                     """comments in code for more details.""")
flags.DEFINE_boolean(
    'redundant', True,
    """Duplicate images for every bounding box so dataset is easier to traverse.
       Please note that this does not substantially affect file size because
       protobuf is the underlying TFRecord data type and it
       has optimizations eliminating repeated identical data entries.
    """)
flags.DEFINE_integer('sigma_divisor', 10,
                     """Sigma divisor for grasp success 2d labels.""")

FLAGS = flags.FLAGS


def parse_example_proto(examples_serialized):
    feature_map = {
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                            default_value=''),
        'image/filename': tf.FixedLenFeature([], dtype=tf.string,
                                             default_value=''),
        'image/height': tf.FixedLenFeature([], dtype=tf.int32,
                                           default_value=''),
        'image/width': tf.FixedLenFeature([], dtype=tf.int32,
                                          default_value='')}
    for i in range(4):
        y_key = 'bbox/y' + str(i)
        x_key = 'bbox/x' + str(i)
        feature_map[y_key] = tf.VarLenFeature(dtype=tf.float32)
        feature_map[x_key] = tf.VarLenFeature(dtype=tf.float32)
    feature_map['bbox/cy'] = tf.VarLenFeature(dtype=tf.float32)
    feature_map['bbox/cx'] = tf.VarLenFeature(dtype=tf.float32)
    feature_map['bbox/tan'] = tf.VarLenFeature(dtype=tf.float32)
    feature_map['bbox/theta'] = tf.VarLenFeature(dtype=tf.float32)
    feature_map['bbox/sin_theta'] = tf.VarLenFeature(dtype=tf.float32)
    feature_map['bbox/cos_theta'] = tf.VarLenFeature(dtype=tf.float32)
    feature_map['bbox/width'] = tf.VarLenFeature(dtype=tf.float32)
    feature_map['bbox/height'] = tf.VarLenFeature(dtype=tf.float32)
    feature_map['bbox/grasp_success'] = tf.VarLenFeature(dtype=tf.int32)

    features = tf.parse_single_example(examples_serialized, feature_map)

    return features


def parse_example_proto_redundant(examples_serialized):
    feature_map = {
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                            default_value=''),
        'image/filename': tf.FixedLenFeature([], dtype=tf.string,
                                             default_value=''),
        'image/height': tf.FixedLenFeature([], dtype=tf.int32,
                                           default_value=''),
        'image/width': tf.FixedLenFeature([], dtype=tf.int32,
                                          default_value='')}
    for i in range(4):
        y_key = 'bbox/y' + str(i)
        x_key = 'bbox/x' + str(i)
        feature_map[y_key] = tf.FixedLenFeature([1], dtype=tf.float32)
        feature_map[x_key] = tf.FixedLenFeature([1], dtype=tf.float32)
    feature_map['bbox/cy'] = tf.FixedLenFeature([1], dtype=tf.float32)
    feature_map['bbox/cx'] = tf.FixedLenFeature([1], dtype=tf.float32)
    feature_map['bbox/tan'] = tf.FixedLenFeature([1], dtype=tf.float32)
    feature_map['bbox/theta'] = tf.FixedLenFeature([1], dtype=tf.float32)
    feature_map['bbox/sin_theta'] = tf.FixedLenFeature([1], dtype=tf.float32)
    feature_map['bbox/cos_theta'] = tf.FixedLenFeature([1], dtype=tf.float32)
    feature_map['bbox/width'] = tf.FixedLenFeature([1], dtype=tf.float32)
    feature_map['bbox/height'] = tf.FixedLenFeature([1], dtype=tf.float32)
    feature_map['bbox/grasp_success'] = tf.FixedLenFeature([1], dtype=tf.int32)

    features = tf.parse_single_example(examples_serialized, feature_map)

    return features


def eval_image(image, height, width):
    image = tf.image.central_crop(image, central_fraction=0.875)
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [height, width],
                                     align_corners=False)
    image = tf.squeeze(image, [0])

    return image


def distort_color(image, thread_id):
    color_ordering = thread_id % 2
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image


def distort_image(image, height, width, thread_id):
    # Need to update coordinates if flipping
    # distorted_image = tf.image.random_flip_left_right(image)
    distorted_image = distort_color(image, thread_id)
    return distorted_image


def image_preprocessing(image_buffer, train, thread_id=0):
    height = FLAGS.image_size
    width = FLAGS.image_size
    image = tf.image.decode_png(image_buffer, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # image = tf.image.resize_images(image, [height, width])
    if train:
        image = distort_image(image, height, width, thread_id)
    #else:
    #    image = eval_image(image, height, width)
    # image = tf.subtract(image, 0.5)
    # image = tf.multiply(image, 2.0)
    return image


def approximate_gaussian_ground_truth_image(image_shape, center, grasp_theta, grasp_width, grasp_height, label, sigma_divisor=None):
    """ Gaussian "ground truth" image approximation for a single proposed grasp at a time.

        For use with the Cornell grasping dataset

       see also: ground_truth_images() in build_cgd_dataset.py
    """
    if sigma_divisor is None:
        sigma_divisor = FLAGS.sigma_divisor
    grasp_dims = keras.backend.concatenate([grasp_width, grasp_height])
    sigma = keras.backend.max(grasp_dims) / sigma_divisor

    # make sure center value for gaussian is 0.5
    gaussian = gaussian_kernel_2D(image_shape[:2], center=center, sigma=sigma)
    # label 0 is grasp failure, label 1 is grasp success, label 0.5 will have "no effect".
    # gaussian center with label 0 should be subtracting 0.5
    # gaussian center with label 1 should be adding 0.5
    gaussian = ((label * 2) - 1.0) * gaussian
    max_num = K.max(K.max(gaussian), K.placeholder(1.0))
    min_num = K.min(K.min(gaussian), K.placeholder(-1.0))
    gaussian = (gaussian - min_num) / (max_num - min_num)
    return gaussian


def batch_inputs(data_files, train, num_epochs, batch_size,
                 num_preprocess_threads, num_readers):
    print(train)
    if train:
        filename_queue = tf.train.string_input_producer(data_files,
                                                        num_epochs,
                                                        shuffle=True,
                                                        capacity=16)
    else:
        filename_queue = tf.train.string_input_producer(data_files,
                                                        num_epochs,
                                                        shuffle=False,
                                                        capacity=1)

    examples_per_shard = 1024
    min_queue_examples = examples_per_shard * FLAGS.input_queue_memory_factor
    if train:
        print('pass')
        examples_queue = tf.RandomShuffleQueue(
            capacity=min_queue_examples+3*batch_size,
            min_after_dequeue=min_queue_examples,
            dtypes=[tf.string])
    else:
        examples_queue = tf.FIFOQueue(
            capacity=examples_per_shard + 3 * batch_size,
            dtypes=[tf.string])

    if num_readers > 1:
        enqueue_ops = []
        for _ in range(num_readers):
            reader = tf.TFRecordReader()
            _, value = reader.read(filename_queue)
            enqueue_ops.append(examples_queue.enqueue([value]))
        tf.train.queue_runner.add_queue_runner(
            tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
        examples_serialized = examples_queue.dequeue()
    else:
        reader = tf.TFRecordReader()
        _, examples_serialized = reader.read(filename_queue)

    features = []
    for thread_id in range(num_preprocess_threads):
        feature = parse_and_preprocess(examples_serialized, train, thread_id)

        features.append(feature)

    features = tf.train.batch_join(
        features,
        batch_size=batch_size,
        capacity=2*num_preprocess_threads*batch_size)

    # height = FLAGS.image_size
    # width = FLAGS.image_size
    # depth = 3

    # features['image/decoded'] = tf.reshape(features['image/decoded'], shape=[batch_size, height, width, depth])

    return features


def height_width_sin_cos_4(height=None, width=None, sin_theta=None, cos_theta=None, features=None):
    """ This is the input to pixelwise grasp prediction on the cornell dataset.
    """
    sin_cos_height_width = []
    if features is not None:
        sin_cos_height_width = [features['bbox/height'], features['bbox/width'],
                                features['bbox/sin_theta'], features['bbox/cos_theta']]
    else:
        con_cos_height_width = [sin_theta, cos_theta, height, width]
    return K.concatenate(sin_cos_height_width)


def grasp_success_yx_3(grasp_success=None, cy=None, cx=None, features=None):
    if features is not None:
        combined = [features['bbox/grasp_success'],
                    features['bbox/cy'], features['bbox/cx']]
    else:
        combined = [grasp_success, cy, cx]
    return K.concatenate(combined)


def parse_and_preprocess(examples_serialized, is_training, label_features_to_extract=None, data_features_to_extract=None):
    if FLAGS.redundant:
        feature = parse_example_proto_redundant(examples_serialized)
    else:
        feature = parse_example_proto(examples_serialized)
    image = preprocess_image(feature['image/encoded'], is_training=is_training)
    feature['image/decoded'] = image
    feature['grasp_success_2D'] = approximate_gaussian_ground_truth_image(
        image_shape=keras.backend.int_shape(image),
        center=[feature['bbox/cy'], feature['bbox/cx']],
        grasp_theta=feature['bbox/theta'],
        grasp_width=feature['bbox/width'],
        grasp_height=feature['bbox/height'],
        label=feature['grasp_success'])
    feature['sin_cos_height_width_4'] = height_width_sin_cos_4(features=feature)
    feature['grasp_success_yx_3'] = grasp_success_yx_3(features=feature)
    if data_features_to_extract is None:
        return feature
    else:
        # extract the features in a specific order if
        # features_to_extract is specified,
        # useful for yielding to a generator
        # like with keras.model.fit().
        return ([feature[feature_name] for feature_name in label_features_to_extract],
                [feature[feature_name] for feature_name in data_features_to_extract])


def yield_record(
        tfrecord_filenames, label_features_to_extract, data_features_to_extract,
        parse_example_proto_fn=parse_and_preprocess, batch_size=32,
        device='/cpu:0', is_training=True, steps=None):
    # based_on https://github.com/visipedia/tfrecords/blob/master/iterate_tfrecords.py
    with tf.device(device):
        dataset = tf.data.TFRecordDataset(tfrecord_filenames)

        # call the parse_example_proto_fn with the is_training flag set.
        def parse_fn_is_training(example):
            return parse_example_proto_fn(example, is_training=is_training,
                                          label_features_to_extract=label_features_to_extract,
                                          data_features_to_extract=data_features_to_extract)
        dataset = dataset.map(parse_example_proto_fn=parse_fn_is_training)  # Parse the record into tensors.
        dataset = dataset.repeat(count=steps)  # Repeat the input indefinitely.
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_initializable_iterator()
    #     # Construct a Reader to read examples from the .tfrecords file
    #     reader = tf.TFRecordReader()
    #     _, serialized_example = reader.read(filename_queue)

    #     features = decode_serialized_example(serialized_example, features_to_extract)

    # coord = tf.train.Coordinator()
    with tf.Session() as sess:

        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        # tf.train.start_queue_runners(sess=sess, coord=coord)

        try:

            # while not coord.should_stop():
            while True:
                next_element = iterator.get_next()
                outputs = sess.run(next_element)

                yield outputs

        except tf.errors.OutOfRangeError as e:
            pass


def distorted_inputs(data_files, num_epochs, train=True, batch_size=None):
    with tf.device('/cpu:0'):
        print(train)
        features = batch_inputs(
            data_files, train, num_epochs, batch_size,
            num_preprocess_threads=FLAGS.num_preprocess_threads,
            num_readers=FLAGS.num_readers)

    return features


def inputs(data_files, num_epochs=1, train=False, batch_size=1):
    with tf.device('/cpu:0'):
        print(train)
        features = batch_inputs(
            data_files, train, num_epochs, batch_size,
            num_preprocess_threads=FLAGS.num_preprocess_threads,
            num_readers=1)

    return features
