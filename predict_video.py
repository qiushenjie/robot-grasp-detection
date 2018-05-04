import cv2
import contextlib
import mmap
import tensorflow as tf
from grasp_inf import inference
from grasp_det import grasp_to_bbox
import time

count=0
filename = 'Z:/0000.jpg'
prev_filename = ""

def draw_bbox(img, bbox):
    p1 = (int(float(bbox[0][0]) / 0.35), int(float(bbox[0][1]) / 0.47))
    p2 = (int(float(bbox[1][0]) / 0.35), int(float(bbox[1][1]) / 0.47))
    p3 = (int(float(bbox[2][0]) / 0.35), int(float(bbox[2][1]) / 0.47))
    p4 = (int(float(bbox[3][0]) / 0.35), int(float(bbox[3][1]) / 0.47))

    cv2.line(img, p1, p2, (0, 0, 255))
    cv2.line(img, p2, p3, (0, 0, 255))
    cv2.line(img, p3, p4, (0, 0, 255))
    cv2.line(img, p4, p1, (0, 0, 255))

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess = tf.Session()
sess.run(init_op)
dg = {}
img_raw_data = tf.placeholder(tf.string)
img_data = tf.image.decode_jpeg(img_raw_data)
img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
img_reshape = tf.image.resize_images(img_data, [224, 224])
img_reshape = tf.reshape(img_reshape, shape=[1, 224, 224, 3])
x_hat, y_hat, tan_hat, w_hat, h_hat = tf.unstack(inference(img_reshape), axis=1)
bbox_hat = grasp_to_bbox(x_hat, y_hat, tan_hat, h_hat, w_hat)

while True:
    with contextlib.closing(mmap.mmap(-1, 1024, tagname='grasp_det', access=mmap.ACCESS_READ)) as m:
        filename = m.read(1024).decode().replace('\x00', '')
        if filename != prev_filename:
            img_show = cv2.imread(filename)
            with open(filename, 'rb') as f:
                img = f.read()
            print(filename+" received")

            if dg == {}:
                lg = ['w1', 'b1', 'w2', 'b2', 'w3', 'b3', 'w4', 'b4', 'w5', 'b5', 'w_fc1', 'b_fc1', 'w_fc2', 'b_fc2',
                      'w_output', 'b_output']
                for i in lg:
                    dg[i] = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name == i + ':0'][0]

                saver_g = tf.train.Saver(dg)
                saver_g.restore(sess, './models/grasp/m4/m4.ckpt')

            bbox_model = sess.run(bbox_hat, feed_dict={img_raw_data: img})
            print(bbox_model)
            draw_bbox(img_show, bbox_model)
            cv2.imshow('bbox', img_show)
            cv2.waitKey(1)
            #cv2.waitKey(1)
        #count += 1
        prev_filename = filename
        #print(s)