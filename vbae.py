import tensorflow.contrib.slim as slim
import tensorflow as tf
import numpy as np
import cPickle as pickle
import os
import random
from random import shuffle
import time
import gzip
import requests
import cv2

from utils import *

batch_size = 784

def encode(encode_inputs):
        W_conv1 = weight_variable([3, 3, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(encode_inputs, W_conv1, 2) + b_conv1)

        W_fc1 = weight_variable([14 * 14 * 32, 512])
        b_fc1 = bias_variable([512])

        h_conv1_flat = tf.reshape(h_conv1, [-1, 14*14*32])

        h_fc1 = tf.nn.relu(tf.matmul(h_conv1_flat, W_fc1) + b_fc1)

        W_fc2 = weight_variable([512, 256])
        b_fc2 = bias_variable([256])

        encoded = tf.matmul(h_fc1, W_fc2) + b_fc2

        return encoded

def decode(decode_inputs):
        W_fc1 = weight_variable([256, 512])
        b_fc1 = bias_variable([512])

        h_fc1 = tf.nn.relu(tf.matmul(decode_inputs, W_fc1) + b_fc1)
        W_fc2 = weight_variable([512, 784])
        b_fc2 = bias_variable([784])

        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
        h_fc2_unflat = tf.reshape(h_fc2, (batch_size, 4, 4, 49))

        t_conv3 = tf.nn.conv2d_transpose(h_fc2_unflat, [1, 2, 2, 1], (batch_size, 3, 3, 512), [1, 1, 1, 1], padding='SAME')
        print('t_conv3...', t_conv3)
        exit()

        return decoded


def model(input_images):
        encoded = encode(input_images)
        decoded = decode(encoded)
        return encoded

def train(input_images):
        with tf.Graph().as_default():
                global_step = tf.train.get_or_create_global_step()

                images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 28, 28, 1))
                model_output = model(images_placeholder)

                # saver for the model
                saver = tf.train.Saver(tf.all_variables())

                init = tf.initialize_all_variables()
                sess = tf.Session()
                sess.run(init)

                try:
                        os.mkdir('images/')
                except:
                        pass
                try:
                        os.mkdir('checkpoint/')
                except:
                        pass

                ckpt = tf.train.get_checkpoint_state('checkpoint/')
                if ckpt and ckpt.model_checkpoint_path:
                        try:
                                saver.restore(sess, ckpt.model_checkpoint_path)
                                print 'Model restored'
                        except:
                                print 'Could not restore model'
                                pass

                step = int(sess.run(global_step))
                while True:
                        step += 1

                        shuffle(input_images)

                        if step%100 == 0:
                                print('*****Saving model*****')
                                saver.save(sess, "checkpoint/checkpoint", global_step=global_step)

                                # write evaluation code here
                                #compare input and output... display

def main(argv=None):
        f = gzip.open('mnist.pkl.gz', 'rb')
        train_set, val_set, test_set = pickle.load(f)

        train_image_list = []

        for img, label in zip(train_set[0], train_set[1]):
                img = np.reshape(img, (28,28,1))
                #cv2.imshow('image', img)
                #cv2.waitKey(0)

                train_image_list.append(img)

        np.asarray(train_image_list)

        train(train_image_list)

if __name__ == '__main__':
        tf.app.run()
