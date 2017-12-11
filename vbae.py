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
import matplotlib.pyplot as plt

from utils import *

batch_size = 784 

def encode(encode_inputs):
	W_conv1 = weight_variable([3, 3, 1, 64])
        b_conv1 = bias_variable([64])
        conv1 = conv2d(encode_inputs, W_conv1, 2)
        #bn_conv1 = tf.layers.batch_normalization(conv1)
        h_conv1 = tf.nn.relu(conv1 + b_conv1)
	
	W_conv2 = weight_variable([2, 2, 64, 32])
	b_conv2 = bias_variable([32])
	conv2 = conv2d(h_conv1, W_conv2, 2)
	#bn_conv2 = tf.layers.batch_normalization(conv2)
	h_conv2 = tf.nn.relu(conv2 + b_conv2)
	
	W_conv3 = weight_variable([3, 3, 32, 16])
	b_conv3 = bias_variable([16])
	conv3 = conv2d(h_conv2, W_conv3, 2)
	#bn_conv3 = tf.layers.batch_normalization(conv3)
	h_conv3 = tf.nn.relu(conv3 + b_conv3)

	h_conv3_flat = tf.reshape(h_conv3, [-1, 4 * 4 * 16])

	W_fc1 = weight_variable([4 * 4 * 16, 8])
	b_fc1 = bias_variable([8])
	fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

	
	W_fc2 = weight_variable([8, 8])
	b_fc2 = bias_variable([8])
	encoded = tf.nn.relu(tf.matmul(fc1, W_fc2) + b_fc2)
		

	return encoded

def decode(decode_inputs):	
	W_fc1 = weight_variable([8, 8])
	b_fc1 = bias_variable([8])
	h_fc1 = tf.nn.relu(tf.matmul(decode_inputs, W_fc1) + b_fc1) 

	W_fc2 = weight_variable([8, 16])
        b_fc2 = bias_variable([16])
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
	h_fc2_unflat = tf.reshape(h_fc2, (batch_size, 4, 4, 1))

	upsample1 = tf.image.resize_images(h_fc2_unflat, size=(7,7), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	conv4 = tf.layers.conv2d(inputs=upsample1, filters=16, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
	upsample2 = tf.image.resize_images(conv4, size=(14,14), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	conv5 = tf.layers.conv2d(inputs=upsample2, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
	upsample3 = tf.image.resize_images(conv5, size=(28,28), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	conv6 = tf.layers.conv2d(inputs=upsample3, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)


	logits = tf.layers.conv2d(inputs=conv6, filters=1, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
	
	decoded = tf.nn.sigmoid(logits)	
	

	return decoded, logits
	

def model(input_images):
	encoded = encode(input_images)
	decoded, logits = decode(encoded)
	# Pass logits through sigmoid and calculate the cross-entropy loss
	loss = tf.nn.l2_loss(input_images - decoded)
	opt = tf.train.AdamOptimizer(0.001).minimize(loss)

	return decoded, loss, opt, logits

def train(input_images):
	with tf.Graph().as_default():
		global_step = tf.train.get_or_create_global_step()	

		images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 28, 28, 1))
		decoded, loss, opt, logits = model(images_placeholder)

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
			_, loss_ = sess.run([opt, loss], feed_dict={images_placeholder: input_images[:batch_size]})
			print 'Step: ' + str(step) + 'Loss: ' + str(loss_)
			c = 0
			if step%1000 == 0:
				print('*****Saving model*****')
				saver.save(sess, "checkpoint/checkpoint", global_step=global_step)
				shuffle(input_images)
				decoded = sess.run(logits, feed_dict={images_placeholder: input_images[:batch_size]})
				# write evaluation code here
				#compare input and output... display
				for real, dec in zip(input_images[:batch_size], decoded):
					dec, real = np.squeeze(dec), np.squeeze(real)
					plt.imsave('images/'+str(step)+'_'+str(c)+'real.png', real)
					plt.imsave('images/'+str(step)+'_'+str(c)+'dec.png', dec)
					if c == 5:
						break
					c+=1
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
