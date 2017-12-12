import tensorflow.contrib.slim as slim
import tensorflow as tf
import numpy as np
import cPickle as pickle
import os
import random
import time
import gzip
import requests
import cv2
import matplotlib.pyplot as plt

batch_size = 784

'''
   Kullback Leibler divergence
   https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
   https://github.com/fastforwardlabs/vae-tf/blob/master/vae.py#L178
'''
def kullbackleibler(mu, log_sigma):
   return -0.5*tf.reduce_sum(1+2*log_sigma-mu**2-tf.exp(2*log_sigma),1)


'''
   Leaky RELU
'''
def lrelu(x, leak=0.2, name="lrelu"):
   return tf.maximum(x, leak*x)


def encoder(encode_inputs):
   z_mean = slim.fully_connected(encode_inputs, 32, normalizer_fn=slim.batch_norm, activation_fn = tf.nn.relu, scope='z_mean')
   z_log_sigma = slim.fully_connected(encode_inputs, 32, normalizer_fn = slim.batch_norm, activation_fn=tf.nn.relu, scope='z_log_sigma')
	
   return z_mean, z_log_sigma 
   

   '''
      z is distributed as a multivariate normal with mean z_mean and diagonal
      covariance values sigma^2
      z ~ N(z_mean, np.exp(z_log_sigma)**2)
   '''

   return z_mean, z_log_sigma

def decoder(decode_inputs):
   decoded = slim.fully_connected(decode_inputs, 1, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='decoded') 
   decoded = tf.nn.sigmoid(decoded)
   
	
   return decoded


def train(mnist_train, mnist_test):
   with tf.Graph().as_default():
      global_step = tf.Variable(0, trainable=False, name='global_step')

      # placeholder for mnist images
      input_placeholder = tf.placeholder(tf.float32, [batch_size, 10, 1])
      test_placeholder = tf.placeholder(tf.float32, [batch_size, 10, 1])

      # encode images to 8 dim vector
      z_mean, z_log_sigma = encoder(input_placeholder)

      # reparameterization trick
      epsilon = tf.random_normal(tf.shape(z_log_sigma), name='epsilon')
      z = z_mean + epsilon * tf.exp(z_log_sigma) # N(mu, sigma**2)

      decoded = decoder(z)
      
      #reconstructed_loss = -tf.reduce_sum(images*tf.log(1e-10 + decoded) + (1-images)*tf.log(1e-10+1-decoded), 1)
      reconstructed_loss = tf.nn.l2_loss(input_placeholder-decoded)
      latent_loss = kullbackleibler(z_mean, z_log_sigma)

      cost = tf.reduce_mean(reconstructed_loss+latent_loss)

      train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)

      # saver for the model
      saver = tf.train.Saver(tf.all_variables())

      init = tf.initialize_all_variables()
      sess = tf.Session()
      sess.run(init)

      try: os.makedirs('checkpoint/images/')
      except: pass

      ckpt = tf.train.get_checkpoint_state('checkpoint/')
      if ckpt and ckpt.model_checkpoint_path:
         try:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print 'Model restored'
         except:
            print 'Could not restore model'
            pass

      step = 0
      while True:
         step += 1

         # get random images from the training set
         batch_images = random.sample(mnist_train, batch_size)

         # send through the network
         _, loss_ = sess.run([train_op, cost], feed_dict={input_placeholder: batch_images})
         loss_ = sess.run([cost], feed_dict={input_placeholder:batch_images})[0]
         print 'Step: ' + str(step) + ' Loss: ' + str(loss_)

         if step%500 == 0:
            print
            print 'Saving model'
            print
            saver.save(sess, "checkpoint/checkpoint", global_step=global_step)

            # get random images from the test set
            batch_images = random.sample(mnist_test, batch_size)

            # encode them using the encoder, then decode them
            encode_decode = sess.run(decoded, feed_dict={test_placeholder: batch_images})

            # write out a few
            c = 0
            for real, dec in zip(batch_images, encode_decode):
               print 'Original: ' + str(real)
               print 'Decoded: ' + str(dec)
               if c == 5:
                  break
               c+=1

def main(argv=None):
   randBinList = lambda n: [random.randint(0,1) for b in range(1,n+1)]
   train_image_list = []
   for x in range(60000):
     input_sample = randBinList(10)
     input_sample = np.reshape(input_sample, (10, 1))
     train_image_list.append(input_sample)
   np.asarray(train_image_list)
		
   test_image_list = []

   for y in range(10000):
     test_sample = randBinList(10)
     test_sample = np.reshape(test_sample, (10,1))
     test_image_list.append(test_sample)

   np.asarray(test_image_list)
   train(train_image_list, test_image_list)

   train(train_image_list, test_image_list)

if __name__ == '__main__':
   tf.app.run()
