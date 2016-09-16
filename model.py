# Original Version: Taehoon Kim (http://carpedm20.github.io)
#   + Source: https://github.com/carpedm20/DCGAN-tensorflow/blob/e30539fb5e20d5a0fed40935853da97e9e55eee8/model.py
#   + License: MIT
# [2016-08-05] Modifications for Completion: Brandon Amos (http://bamos.github.io)
#   + License: MIT

from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
from six.moves import xrange

from ops import *
from utils import *

class DCGAN(object):
    def __init__(self, sess, image_size=64, is_crop=False,
                 batch_size=64, sample_size=64,
                 z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3,
                 checkpoint_dir=None, lam=0.1):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen untis for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. [3]
        """
        self.sess = sess
        self.is_crop = is_crop
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.image_shape = [image_size, image_size, 3]

        ##self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.lam = lam

        self.c_dim = 3

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')
        self.g_bn4 = batch_norm(name='g_bn4')
        self.g_bn5 = batch_norm(name='g_bn5')

        self.checkpoint_dir = checkpoint_dir
        self.build_model()

        self.model_name = "DCGAN.model"

    def build_model(self):
        self.images = tf.placeholder(
            tf.float32, [None] + self.image_shape, name='real_images')
        self.sample_images= tf.placeholder(
            tf.float32, [None] + self.image_shape, name='sample_images')
        #updated
        self.z= tf.placeholder(
            tf.float32, [None] + self.image_shape, name='mask_images')



        #updated
        self.G = self.generator(self.z)
        self.D, self.D_logits = self.discriminator(self.images)

        self.sampler = self.sampler(self.z)
        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

        self.d_sum = tf.histogram_summary("d", self.D)
        self.d__sum = tf.histogram_summary("d_", self.D_)
        self.G_sum = tf.image_summary("G", self.G)

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits,
                                                    tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_,
                                                    tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_,
                                                    tf.ones_like(self.D_)))

        self.d_loss_real_sum = tf.scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.scalar_summary("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = tf.scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver(max_to_keep=1)

        # Completion.
        self.mask = tf.placeholder(tf.float32, [None] + self.image_shape, name='mask')

    def train(self, config):
        data = glob(os.path.join(config.dataset, "*.png"))
        assert(len(data) > 0)

        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1, epsilon = 0.1) \
                          .minimize(self.d_loss, var_list=self.d_vars)

        g_optim_init = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1, epsilon = 0.1) \
                          .minimize(self.g_loss, var_list=self.g_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate/5, beta1=config.beta1, epsilon = 0.1) \
                          .minimize(self.g_loss, var_list=self.g_vars)
        tf.initialize_all_variables().run()

        self.g_sum = tf.merge_summary(
            [self.d__sum, self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.merge_summary(
            [self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.train.SummaryWriter("./logs", self.sess.graph)

        ##sample_z = np.random.uniform(-1, 1, size=(self.sample_size , self.z_dim))
        sample_files = data[0:self.sample_size]
        sample = [get_image(sample_file, self.image_size, is_crop=self.is_crop) for sample_file in sample_files]
        sample_images = np.array(sample).astype(np.float32)

        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in xrange(config.epoch):
            data = glob(os.path.join(config.dataset, "*.png"))
            batch_idxs = min(len(data), config.train_size) // self.batch_size

            for idx in xrange(0, batch_idxs):
                # changed batching to randomize and avoid training on images of same celeb
                #batch_files = data[idx*config.batch_size:(idx+1)*config.batch_size]

                idx_batch = np.random.randint(0,len(data),config.batch_size).tolist()
                batch_files = []
                for Bidx in idx_batch:
                    batch_files.append(data[Bidx])
                batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop)
                         for batch_file in batch_files]

                batch_images = np.array(batch).astype(np.float32)

                l = np.random.randint(int(self.image_size/6),int(self.image_size/2))
                l_end = self.image_size
                mask = np.ones(self.image_shape)
                noise_mat = np.random.normal(size=self.image_shape)
                #randomly choose which edge to complete
                rand_mask = np.random.randint(1,5)
                if rand_mask == 1:
                    mask[:, 0:l, :] = np.multiply(mask[:, 0:l, :], noise_mat[:, 0:l, :])
                elif rand_mask == 2:
                    mask[0:l, :, :] = np.multiply(mask[0:l, :, :], noise_mat[0:l, :, :])
                elif rand_mask == 3:
                    mask[:, l_end-l:l_end, :] = np.multiply(mask[:, l_end-l:l_end, :], noise_mat[:, l_end-l:l_end, :])
                elif rand_mask == 4:
                    mask[l_end-l:l_end, :, :] = np.multiply(mask[l_end-l:l_end, :, :], noise_mat[l_end-l:l_end, :, :])

                batch_mask = np.resize(mask, [self.batch_size] + self.image_shape)
                masked_images = np.multiply(batch_images, batch_mask)
                batch_z = masked_images

                if idx>1 and (errD_fake+errD_real)<0.6:
                    # Update D network only sometimes
                    if np.random.randint(0,100)>90:
                        _, summary_str = self.sess.run([d_optim, self.d_sum],
                            feed_dict={ self.images: batch_images, self.z: batch_z })
                        self.writer.add_summary(summary_str, counter)
                        print('D learn')

                else:
                    # Update D network
                    _, summary_str = self.sess.run([d_optim, self.d_sum],
                        feed_dict={ self.images: batch_images, self.z: batch_z })
                    self.writer.add_summary(summary_str, counter)
                    #print('D learn')

                # turn off G to let D learn
                if idx < 2 or errG > (errD_fake+errD_real):
                    # Update G network
                    if epoch < 5:
                        _, summary_str = self.sess.run([g_optim_init, self.g_sum],
                        feed_dict={ self.z: batch_z })
                        self.writer.add_summary(summary_str, counter)
                        #print('G learn fast')
                    else:
                        _, summary_str = self.sess.run([g_optim, self.g_sum],
                                feed_dict={ self.z: batch_z })
                        self.writer.add_summary(summary_str, counter)
                        #print('G learn slow')

                errD_fake = self.d_loss_fake.eval({self.z: batch_z})
                errD_real = self.d_loss_real.eval({self.images: batch_images})
                errG = self.g_loss.eval({self.z: batch_z})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                        time.time() - start_time, errD_fake+errD_real, errG))

                if np.mod(counter, 10) == 1:
                    samples, d_loss, g_loss = self.sess.run(
                        [self.sampler, self.d_loss, self.g_loss],
                        feed_dict={self.z: batch_z, self.images: batch_images}
                    )
                    save_images(samples, [8, 8],
                                './samples/train_{:02d}_{:04d}.png'.format(epoch, idx))
                    print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

                if np.mod(counter, 10) == 1:
                    self.save(config.checkpoint_dir, counter)



    def discriminator(self, image, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
        h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
        #h1 = lrelu(conv2d(h0, self.df_dim*2, name='d_h1_conv'))
        #h2 = lrelu(conv2d(h1, self.df_dim*4, name='d_h2_conv'))
        #h3 = lrelu(conv2d(h2, self.df_dim*8, name='d_h3_conv'))
        h4 = linear(tf.reshape(h3, [-1, 8192]), 1, 'd_h3_lin')

        return tf.nn.sigmoid(h4), h4


    def discriminator_Sauer(self, image, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        h0 = lrelu(conv2d(image, self.df_dim, k_h = 3, k_w = 3, d_h = 1, d_w = 1, name='d_h0_conv'))
        h1 = lrelu((conv2d(h0, self.df_dim, k_h = 3, k_w = 3, d_h = 2, d_w = 2,name='d_h1_conv')))
        h2 = lrelu((conv2d(h1, self.df_dim, k_h = 3, k_w = 3, d_h = 1, d_w = 1, name='d_h2_conv')))
        h3 = lrelu((conv2d(h2, self.df_dim, k_h = 3, k_w = 3, d_h = 2, d_w = 2, name='d_h3_conv')))
        h4 = lrelu((conv2d(h3, self.df_dim, k_h = 1, k_w = 1, d_h = 1, d_w = 1, name='d_h4_conv')))
        h5 = linear(tf.reshape(h3, [-1, 8192]), 1, 'd_h5_lin')

        return tf.nn.sigmoid(h5), h5

    def generator(self, z_image):

        h0 = lrelu(conv2d(z_image, self.gf_dim, d_h=1, d_w=1,name='g_h0'))
        h1 = lrelu(self.g_bn1(conv2d(h0, self.gf_dim, d_h=1, d_w=1,name='g_h1')))
        h2 = lrelu(self.g_bn2(conv2d(h1, self.gf_dim, d_h=1, d_w=1,name='g_h2')))
        h3 = lrelu(self.g_bn3(conv2d(h2, self.gf_dim, d_h=1, d_w=1,name='g_h3')))
        h4 = lrelu(self.g_bn4(conv2d(h3, self.gf_dim, d_h=1, d_w=1, name='g_h4')))
        h5 = lrelu(self.g_bn5(conv2d(h4, 3, k_h=3, k_w=3, d_h=1, d_w=1, name='g_h5')))
        '''
        h1 = lrelu((conv2d(h0, self.gf_dim, d_h=1, d_w=1,name='g_h1')))
        h2 = lrelu((conv2d(h1, self.gf_dim, d_h=1, d_w=1,name='g_h2')))
        h3 = lrelu((conv2d(h2, self.gf_dim, d_h=1, d_w=1,name='g_h3')))
        h4 = lrelu((conv2d(h3, self.gf_dim, d_h=1, d_w=1, name='g_h4')))
        h5 = lrelu((conv2d(h4, 3, k_h=3, k_w=3, d_h=1, d_w=1, name='g_h5')))

        '''
        return tf.nn.tanh(h5)


    def sampler(self, z_image):
        tf.get_variable_scope().reuse_variables()

        h0 = lrelu(conv2d(z_image, self.gf_dim, d_h=1, d_w=1,name='g_h0'))

        h1 = lrelu(self.g_bn1(conv2d(h0, self.gf_dim, d_h=1, d_w=1,name='g_h1')))
        h2 = lrelu(self.g_bn2(conv2d(h1, self.gf_dim, d_h=1, d_w=1,name='g_h2')))
        h3 = lrelu(self.g_bn3(conv2d(h2, self.gf_dim, d_h=1, d_w=1,name='g_h3')))
        h4 = lrelu(self.g_bn4(conv2d(h3, self.gf_dim, d_h=1, d_w=1, name='g_h4')))
        h5 = lrelu(self.g_bn5(conv2d(h4, 3, k_h=3, k_w=3, d_h=1, d_w=1, name='g_h5')))
        '''
        h1 = lrelu((conv2d(h0, self.gf_dim, d_h=1, d_w=1,name='g_h1')))
        h2 = lrelu((conv2d(h1, self.gf_dim, d_h=1, d_w=1,name='g_h2')))
        h3 = lrelu((conv2d(h2, self.gf_dim, d_h=1, d_w=1,name='g_h3')))
        h4 = lrelu((conv2d(h3, self.gf_dim, d_h=1, d_w=1, name='g_h4')))
        h5 = lrelu((conv2d(h4, 3, k_h=3, k_w=3, d_h=1, d_w=1, name='g_h5')))
        '''
        return tf.nn.tanh(h5)


    def save(self, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, self.model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            return True
        else:
            return False
