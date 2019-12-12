import tensorflow.keras.backend as K
import tensorflow as tf

from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.layers import Layer, Input, Conv2D, Activation, add, BatchNormalization, UpSampling2D, ZeroPadding2D, Conv2DTranspose, Flatten, MaxPooling2D, AveragePooling2D, InputSpec, LeakyReLU, Dense

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import mean
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.utils import plot_model
#from tensorflow.keras.engine.topology import Network

import random
import datetime
import time
import json
import math
import csv
import sys
from os.path import join, exists
from os import mkdir, makedirs
import numpy as np

# sys.path.append('../')
#import load_data

from factories.optimizer_factory import Optimizer_Factory
from layers.reflection_padding_2d import ReflectionPadding2D
from metrics.loss_functions import lse, cycle_loss

def save_img(img, fname):
    if len(tf.shape(img)) > 3:
        img = img[tf.constant(0),:,:,:]
    result_img = tf.image.encode_png(tf.image.convert_image_dtype(img, dtype=tf.uint8))
    tf.io.write_file(tf.constant(fname), result_img)

class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            if len(image.shape) == 3:
                image = image[np.newaxis, :, :, :]

            if self.num_imgs < self.pool_size:  # fill up the image pool
                self.num_imgs = self.num_imgs + 1
                if len(self.images) == 0:
                    self.images = image
                else:
                    self.images = np.vstack((self.images, image))

                if len(return_images) == 0:
                    return_images = image
                else:
                    return_images = np.vstack((return_images, image))

            else:  # 50% chance that we replace an old synthetic image
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id, :, :, :]
                    tmp = tmp[np.newaxis, :, :, :]
                    self.images[random_id, :, :, :] = image[0, :, :, :]
                    if len(return_images) == 0:
                        return_images = tmp
                    else:
                        return_images = np.vstack((return_images, tmp))
                else:
                    if len(return_images) == 0:
                        return_images = image
                    else:
                        return_images = np.vstack((return_images, image))

        return return_images


class CycleGAN():
    """docstring for CycleGAN."""

    def __init__(self, cf):
        self.model = None
        self.cf = cf

        # from Keras documentation: Total number of steps (batches of samples) to yield from generator before declaring one epoch finished and starting the next epoch. It should typically be equal to the number of unique samples of your dataset divided by the batch size.
        if(cf.n_images_train is not None):
            self.steps_per_epoch = int(np.ceil(cf.n_images_train / float(cf.batch_size_train)))
        else:
            samplesTrainA = self.get_file_len(cf.trainA_file_path_full)
            samplesTrainB = self.get_file_len(cf.trainB_file_path_full)
            samplesTrain = min(samplesTrainA, samplesTrainB)
            samplesTestA = self.get_file_len(cf.testA_file_path_full)
            samplesTestB = self.get_file_len(cf.testB_file_path_full)
            samplesValid = min(samplesTestA, samplesTestB)
            # self.steps_per_epoch = int(np.ceil(get_file_len(cf.train_file_path_full) / float(cf.batch_size_train)))
            # self.validation_steps = int(np.ceil(get_file_len(cf.valid_file_path_full) / float(cf.batch_size_valid)))
            self.steps_per_epoch = int(np.ceil(samplesTrain / float(cf.batch_size_train)))
            self.validation_steps = int(np.ceil(samplesValid / float(cf.batch_size_valid)))


    def get_file_len(self, file_path_full):
        return len(["" for line in open(file_path_full, "r")])

    def make_discriminators(self):
        # ======= Discriminator model ==========
        if self.cf.use_multiscale_discriminator:
            D_A = self.modelMultiScaleDiscriminator()
            D_B = self.modelMultiScaleDiscriminator()
            loss_weights_D = [0.5, 0.5] # 0.5 since we train on real and synthetic images
        else:
            D_A = self.modelDiscriminator()
            D_B = self.modelDiscriminator()
            loss_weights_D = [0.5]  # 0.5 since we train on real and synthetic images

        # Discriminator builds
        image_A = Input(shape=self.cf.input_shape)
        image_B = Input(shape=self.cf.input_shape)
        guess_A = D_A(image_A)
        guess_B = D_B(image_B)
        self.D_A = Model(inputs=image_A, outputs=guess_A, name='D_A_model')
        self.D_B = Model(inputs=image_B, outputs=guess_B, name='D_B_model')

        # self.D_A.summary()
        # self.D_B.summary()
        optimizer_D = Optimizer_Factory().make(self.cf.optimizer, self.cf.learning_rate_D)
        self.D_A.compile(optimizer=optimizer_D,
                         loss=lse,
                         loss_weights=loss_weights_D)
        self.D_B.compile(optimizer=optimizer_D,
                         loss=lse,
                         loss_weights=loss_weights_D)

        # Use Networks to avoid falsy keras error about weight descripancies
        self.D_A_static = Model(inputs=image_A, outputs=guess_A, name='D_A_static_model')
        self.D_B_static = Model(inputs=image_B, outputs=guess_B, name='D_B_static_model')

        # Do note update discriminator weights during generator training
        self.D_A_static.trainable = False
        self.D_B_static.trainable = False

    def make_generators(self):
        # Generators
        self.G_A2B = self.modelGenerator(name='G_A2B_model')
        self.G_B2A = self.modelGenerator(name='G_B2A_model')
        # self.G_A2B.summary()

        optimizer_G = Optimizer_Factory().make(self.cf.optimizer, self.cf.learning_rate_G)

        if self.cf.use_identity_learning:
            self.G_A2B.compile(optimizer=optimizer_G, loss='MAE')
            self.G_B2A.compile(optimizer=optimizer_G, loss='MAE')

        # Generator builds
        real_A = Input(shape=self.cf.input_shape, name='real_A')
        real_B = Input(shape=self.cf.input_shape, name='real_B')
        synthetic_B = self.G_A2B(real_A)
        synthetic_A = self.G_B2A(real_B)
        dA_guess_synthetic = self.D_A_static(synthetic_A)
        dB_guess_synthetic = self.D_B_static(synthetic_B)
        reconstructed_A = self.G_B2A(synthetic_B)
        reconstructed_B = self.G_A2B(synthetic_A)

        model_outputs = [reconstructed_A, reconstructed_B]
        compile_losses = [cycle_loss, cycle_loss,
                          lse, lse]
        compile_weights = [self.cf.lambda_1, self.cf.lambda_2,
                           self.cf.lambda_D, self.cf.lambda_D]

        if self.cf.use_multiscale_discriminator:
            for _ in range(2):
                compile_losses.append(lse)
                compile_weights.append(self.cf.lambda_D)  # * 1e-3)  # Lower weight to regularize the model
            for i in range(2):
                model_outputs.append(dA_guess_synthetic[i])
                model_outputs.append(dB_guess_synthetic[i])
        else:
            model_outputs.append(dA_guess_synthetic)
            model_outputs.append(dB_guess_synthetic)

        if self.cf.use_supervised_learning:
            model_outputs.append(synthetic_A)
            model_outputs.append(synthetic_B)
            compile_losses.append('MAE')
            compile_losses.append('MAE')
            compile_weights.append(self.cf.supervised_weight)
            compile_weights.append(self.cf.supervised_weight)

        self.model = Model(inputs=[real_A, real_B],
                             outputs=model_outputs,
                             name='G_model')

        self.model.compile(optimizer=optimizer_G,
                             loss=compile_losses,
                             loss_weights=compile_weights)
        # self.G_A2B.summary()

    def make(self):

        # Load pretrained weights
        if not self.cf.load_pretrained:
            print('   turning off loading pretrained weights not implemented...')

        # Build model
        self.make_discriminators()

        # make generators
        self.make_generators()

        # if cf.resume_training:
        #     model.load_weights(cf.checkpoint_path)
        #
        # Show model structure
        if self.cf.show_model:
            model.summary()
            plot_model(model, to_file=join(self.cf.savepath, 'model.png'))

        # Output the model
        print ('   Model: ' + self.cf.model_name)


    # Learning rate #
    def get_lr_linear_decay_rate(self, max_nr_images):
        # Calculate decay rates
        updates_per_epoch_D = 2 * max_nr_images + self.cf.discriminator_iterations - 1
        updates_per_epoch_G = max_nr_images + self.cf.generator_iterations - 1
        if self.cf.use_identity_learning:
            updates_per_epoch_G *= (1 + 1 / self.cf.identity_mapping_modulus)
        denominator_D = (self.cf.n_epochs - self.cf.decay_epoch) * updates_per_epoch_D
        denominator_G = (self.cf.n_epochs - self.cf.decay_epoch) * updates_per_epoch_G
        decay_D = self.cf.learning_rate_D / denominator_D
        decay_G = self.cf.learning_rate_G / denominator_G

        return decay_D, decay_G

    def update_lr(self, model, decay):
        new_lr = K.get_value(model.optimizer.lr) - decay
        if new_lr < 0:
            new_lr = 0
        # print(K.get_value(model.optimizer.lr))
        K.set_value(model.optimizer.lr, new_lr)


    # Images #
    def save_tmp_images(self, real_image_A, real_image_B, synthetic_image_A, synthetic_image_B):
        try:
            reconstructed_image_A = self.G_B2A.predict(synthetic_image_B)
            reconstructed_image_B = self.G_A2B.predict(synthetic_image_A)

            real_images = np.vstack((real_image_A[0], real_image_B[0]))
            synthetic_images = np.vstack((synthetic_image_B[0], synthetic_image_A[0]))
            reconstructed_images = np.vstack((reconstructed_image_A[0], reconstructed_image_B[0]))

            # save path
            result_path = join(self.cf.savepath, 'tmp_images')
            if exists(result_path) == False:
                mkdir(result_path)

            # write images
            save_img(real_image_A[0,:,:,:], join(result_path, 'realA.png'))
            save_img(real_image_B[0,:,:,:], join(result_path, 'realB.png'))
            save_img(synthetic_image_A[0,:,:,:], join(result_path, 'syntheticA.png'))
            save_img(synthetic_image_B[0,:,:,:], join(result_path, 'syntheticB.png'))
            save_img(reconstructed_image_A[0,:,:,:], join(result_path, 'reconstructedA.png'))
            save_img(reconstructed_image_B[0,:,:,:], join(result_path, 'reconstructedB.png'))

        except: # Ignore if file is open
            pass

    def saveImages(self, epoch, real_image_A, real_image_B, num_saved_images=1):
        result_path = join(self.cf.savepath, 'images')
        # if not exists(join(result_path, 'A')):
        #     makedirs(join(result_path, 'A'))
        #     makedirs(join(result_path, 'B'))
        #     makedirs(join(result_path, 'Atest'))
        #     makedirs(join(result_path, 'Btest'))

        # testString = ''

        for i in range(num_saved_images):
            # if i == num_saved_images:
            #     real_image_A = self.A_test[0]
            #     real_image_B = self.B_test[0]
            #     real_image_A = np.expand_dims(real_image_A, axis=0)
            #     real_image_B = np.expand_dims(real_image_B, axis=0)
            #     testString = 'test'
            #
            # else:
            #     #real_image_A = self.A_train[rand_A_idx[i]]
            #     #real_image_B = self.B_train[rand_B_idx[i]]
            #     if len(real_image_A.shape) < 4:
            #         real_image_A = np.expand_dims(real_image_A, axis=0)
            #         real_image_B = np.expand_dims(real_image_B, axis=0)

            synthetic_image_B = self.G_A2B.predict(real_image_A)
            synthetic_image_A = self.G_B2A.predict(real_image_B)
            reconstructed_image_A = self.G_B2A.predict(synthetic_image_B)
            reconstructed_image_B = self.G_A2B.predict(synthetic_image_A)

            # write images
            print("SHAPE")
            print(tf.shape(real_image_A))
            save_img(real_image_A[i,:,:,:], join(result_path, str(i) + '_realA.png'))
            save_img(real_image_B[i,:,:,:], join(result_path, str(i) + '_realB.png'))
            save_img(synthetic_image_A[i,:,:,:], join(result_path, str(i) + '_syntheticA.png'))
            save_img(synthetic_image_B[i,:,:,:], join(result_path, str(i) + '_syntheticB.png'))
            save_img(reconstructed_image_A[i,:,:,:], join(result_path, str(i) + '_reconstructedA.png'))
            save_img(reconstructed_image_B[i,:,:,:], join(result_path, str(i) + '_reconstructedB.png'))

    # Training #
    def print_ETA(self, start_time, epoch, loop_index):
        passed_time = time.time() - start_time

        iterations_so_far = ((epoch - 1) * self.steps_per_epoch + loop_index) / self.cf.batch_size_train
        iterations_total = self.cf.n_epochs * self.steps_per_epoch / self.cf.batch_size_train
        iterations_left = iterations_total - iterations_so_far
        eta = round(passed_time / (iterations_so_far + 1e-5) * iterations_left)

        passed_time_string = str(datetime.timedelta(seconds=round(passed_time)))
        eta_string = str(datetime.timedelta(seconds=eta))
        print('Time passed', passed_time_string, ': ETA in', eta_string)

    def _run_training_iteration(self, loop_index, epoch, real_images_A, real_images_B, ones, zeros, synthetic_pool_A, synthetic_pool_B):
        # ======= Discriminator training ==========
            # Generate batch of synthetic images
        synthetic_images_B = self.G_A2B.predict(real_images_A)
        synthetic_images_A = self.G_B2A.predict(real_images_B)
        synthetic_images_A = synthetic_pool_A.query(synthetic_images_A)
        synthetic_images_B = synthetic_pool_B.query(synthetic_images_B)

        for _ in range(self.cf.discriminator_iterations):
            DA_loss_real = self.D_A.train_on_batch(x=real_images_A, y=ones)
            DB_loss_real = self.D_B.train_on_batch(x=real_images_B, y=ones)
            DA_loss_synthetic = self.D_A.train_on_batch(x=synthetic_images_A, y=zeros)
            DB_loss_synthetic = self.D_B.train_on_batch(x=synthetic_images_B, y=zeros)
            if self.cf.use_multiscale_discriminator:
                DA_loss = sum(DA_loss_real) + sum(DA_loss_synthetic)
                DB_loss = sum(DB_loss_real) + sum(DB_loss_synthetic)
                print('DA_losses: ', np.add(DA_loss_real, DA_loss_synthetic))
                print('DB_losses: ', np.add(DB_loss_real, DB_loss_synthetic))
            else:
                DA_loss = DA_loss_real + DA_loss_synthetic
                DB_loss = DB_loss_real + DB_loss_synthetic
            D_loss = DA_loss + DB_loss

            if self.cf.discriminator_iterations > 1:
                print('D_loss:', D_loss)
                sys.stdout.flush()

        # ======= Generator training ==========
        target_data = [real_images_A, real_images_B]  # Compare reconstructed images to real images
        if self.cf.use_multiscale_discriminator:
            for i in range(2):
                target_data.append(ones[i])
                target_data.append(ones[i])
        else:
            target_data.append(ones)
            target_data.append(ones)

        if self.cf.use_supervised_learning:
            target_data.append(real_images_A)
            target_data.append(real_images_B)

        for _ in range(self.cf.generator_iterations):
            G_loss = self.model.train_on_batch(x=[real_images_A, real_images_B], y=target_data)
            if self.cf.generator_iterations > 1:
                print('G_loss:', G_loss)
                sys.stdout.flush()

        gA_d_loss_synthetic = G_loss[1]
        gB_d_loss_synthetic = G_loss[2]
        reconstruction_loss_A = G_loss[3]
        reconstruction_loss_B = G_loss[4]

        # Identity training
        if self.cf.use_identity_learning and loop_index % self.cf.identity_mapping_modulus == 0:
            G_A2B_identity_loss = self.G_A2B.train_on_batch(
                x=real_images_B, y=real_images_B)
            G_B2A_identity_loss = self.G_B2A.train_on_batch(
                x=real_images_A, y=real_images_A)
            print('G_A2B_identity_loss:', G_A2B_identity_loss)
            print('G_B2A_identity_loss:', G_B2A_identity_loss)

        # Update learning rates
        if self.cf.use_linear_decay and epoch > self.cf.decay_epoch:
            self.update_lr(self.D_A, self.decay_D)
            self.update_lr(self.D_B, self.decay_D)
            self.update_lr(self.model, self.decay_G)

        # Store training data
        self.training_history['DA_losses'].append(DA_loss)
        self.training_history['DB_losses'].append(DB_loss)
        self.training_history['gA_d_losses_synthetic'].append(gA_d_loss_synthetic)
        self.training_history['gB_d_losses_synthetic'].append(gB_d_loss_synthetic)
        self.training_history['gA_losses_reconstructed'].append(reconstruction_loss_A)
        self.training_history['gB_losses_reconstructed'].append(reconstruction_loss_B)

        GA_loss = gA_d_loss_synthetic + reconstruction_loss_A
        GB_loss = gB_d_loss_synthetic + reconstruction_loss_B
        reconstruction_loss = reconstruction_loss_A + reconstruction_loss_B

        self.training_history['D_losses'].append(D_loss)
        self.training_history['G_losses'].append(G_loss)
        self.training_history['reconstruction_losses'].append(reconstruction_loss)

        #GA_losses.append(GA_loss)
        #GB_losses.append(GB_loss)

        print('\n')
        print('Epoch----------------', epoch, '/', self.cf.n_epochs)
        print('Loop index----------------', loop_index + 1, '/', self.steps_per_epoch)
        print('D_loss: ', D_loss)
        print('G_loss: ', G_loss[0])
        print('reconstruction_loss: ', reconstruction_loss)
        print('DA_loss:', DA_loss)
        print('DB_loss:', DB_loss)

        if loop_index % 20 == 0:
            # Save temporary images continously
            #self.save_tmp_images(real_images_A, real_images_B, synthetic_images_A, synthetic_images_B)
            self.print_ETA(self.start_time, epoch, loop_index)

    def train(self, train_gen, cb):
        if (not self.cf.train_model):
            return None

        print('\n > Training the model...')
        # ======================================================================
        # Begin training
        # ======================================================================
        training_history = {}

        DA_losses = []
        DB_losses = []
        gA_d_losses_synthetic = []
        gB_d_losses_synthetic = []
        gA_losses_reconstructed = []
        gB_losses_reconstructed = []

        GA_losses = []
        GB_losses = []
        reconstruction_losses = []
        D_losses = []
        G_losses = []

        self.training_history = {
            'DA_losses': DA_losses,
            'DB_losses': DB_losses,
            'gA_d_losses_synthetic': gA_d_losses_synthetic,
            'gB_d_losses_synthetic': gB_d_losses_synthetic,
            'gA_losses_reconstructed': gA_losses_reconstructed,
            'gB_losses_reconstructed': gB_losses_reconstructed,
            'D_losses': D_losses,
            'G_losses': G_losses,
            'reconstruction_losses': reconstruction_losses}

        # Image pools used to update the discriminators
        synthetic_pool_A = ImagePool(self.cf.synthetic_pool_size)
        synthetic_pool_B = ImagePool(self.cf.synthetic_pool_size)

        # self.saveImages('(init)')

        # Tweaks
        self.REAL_LABEL = 1.0  # Use e.g. 0.9 to avoid training the discriminators to zero loss

        # labels
        if self.cf.use_multiscale_discriminator:
            label_shape1 = (self.cf.batch_size_train,) + self.D_A.output_shape[0][1:]
            label_shape2 = (self.cf.batch_size_train,) + self.D_A.output_shape[1][1:]
            #label_shape4 = (self.cf.batch_size_train,) + self.D_A.output_shape[2][1:]
            ones1 = np.ones(shape=label_shape1) * self.REAL_LABEL
            ones2 = np.ones(shape=label_shape2) * self.REAL_LABEL
            #ones4 = np.ones(shape=label_shape4) * self.REAL_LABEL
            ones = [ones1, ones2]  # , ones4]
            zeros1 = ones1 * 0
            zeros2 = ones2 * 0
            #zeros4 = ones4 * 0
            zeros = [zeros1, zeros2]  # , zeros4]
        else:
            label_shape = (self.cf.batch_size_train,) + self.D_A.output_shape[1:]
            ones = np.ones(shape=label_shape) * self.REAL_LABEL
            zeros = ones * 0

        # Linear decay
        if self.cf.use_linear_decay:
            self.decay_D, self.decay_G = self.get_lr_linear_decay_rate(self.steps_per_epoch * self.cf.n_epochs)

        # Start stopwatch for ETAs
        self.start_time = time.time()

        for epoch in range(1, self.cf.n_epochs + 1):
            loop_index = 1
            for images in train_gen:
                real_images_A = images[0]
                real_images_B = images[1]

                # Convert grey image into RGB
                if real_images_A.shape[3] == 1:
                    real_images_A = np.repeat(real_images_A, 3, axis=3)
                if real_images_B.shape[3] == 1:
                    real_images_B = np.repeat(real_images_B, 3, axis=3)

                if len(real_images_A.shape) == 3:
                    real_images_A = real_images_A[:, :, :, np.newaxis]
                    real_images_B = real_images_B[:, :, :, np.newaxis]

                # Run all training steps
                self._run_training_iteration(loop_index, epoch, real_images_A, real_images_B, ones, zeros, synthetic_pool_A, synthetic_pool_B)

                # Store models
                if loop_index % 20000 == 0:
                    self.save_model(self.D_A, loop_index)
                    self.save_model(self.D_B, loop_index)
                    self.save_model(self.G_A2B, loop_index)
                    self.save_model(self.G_B2A, loop_index)

                # Break if loop has ended
                if loop_index >= self.steps_per_epoch:
                    break

                loop_index += 1



            #================== within epoch loop end ==========================

            if epoch % self.cf.save_interval == 0:
                print('\n', '\n', '-------------------------Saving images for epoch', epoch, '-------------------------', '\n', '\n')
                self.saveImages(epoch, real_images_A, real_images_B, tf.shape(real_images_A)[0])

            if epoch % 20 == 0:
                # self.save_model(self.model)
                self.save_model(self.D_A, epoch)
                self.save_model(self.D_B, epoch)
                self.save_model(self.G_A2B, epoch)
                self.save_model(self.G_B2A, epoch)

            self.writeLossDataToFile(self.training_history)

            # Flush out prints each loop iteration
            sys.stdout.flush()

        self.save_model(self.D_A, epoch)
        self.save_model(self.D_B, epoch)
        self.save_model(self.G_A2B, epoch)
        self.save_model(self.G_B2A, epoch)
        print('   Training finished.')


    def predict(self, test_gen, tag='pred'):
        pass


    def test(self, test_gen):
        if not self.cf.test_model:
            return

        print('\n > Testing the model...')

        # Load best trained model
        self.model.load_weights(self.cf.weights_file)

        # get correct number of test samples depending on debugging or not
        if self.cf.n_images_test is not None:
            nb_test_samples = self.cf.n_images_test
        else:
            nb_test_samples = get_file_len(self.cf.test_file_path_full)

        # Evaluate model
        start_time = time.time()
        y_predictions = self.model.predict(test_gen.make_one_shot_iterator(), steps=nb_test_samples)
        total_time = time.time() - start_time
        fps = float(nb_test_samples) / total_time
        s_p_f = total_time / float(nb_test_samples)
        print ('   Testing time: {}. FPS: {}. Seconds per Frame: {}'.format(total_time, fps, s_p_f))

        # store predicted labels
        result_path = join(self.cf.savepath, 'predicted_labels')
        if exists(result_path) == False:
            mkdir(result_path)

        results = []
        fp = open(self.cf.test_file_path_full)
        image_names = fp.readlines()
        fp.close()
        for (idx, img_num) in enumerate(image_names):
            if idx > nb_test_samples-1:
                continue
            y_sample_prediction = y_predictions[idx,:,:,:,:]

            # print('sample: ' + img_num + ', idx: ' + str(idx))
            # print('min: ' + str(np.min(y_sample_prediction)))
            # print('max: ' + str(np.max(y_sample_prediction)))

            # compress to top probabilty

            result = np.argmax(y_sample_prediction, axis=-1)
            z=np.shape(result)[-1]
            for i in range(z):
                img_num = img_num.strip('\n')
                save_img(result[:,:,i:i+1], join(result_path, img_num + '_slice'+str(i)+'.png'))


#===============================================================================
# Architecture functions

    def ck(self, x, k, use_normalization):
        x = Conv2D(filters=k, kernel_size=4, strides=2, padding='same')(x)
        # Normalization is not done on the first discriminator layer
        if use_normalization:
            x = InstanceNormalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = LeakyReLU(alpha=0.2)(x)
        return x

    def c7Ak(self, x, k):
        x = Conv2D(filters=k, kernel_size=7, strides=1, padding='valid')(x)
        x = InstanceNormalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x

    def dk(self, x, k):
        x = Conv2D(filters=k, kernel_size=3, strides=2, padding='same')(x)
        x = InstanceNormalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x

    def Rk(self, x0):
        k = int(x0.shape[-1])
        # first layer
        x = ReflectionPadding2D((1,1))(x0)
        x = Conv2D(filters=k, kernel_size=3, strides=1, padding='valid')(x)
        x = InstanceNormalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        # second layer
        x = ReflectionPadding2D((1, 1))(x)
        x = Conv2D(filters=k, kernel_size=3, strides=1, padding='valid')(x)
        x = InstanceNormalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        # merge
        x = add([x, x0])
        return x

    def uk(self, x, k):
        # (up sampling followed by 1x1 convolution <=> fractional-strided 1/2)
        if self.cf.use_resize_convolution:
            x = UpSampling2D(size=(2, 2))(x)  # Nearest neighbor upsampling
            x = ReflectionPadding2D((1, 1))(x)
            x = Conv2D(filters=k, kernel_size=3, strides=1, padding='valid')(x)
        else:
            x = Conv2DTranspose(filters=k, kernel_size=3, strides=2, padding='same')(x)  # this matches fractinoally stided with stride 1/2
        x = InstanceNormalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x

#===============================================================================
# Models

    def modelMultiScaleDiscriminator(self, name=None):
        x1 = Input(shape=self.cf.input_shape)
        x2 = AveragePooling2D(pool_size=(2, 2))(x1)
        #x4 = AveragePooling2D(pool_size=(2, 2))(x2)

        out_x1 = self.modelDiscriminator('D1')(x1)
        out_x2 = self.modelDiscriminator('D2')(x2)
        #out_x4 = self.modelDiscriminator('D4')(x4)

        return Model(inputs=x1, outputs=[out_x1, out_x2], name=name)

    def modelDiscriminator(self, name=None):
        # Specify input
        input_img = Input(shape=self.cf.input_shape)
        # Layer 1 (#Instance normalization is not used for this layer)
        x = self.ck(input_img, 64, False)
        # Layer 2
        x = self.ck(x, 128, True)
        # Layer 3
        x = self.ck(x, 256, True)
        # Layer 4
        x = self.ck(x, 512, True)
        # Output layer
        if self.cf.use_patchgan:
            x = Conv2D(filters=1, kernel_size=4, strides=1, padding='same')(x)
        else:
            x = Flatten()(x)
            x = Dense(1)(x)
        #x = Activation('sigmoid')(x) - No sigmoid to avoid near-fp32 machine epsilon discriminator cost
        return Model(inputs=input_img, outputs=x, name=name)

    def modelGenerator(self, name=None):
        # Specify input
        input_img = Input(shape=self.cf.input_shape)
        # Layer 1
        x = ReflectionPadding2D((3, 3))(input_img)
        x = self.c7Ak(x, 32)
        # Layer 2
        x = self.dk(x, 64)
        # Layer 3
        x = self.dk(x, 128)

        if self.cf.use_multiscale_discriminator:
            # Layer 3.5
            x = self.dk(x, 256)

        # Layer 4-12: Residual layer
        for _ in range(4, 13):
            x = self.Rk(x)

        if self.cf.use_multiscale_discriminator:
            # Layer 12.5
            x = self.uk(x, 128)

        # Layer 13
        x = self.uk(x, 64)
        # Layer 14
        x = self.uk(x, 32)
        x = ReflectionPadding2D((3, 3))(x)
        x = Conv2D(self.cf.channel_size, kernel_size=7, strides=1)(x)
        x = Activation('tanh')(x)  # They say they use Relu but really they do not
        return Model(inputs=input_img, outputs=x, name=name)

#===============================================================================
# Loading / Saving

    def save_model(self, model, epoch):
        # Create folder to save model architecture and weights
        directory = join(self.cf.savepath, 'saved_models')
        if not exists(directory):
            makedirs(directory)

        # model_path_w = 'saved_models/{}_weights_epoch_{}.hdf5'.format(model.name, epoch)
        model_path_w = join(directory, '{}_weights_epoch_{}.hdf5'.format(model.name, epoch))
        model.save_weights(model_path_w)
        #model_path_m = 'saved_models/{}_model_epoch_{}.json'.format(model.name, epoch)
        model_path_m = join(directory, '{}_model_epoch_{}.json'.format(model.name, epoch))
        model.save_weights(model_path_m)
        json_string = model.to_json()
        with open(model_path_m, 'w') as outfile:
            json.dump(json_string, outfile)
        print('{} has been saved in saved_models'.format(model.name))

    def writeLossDataToFile(self, history):
        keys = sorted(history.keys())
        out_file = join(self.cf.savepath, 'loss_output.csv')
        with open(out_file, 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(keys)
            writer.writerows(zip(*[history[key] for key in keys]))
