import math
import numpy as np
import tensorflow as tf

import dm_arch
import dm_utils

FLAGS = tf.app.flags.FLAGS

def _residual_block(model, num_units, mapsize, nlayers=2):
    """Adds a residual block similar to Arxiv 1512.03385, Figure 3.
    """

    # TBD: Try pyramidal block as per arXiv 1610.02915.
    # Note Figure 6d (the extra BN compared to 6b seems to help as per Table 2)
    # Also note Figure 5b.

    assert len(model.get_output().get_shape()) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels)"

    # Add *linear* projection in series if needed prior to shortcut
    if num_units != int(model.get_output().get_shape()[3]):
        model.add_conv2d(num_units, mapsize=1, stride=1)

    if nlayers > 0:
        # Batch norm not needed for every conv layer
        # and it slows down training substantially
        model.add_batch_norm()

        for _ in range(nlayers):
            # Bypassing on every conv layer, as implied by Arxiv 1612.07771
            # Experimental results particularly favor one (Arxiv 1512.03385) or the other (this)
            bypass = model.get_output()
            model.add_relu()
            model.add_conv2d(num_units, mapsize=mapsize, is_residual=True)
            model.add_sum(bypass)

    return model


def _generator_model(sess, features):
    # See Arxiv 1603.05027
    model = dm_arch.Model('GENE', 2 * features - 1)

    mapsize = 3

    # Encoder
    layers  = [24, 48]
    for nunits in layers:
        _residual_block(model, nunits, mapsize)
        model.add_avg_pool()

    # Decoder
    layers  = [96, 64]
    for nunits in layers:
        _residual_block(model, nunits, mapsize)
        _residual_block(model, nunits, mapsize)
        model.add_upscale()

    nunits = 48
    _residual_block(model, nunits, mapsize)
    _residual_block(model, nunits, mapsize)
    model.add_conv2d(3, mapsize=1)
    model.add_sigmoid(1.1)
    
    return model


def _discriminator_model(sess, image):
    model = dm_arch.Model('DISC', 2 * image - 1.0)

    mapsize = 3
    layers  = [64, 96, 128, 192] #[32, 48, 96, 128]

    for nunits in layers:
        model.add_batch_norm()
        model.add_lrelu()
        model.add_conv2d(nunits, mapsize=mapsize)
            
        model.add_avg_pool()

    nunits = layers[-1]
    model.add_batch_norm()
    model.add_lrelu()
    model.add_conv2d(nunits, mapsize=mapsize)

    #model.add_batch_norm()
    model.add_lrelu()
    model.add_conv2d(1, mapsize=mapsize)
    
    model.add_mean()

    return model


def _generator_loss(features, gene_output, disc_fake_output, annealing):
    # I.e. did we fool the discriminator?
    gene_adversarial_loss = tf.reduce_mean(-disc_fake_output, name='gene_adversarial_loss')
        

    return gene_adversarial_loss # gene_loss


def _discriminator_loss(disc_real_output, disc_fake_output):
    # I.e. did we correctly identify the input as real or not?
    disc_real_loss = -disc_real_output
    disc_fake_loss =  disc_fake_output
    print disc_real_loss
    print disc_fake_loss

    disc_real_loss = tf.reduce_mean(disc_real_loss, name='disc_real_loss')
    disc_fake_loss = tf.reduce_mean(disc_fake_loss, name='disc_fake_loss')
    disc_loss      = tf.add(disc_real_loss, disc_fake_loss, name='disc_loss')

    return disc_loss, disc_real_loss, disc_fake_loss


def _clip_weights(var_list, weights_threshold):
    """Clips all the given weights to fall within the range [-weight_threshold, weight_threshold]"""
    ops = []
    for var in var_list:
        clipped = tf.clip_by_value(var, -weights_threshold, weights_threshold)
        op      = tf.assign(var, clipped)
        ops.append(op)

    return tf.group(*ops, name='clip_weights')


def create_model(sess, source_images, target_images=None, annealing=None, verbose=False, gan_type="wgan"):  #default wgan unless specified  
    rows  = int(source_images.get_shape()[1])
    cols  = int(source_images.get_shape()[2])
    depth = int(source_images.get_shape()[3])

    #
    # Generator
    #
    gene          = _generator_model(sess, source_images)
    gene_out      = gene.get_output()
    gene_var_list = gene.get_all_variables()

    if verbose:
        print("Generator input (feature) size is %d x %d x %d = %d" %
              (rows, cols, depth, rows*cols*depth))

        print("Generator has %4.2fM parameters" % (gene.get_num_parameters()/1e6,))
        print()

    if target_images is not None:
        learning_rate = tf.maximum(FLAGS.learning_rate_start * annealing, FLAGS.learning_rate_end, name='learning_rate')

        # Instance noise used to aid convergence.
        # See http://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/
        noise_shape = [FLAGS.batch_size, rows, cols, depth]
        noise = tf.truncated_normal(noise_shape, mean=0.0, stddev=FLAGS.instance_noise*annealing, name='instance_noise')
        noise = tf.reshape(noise, noise_shape) # TBD: Why is this even necessary? I don't get it.
        noise = 0.0

        #
        # Discriminator: one takes real inputs, another takes fake (generated) inputs
        #
        disc_real     = _discriminator_model(sess, target_images + noise)
        disc_real_out = disc_real.get_output()
        disc_var_list = disc_real.get_all_variables()

        if gan_type=="wgan":
            disc_fake     = _discriminator_model(sess, gene_out + noise)
            disc_fake_out = disc_fake.get_output()
        
            if verbose:
                print("Discriminator input (feature) size is %d x %d x %d = %d" %
                      (rows, cols, depth, rows*cols*depth))

                print("Discriminator has %4.2fM parameters" % (disc_real.get_num_parameters()/1e6,))
                print()

            #
            # Losses and optimizers
            #
            gene_loss = _generator_loss(source_images, gene_out, disc_fake_out, annealing)
            
            disc_loss, disc_real_loss, disc_fake_loss = _discriminator_loss(disc_real_out, disc_fake_out)

            gene_opti = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                               name='gene_optimizer')

            # Note WGAN doesn't work well with Adam or any other optimizer that relies on momentum
            disc_opti = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=0.0,
                                                  name='disc_optimizer')

            gene_minimize = gene_opti.minimize(gene_loss, var_list=gene_var_list, name='gene_loss_minimize')    
            disc_minimize = disc_opti.minimize(disc_loss, var_list=disc_var_list, name='disc_loss_minimize')

            # Weight clipping a la WGAN (arXiv 1701.07875)
            # TBD: We shouldn't be clipping all variables (incl biases), just the weights
            disc_clip_weights = _clip_weights(disc_var_list, FLAGS.disc_weights_threshold)
            disc_minimize     = tf.group(disc_minimize, disc_clip_weights)

        if gan_type=="imp_wgan":
            # fake_data = Generator(BATCH_SIZE)
            # real_data_gen_raw = _generator_model(sess, target_images)
            # real_data_gen = real_data_gen_raw.get_output()
            real_data_gen = target_images
            fake_data_gen = gene_out

            disc_fake     = _discriminator_model(sess, gene_out + noise)
            disc_fake_out = disc_fake.get_output()

            gene_loss = -tf.reduce_mean(disc_fake_out)
            disc_loss = tf.reduce_mean(disc_fake_out) - tf.reduce_mean(disc_real_out)
            _, disc_real_loss, disc_fake_loss = _discriminator_loss(disc_real_out, disc_fake_out)


            alpha = tf.random_uniform(
                shape=[FLAGS.batch_size, 100, 80, 3], 
                minval=0.,
                maxval=1.
            )
            differences = fake_data_gen - real_data_gen
            interpolates = real_data_gen + (alpha*differences)

            gradients = tf.gradients(_discriminator_model(sess, interpolates).get_output(), [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes-1.)**2)
            disc_loss += FLAGS.lambd*gradient_penalty

            gen_train_op = tf.train.AdamOptimizer(
                learning_rate=learning_rate, 
                beta1=0.5,
                beta2=0.9
            )
            disc_train_op = tf.train.AdamOptimizer(
                learning_rate=learning_rate, 
                beta1=0.5, 
                beta2=0.9
            )

            gene_minimize = gen_train_op.minimize(gene_loss, var_list=gene_var_list, name='gene_loss_minimize')    
            disc_minimize = disc_train_op.minimize(disc_loss, var_list=disc_var_list, name='disc_loss_minimize')

            disc_clip_weights = _clip_weights(disc_var_list, FLAGS.disc_weights_threshold)
            disc_minimize     = tf.group(disc_minimize, disc_clip_weights)

        if gan_type=="imp_dcgan": #mixed with wgan for now just to test one sided label smoothing
            disc_fake     = _discriminator_model(sess, gene_out + noise)
            disc_fake_out = disc_fake.get_output()

            #one sided label smoothing
            tf.add(disc_real_out, tf.random_uniform(disc_real_out.shape,-0.3,0.3))
            #tf.add(disc_fake_out, tf.random_uniform(disc_fake_out.shape,0,0.3))
        
            if verbose:
                print("Discriminator input (feature) size is %d x %d x %d = %d" %
                      (rows, cols, depth, rows*cols*depth))

                print("Discriminator has %4.2fM parameters" % (disc_real.get_num_parameters()/1e6,))
                print()

            #
            # Losses and optimizers
            #
            gene_loss = _generator_loss(source_images, gene_out, disc_fake_out, annealing)
            
            disc_loss, disc_real_loss, disc_fake_loss = _discriminator_loss(disc_real_out, disc_fake_out)

            gene_opti = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                               name='gene_optimizer')

            # Note WGAN doesn't work well with Adam or any other optimizer that relies on momentum
            disc_opti = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=0.0,
                                                  name='disc_optimizer')

            gene_minimize = gene_opti.minimize(gene_loss, var_list=gene_var_list, name='gene_loss_minimize')    
            disc_minimize = disc_opti.minimize(disc_loss, var_list=disc_var_list, name='disc_loss_minimize')

            # Weight clipping a la WGAN (arXiv 1701.07875)
            # TBD: We shouldn't be clipping all variables (incl biases), just the weights
            disc_clip_weights = _clip_weights(disc_var_list, FLAGS.disc_weights_threshold)
            disc_minimize     = tf.group(disc_minimize, disc_clip_weights)

    # Package everything into an dumb object
    model = dm_utils.Container(locals())

    return model
