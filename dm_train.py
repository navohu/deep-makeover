import csv
import numpy as np
import os.path
import tensorflow as tf
import time

import glob
import re

import dm_arch
import dm_input
import dm_utils

FLAGS = tf.app.flags.FLAGS


def _save_image(train_data, feature, gene_output, batch, suffix, max_samples=None):
    """Saves a picture showing the current progress of the model"""
    
    if max_samples is None:
        max_samples = int(feature.shape[0])
    
    td  = train_data

    clipped  = np.clip(gene_output, 0, 1)
    image    = np.concatenate([feature, clipped], 2)

    image    = image[:max_samples,:,:,:]
    cols     = []
    num_cols = 4
    samples_per_col = max_samples//num_cols
    
    for c in range(num_cols):
        col   = np.concatenate([image[samples_per_col*c + i,:,:,:] for i in range(samples_per_col)], 0)
        cols.append(col)

    image   = np.concatenate(cols, 1)

    filename = 'batch%06d_%s.png' % (batch, suffix)
    filename = os.path.join(FLAGS.train_dir, filename)
    
    dm_utils.save_image(image, filename)


def _save_checkpoint(train_data, batch):
    """Saves a checkpoint of the model which can later be restored"""
    td = train_data

    oldname = 'checkpoint_old.txt'
    newname = 'checkpoint_new.txt'

    oldname = os.path.join(FLAGS.checkpoint_dir, oldname)
    newname = os.path.join(FLAGS.checkpoint_dir, newname)

    # Delete oldest checkpoint
    try:
        tf.gfile.Remove(oldname)
        tf.gfile.Remove(oldname + '.meta')
    except:
        pass

    # Rename old checkpoint
    try:
        tf.gfile.Rename(newname, oldname)
        tf.gfile.Rename(newname + '.meta', oldname + '.meta')
    except:
        pass

    # Generate new checkpoint
    saver = tf.train.Saver()
    saver.save(td.sess, newname)

    print("    Checkpoint saved")

def make_log_dir():
    """Create directories of logs for each time we run the model to track the performance"""
    files=glob.glob(FLAGS.logs + "/log_*.csv")
    file_nums=[]
    if len(files) is 0:
        new_number = 0
    else:
        for i, s in enumerate(files):
            num_str = re.search("(\d+).csv$",  files[i]) #capture only integer before ".csv" and EOL
            file_nums.append(parseInt(num_str.group(1)))  #convert to number

        new_number=max(file_nums)+1 #find largest and increment

    filename = os.path.join(FLAGS.logs, 'log_%06d.csv' % (new_number))
    with open(filename, "w") as empty_csv:
        writer = csv.writer(empty_csv)
        writer.writerow(['Step', 'Generator', 'Discriminator', 'Real', 'Fake'])

    return FLAGS.logs + filename

def save_values_to_csv(log_values, filename):
    with open(filename, 'a') as file:
        writer = csv.writer(file)
        writer.writerow(log_values)




def train_model(train_data):
    """Trains the given model with the given dataset"""
    td  = train_data
    tda = td.train_model
    tde = td.test_model

    dm_arch.enable_training(True)
    dm_arch.initialize_variables(td.sess)

    # Train the model
    minimize_ops = [tda.gene_minimize, tda.disc_minimize]
    show_ops     = [td.annealing, tda.gene_loss, tda.disc_loss, tda.disc_real_loss, tda.disc_fake_loss]

    start_time = time.time()
    step       = 0
    done       = False
    gene_decor = " "

    log_name = make_log_dir()

    print('\nModel training...')
    step = 0
    while not done:
        # Show progress with test features
        if step % FLAGS.summary_period == 0:
            feature, gene_mout = td.sess.run([tde.source_images, tde.gene_out])
            _save_image(td, feature, gene_mout, step, 'out')

        # Compute losses and show that we are alive
        annealing, gene_loss, disc_loss, disc_real_loss, disc_fake_loss = td.sess.run(show_ops)
        elapsed   = int(time.time() - start_time)/60
        print('  Progress[%3d%%], ETA[%4dm], Step [%5d], temp[%3.3f], %sgene[%-3.3f], *disc[%-3.3f] real[%-3.3f] fake[%-3.3f]' %
              (int(100*elapsed/FLAGS.train_time), FLAGS.train_time - elapsed, step,
               annealing, gene_decor, gene_loss, disc_loss, disc_real_loss, disc_fake_loss))

        log_values = [step, gene_loss, disc_loss, disc_real_loss, disc_fake_loss]
        save_values_to_csv(log_values, log_name)


        # Tight loop to maximize GPU utilization
        # TBD: Is there any way to make Tensorflow repeat multiple times an operation with a single sess.run call?
        if step < 200:
            # Discriminator doing poorly --> train discriminator only
            gene_decor = " "
            for _ in range(10):
                td.sess.run(tda.disc_minimize)
        else:
            # Discriminator doing well --> train both generator and discriminator, but mostly discriminator
            gene_decor = "*"
            for _ in range(2):
                td.sess.run(minimize_ops)
                td.sess.run(tda.disc_minimize)
                td.sess.run(tda.disc_minimize)
                td.sess.run(tda.disc_minimize)
        step += 1

        # Save checkpoint
        if step % FLAGS.checkpoint_period == 0:
           _save_checkpoint(td, step)

        # Finished?
        current_progress = elapsed / FLAGS.train_time
        if current_progress >= 1.0:
            done = True
        
        # Decrease annealing temperature exponentially
        if step % FLAGS.annealing_half_life == 0:
            td.sess.run(td.halve_annealing)


    _save_checkpoint(td, step)
    print('Finished training!')
