from __future__ import division
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'

import numpy as np
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import imageio

from depthoptic_multi_utils import *
from depthoptic_multi_model import *

flags = tf.app.flags
FLAGS = flags.FLAGS

NET_LIST = ['disnet']

flags.DEFINE_string(            'mode',           'test',     'train or test')
flags.DEFINE_string(      'model_name',    'depth_optic',     'model name for saving')
#----------------------------------------Data information------------------------------------------------
flags.DEFINE_string( 'datapath_prefix',               '',     'path to the data')
flags.DEFINE_string(   'datapath_file',               '',     'path to the filenames text file')
flags.DEFINE_integer(   'input_height',              256,     'input height')
flags.DEFINE_integer(    'input_width',              480,     'input width')
#----------------------------------------Training parameters---------------------------------------------
flags.DEFINE_integer(     'batch_size',                8,     'size of batch')
flags.DEFINE_integer(     'num_epochs',                1,     'number of epochs')
flags.DEFINE_float(          'init_lr',             3e-5,     'initial learning rate')
flags.DEFINE_string(            'gpus',              '0',     'which GPU use for training')
flags.DEFINE_integer(    'num_threads',                8,     'number of threads to use for data loading')
flags.DEFINE_string(   'log_directory',        'record/',     'directory to save checkpoints and summaries')
flags.DEFINE_string( 'checkpoint_path',               '',     'path to a specific checkpoint to load')

flags.DEFINE_string(   'build_network',         'disnet',     'train network componants, disnet, ae')
flags.DEFINE_string(   'train_network',         'disnet',     'train network componants, disnet, ae')
flags.DEFINE_string(    'load_network',               '',     'train network componants, disnet, ae')
#----------------------------------------Testing parameters----------------------------------------------
flags.DEFINE_string(  'load_directory',               '',     'pretrained model')
flags.DEFINE_string('output_directory',        'output/',     'directory for output')
flags.DEFINE_boolean(       'save_gif',            False,     'if set, will save gif')
flags.DEFINE_boolean(       'save_img',             True,     'if set, will save img')


gpu_list = FLAGS.gpus.strip().split()
print("Using GPU:{}".format(gpu_list))
os.environ['CUDA_VISIBLE_DEVICES']=','.join([str(i) for i in gpu_list])

def count_text_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)

def train():
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        global_step = tf.Variable(0, trainable=False)

        num_training_samples = count_text_lines(FLAGS.datapath_file)
        steps_per_epoch      = np.floor(num_training_samples / FLAGS.batch_size).astype(np.int32)
        num_total_steps      = FLAGS.num_epochs * steps_per_epoch
        num_steps            = int(num_training_samples/FLAGS.batch_size)
        boundaries           = [np.int32((3/5) * num_total_steps), np.int32((4/5) * num_total_steps)]
        values               = [FLAGS.init_lr, FLAGS.init_lr/2, FLAGS.init_lr/4]
        learning_rate        = tf.train.piecewise_constant(global_step, boundaries, values)

        print("total number of samples: {}".format(num_training_samples))
        print("total number of steps:   {}".format(num_total_steps))
        train_dataset, train_datasize      = create_dataset(FLAGS.datapath_file)
        train_dataset = train_dataset.shuffle(train_datasize + FLAGS.batch_size)
        
        iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
        train_init_op = iterator.make_initializer(train_dataset)
        img1, depth1, img2, depth2, optic = iterator.get_next()
        optic = tf.reshape(optic, [-1, 2, FLAGS.input_height, FLAGS.input_width])
        
        ## split data for each gpu
        img1_splits     = tf.split(img1,  len(gpu_list), 0)
        depth1_splits      = tf.split(depth1,   len(gpu_list), 0)
        img2_splits     = tf.split(img2,  len(gpu_list), 0)
        depth2_splits      = tf.split(depth2,   len(gpu_list), 0)
        optic_splits     = tf.split(optic,   len(gpu_list), 0)

        G_tower_grads    = []
        G_tower_losses   = []
        G_opt_step           = tf.train.AdamOptimizer(learning_rate)

        reuse_variables  = None
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(len(gpu_list)):
                with tf.device('/gpu:%d' % i):
                    model           = Model(img1_splits[i], depth1_splits[i], img2_splits[i], depth2_splits[i], \
                                            optic_splits[i], reuse_variables, i)
                    G_loss          = model.optic_loss
                    G_tower_losses.append(G_loss)
                    reuse_variables = True
                    G_grads         = G_opt_step.compute_gradients(G_loss,var_list=\
                    [var for var_list in [getattr(model, net+"_vars") for net in [FLAGS.train_network]] for var in var_list])
                    G_tower_grads.append(G_grads)

        ## G update
        G_grads                     = average_gradients(G_tower_grads)
        apply_G_gradient_op         = G_opt_step.apply_gradients(G_grads, global_step=global_step)
        G_loss                      = tf.reduce_mean(G_tower_losses)

        summary_op                  = tf.summary.merge_all('model_0')

        # SESSION
        config     = tf.ConfigProto()
        config.allow_soft_placement=True
        #config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        sess       = tf.Session(config=config)

        # SAVER
        summary_writer     = tf.summary.FileWriter(FLAGS.log_directory + '/' + FLAGS.model_name, sess.graph)
        
        train_saver = {}
        for net in [FLAGS.build_network]:
            train_saver[net] = tf.train.Saver(var_list=getattr(model, net + "_vars"))

        # COUNT PARAMS 
        total_num_parameters = 0
        for variable in tf.trainable_variables():
            total_num_parameters += np.array(variable.get_shape().as_list()).prod()
        print("number of trainable parameters: {}".format(total_num_parameters))

        # INIT
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

        # Restore
        for net in [FLAGS.load_network]:
            if net == '':
                break
            print("---Load network : " + net)
            train_saver[net].restore(sess,FLAGS.load_directory + net.upper())
            print("Load network complete")

        # GO!
        start_step = global_step.eval(session=sess)
        start_time = time.time()
        print('Start training')
        for epoch in range(FLAGS.num_epochs):
            sess.run(train_init_op)
            for step in range(num_steps):
                before_op_time = time.time()
                _, loss_value, summary_str = sess.run([apply_G_gradient_op, G_loss, summary_op])
                duration = time.time() - before_op_time
                if step % 100 == 0:
                    examples_per_sec = FLAGS.batch_size / duration
                    time_sofar = (time.time() - start_time) / 3600
                    training_time_left = (num_total_steps / (step+num_steps*epoch) - 1.0) * time_sofar
                    print_string = 'batch {:>6} | examples/s: {:4.2f} | \
                        loss: {:.5f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                    print(print_string.format((step+num_steps*epoch), examples_per_sec, loss_value, time_sofar, training_time_left))
                    summary_writer.add_summary(summary_str, global_step=step+num_steps*epoch)
                    
                if step and step % 1000 == 0:
                    for net in [FLAGS.train_network]:
                        train_saver[net].save(sess, FLAGS.log_directory + '/' + FLAGS.model_name+'/'+net.upper())
                        
        for net in [FLAGS.train_network]:
            train_saver[net].save(sess, FLAGS.log_directory + '/' + FLAGS.model_name+'/'+net.upper())
        print('Training {} complete'.format(FLAGS.model_name))

def test():
    print ('Create dataset')
    test_dataset, test_datasize      = create_dataset(FLAGS.datapath_file)
    iterator = tf.data.Iterator.from_structure(test_dataset.output_types, test_dataset.output_shapes)
    test_init_op = iterator.make_initializer(test_dataset)
    img1, depth1, img2, depth2, optic = iterator.get_next()
    print ('Create dataset complete')
    print ('Test Data Imgs:{}'.format(int(test_datasize*2)))
    optic = tf.reshape(optic, [-1, 2, FLAGS.input_height, FLAGS.input_width])
    
    print ('Create model')
    model = Model(img1, depth1, img2, depth2, optic)
    train_saver = {}
    for net in [FLAGS.build_network]:
        train_saver[net] = tf.train.Saver(var_list=getattr(model, net + "_vars"))
    print ('Create model complete')
    
    #Session Init
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(test_init_op)
    summary_writer     = tf.summary.FileWriter(FLAGS.log_directory + '/' + FLAGS.model_name, sess.graph)
    coordinator = tf.train.Coordinator()
    threads     = tf.train.start_queue_runners(sess=sess, coord=coordinator)
    
    print('Restore weights from {}{}'.format(FLAGS.log_directory, FLAGS.model_name))
    for net in [FLAGS.load_network]:
        print("Load network : " + net)
        train_saver[net].restore(sess, FLAGS.log_directory + '/' + FLAGS.model_name +'/'+ net.upper())
    print('Restore weights complete')
    
    depth_gif_seq = []
    img_gif_seq  = []
    gt_depth_gif_seq = []
    total_depth_l1 = 0.0
    total_depth_l2 = 0.0
    total_depth_consist = 0.0
    print('Start testing: {} imgs'.format(test_datasize*2))
    print('Saving output to directory: {}'.format(FLAGS.model_name))
    line = 'Saving '
    if FLAGS.save_gif:
        line = line + 'gif '
    if FLAGS.save_img:
        line = line + 'img '
    print('{}'.format(line))
    for step in range(test_datasize):
        depth1, gt_depth1, img1, depth2, gt_depth2, img2, depth_l1, depth_l2, depth_consist =\
         sess.run([model.pre_depth1_pyramid[0], model.gt_depth1_pyramid[0], model.input_img1_pyramid[0], \
                   model.pre_depth2_pyramid[0], model.gt_depth2_pyramid[0], model.input_img2_pyramid[0], \
                   model.depth_l1, model.depth_l2, model.depth_consist])
        
        if step>0 and step%100==0 :
            print('Testing {}/{} ({:.2f}%)'.format(step, test_datasize, (100.0*step/test_datasize)))
        
        total_depth_l1      += depth_l1
        total_depth_l2      += depth_l2
        total_depth_consist += depth_consist
            
        if FLAGS.save_img:
            save_img_name = "{}/{}/img/{}_".format(FLAGS.output_directory, FLAGS.model_name, step*2+1)
            save_output_img(depth1, gt_depth1, img1, save_img_name)
            save_img_name = "{}/{}/img/{}_".format(FLAGS.output_directory, FLAGS.model_name, step*2+2)
            save_output_img(depth2, gt_depth2, img2, save_img_name)

        if FLAGS.save_gif:
            depth_gif_seq.append(depth1[0])
            gt_depth_gif_seq.append(gt_depth1[0])
            img_gif_seq.append(img1[0])
            depth_gif_seq.append(depth2[0])
            gt_depth_gif_seq.append(gt_depth2[0])
            img_gif_seq.append(img2[0])
            if step>0 and step%100==0 :
                save_gif_name = "{}/{}/gif/{}_".format(FLAGS.output_directory, FLAGS.model_name, step)
                save_output_gif(depth_gif_seq, gt_depth_gif_seq, img_gif_seq, save_gif_name)
            
    print('Complete testing {}'.format(FLAGS.model_name))
    print('Depth L1 Error: {:.4f}'.format(total_depth_l1/test_datasize))
    print('Depth L2 Error: {:.4f}'.format(total_depth_l2/test_datasize))
    print('Depth Consistency Error: {:.4f}'.format(total_depth_consist/test_datasize))
    
def main(unused_argv):
    if FLAGS.mode == 'train':
        train()
    elif FLAGS.mode == 'test':
        test()

if __name__ == '__main__':
    tf.app.run()
