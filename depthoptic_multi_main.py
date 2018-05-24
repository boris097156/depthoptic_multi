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
flags.DEFINE_integer(   'total_epochs',                1,     'number of epochs')
flags.DEFINE_float(          'init_lr',             3e-5,     'initial learning rate')
flags.DEFINE_integer(      'using_gpu',                0,     'which GPU use for training')
flags.DEFINE_string(   'log_directory',        'record/',     'directory to save checkpoints and summaries')
flags.DEFINE_string( 'checkpoint_path',               '',     'path to a specific checkpoint to load')
flags.DEFINE_string(   'build_network',         'disnet',     'train network componants, disnet, ae')
flags.DEFINE_string(   'train_network',         'disnet',     'train network componants, disnet, ae')
flags.DEFINE_string(    'load_network',               '',     'train network componants, disnet, ae')
#----------------------------------------Testing parameters----------------------------------------------
flags.DEFINE_string(  'load_directory',               '',     'pretrained model')
flags.DEFINE_string('output_directory',        'output/',     'directory for output')
flags.DEFINE_boolean(       'save_gif',            False,     'if set, will save gif')
flags.DEFINE_boolean(       'save_img',            False,     'if set, will save img')

print("Using GPU:{}".format(FLAGS.using_gpu))
os.environ['CUDA_VISIBLE_DEVICES']=('{}'.format(FLAGS.using_gpu))

def configure():
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.5    
    return config

def train():
    with tf.Graph().as_default():
        train_dataset, train_datasize = create_dataset(FLAGS.datapath_file)
        train_dataset = train_dataset.shuffle(train_datasize + FLAGS.batch_size)
        iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
        train_init_op = iterator.make_initializer(train_dataset)
        img1, depth1, img2, depth2, optic = iterator.get_next()
        optic = tf.reshape(optic, [-1, 2, FLAGS.input_height, FLAGS.input_width])
        model = Model(img1, depth1, img2, depth2, optic, None)

        print("Training Data: {}".format(train_datasize))
        print("Trainable Parameters: {}".format(count_parameters()))

        with tf.Session(config=configure()) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            summary_writer = tf.summary.FileWriter("{}/{}".format(FLAGS.log_directory, FLAGS.model_name), sess.graph)
            steps_per_epoch = int(train_datasize/FLAGS.batch_size)
            total_steps = FLAGS.total_epochs * steps_per_epoch
            start_time = time.time()
            saver = tf.train.Saver()
            global_step = 0
            print("Training image:{}".format(train_datasize))
            print("Batch size:{}   Step per epoch:{}".format(FLAGS.batch_size, steps_per_epoch))
            print("Total epochs:{}  Total steps:{}".format(FLAGS.total_epochs, total_steps))
            for epoch in range(FLAGS.total_epochs):
                sess.run(train_init_op)
                for step in range(steps_per_epoch):
                    summary, loss, _, omax, omin = sess.run([model.merge_op, model.total_loss, model.train_op, model.max_optic, model.min_optic])
                    print("max:{}   min:{}".format(omax, omin))
                    if (step%int(steps_per_epoch/1)) == 0 and global_step > 0:
                        summary_writer.add_summary(summary, global_step+1)
                        elapsed_time, estimated_time_arrival = record_time(start_time, ((total_steps-global_step)/global_step))
                        print("{} epoch:{}/{} step:{} loss:{:.5f} ET:{:.2f}h ETA:{:.2f}h ({:.4f}%)".format(FLAGS.model_name, \
                            epoch+1, FLAGS.total_epochs, global_step, loss, elapsed_time, \
                            estimated_time_arrival, float(global_step*100)/total_steps))
                    global_step += 1

                if epoch>0 and epoch%2==0:
                    saver.save(sess, '{}/{}/{}'.format(FLAGS.log_directory, FLAGS.model_name, FLAGS.model_name), \
                        global_step=global_step)
            print("{} Complete Training".format(FLAGS.model_name))


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
