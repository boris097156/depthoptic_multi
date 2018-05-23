import tensorflow as tf
import numpy as np
import time
import imageio

flags = tf.app.flags
FLAGS = flags.FLAGS

def save_output_img(mask, gt_mask, img, name):
    mask = np.asarray(mask).reshape(-1, FLAGS.input_height, FLAGS.input_width)
    img = np.asarray(img).reshape(-1, FLAGS.input_height, FLAGS.input_width, 3)
    imageio.imwrite("{}pre_depth.png".format(name), mask[0])
    imageio.imwrite("{}gt_depth.png".format(name), gt_mask[0])                
    imageio.imwrite("{}input.jpg".format(name), img[0])
    
def save_output_gif(mask_gif_seq, gt_mask_gif_seq, img_gif_seq, name):
    imageio.mimsave("{}gt_depth.gif".format(name), gt_mask_gif_seq)
    imageio.mimsave("{}pre_depth.gif".format(name), mask_gif_seq)
    imageio.mimsave("{}input.gif".format(name), img_gif_seq)
    del mask_gif_seq[:]
    del img_gif_seq[:]
    del gt_mask_gif_seq[:]

def _load_filenames(file_name):
    img1_paths = []
    mask1_paths = []
    img2_paths = []
    mask2_paths = []
    optic_paths = []
    path_file = file_name
    with open(path_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            img1_paths.append((FLAGS.datapath_prefix + line[0]))
            mask1_paths.append((FLAGS.datapath_prefix + line[1]))
            img2_paths.append((FLAGS.datapath_prefix + line[2]))
            mask2_paths.append((FLAGS.datapath_prefix + line[3]))
            optic_paths.append((FLAGS.datapath_prefix + line[4]))
    return img1_paths, mask1_paths, img2_paths, mask2_paths, optic_paths, len(lines)

def _rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

def _read_npy_file(optic_string):
    data = np.load(optic_string.decode("utf-8")).astype(np.float32)
    #print(data.shape)
    resized_data = np.zeros((1, 2, FLAGS.input_height, FLAGS.input_width))
    resized_data[0, 0, :, :] += _rebin(data[0,0,:,:], (FLAGS.input_height, FLAGS.input_width))
    resized_data[0, 1, :, :] += _rebin(data[0,1,:,:], (FLAGS.input_height, FLAGS.input_width))
    #_rebin(data, (FLAGS.input_heightm FLAGS.input_width, 2))
    #print(resized_data.shape)
    return resized_data.astype(np.float32)

def _read_img(img_paths, ch):
    img_string = tf.read_file(img_paths)
    img = tf.image.decode_jpeg(img_string, channels=ch)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img = tf.image.resize_images(img, [FLAGS.input_height, FLAGS.input_width])
    return img

def _data_reader(img1_paths, mask1_paths, img2_paths, mask2_paths, optic_paths):
    img1 = _read_img(img1_paths, 3)
    mask1 = _read_img(mask1_paths, 1)
    img2 = _read_img(img2_paths, 3)
    mask2 = _read_img(mask2_paths, 1)
    optic = tf.py_func(_read_npy_file, [optic_paths], tf.float32)
    return img1, mask1, img2, mask2, optic

def create_dataset(file_name):
    img1_paths, mask1_paths, img2_paths, mask2_paths, optic_paths, data_size = _load_filenames(file_name)
    dataset = tf.data.Dataset.from_tensor_slices((img1_paths, mask1_paths, img2_paths, mask2_paths, optic_paths))
    dataset = dataset.map(_data_reader).batch(FLAGS.batch_size)
    return dataset, data_size

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def _get_pixel_value(img, x, y):
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))
    indices = tf.stack([b, y, x], 3)
    return tf.gather_nd(img, indices)

def tf_warp(img, flow, H, W):
    with tf.name_scope('warp_optic'):
        x,y = tf.meshgrid(tf.range(W), tf.range(H))
        x = tf.expand_dims(x,0)
        x = tf.expand_dims(x,0)
        y  =tf.expand_dims(y,0)
        y = tf.expand_dims(y,0)
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        grid  = tf.concat([x,y],axis = 1)
        flows = grid+flow
        max_y = tf.cast(H - 1, tf.int32)
        max_x = tf.cast(W - 1, tf.int32)
        zero = tf.zeros([], dtype=tf.int32)

        x = flows[:,0,:,:]
        y = flows[:,1,:,:]
        x0 = x
        y0 = y
        x0 = tf.cast(x0, tf.int32)
        x1 = x0 + 1
        y0 = tf.cast(y0,  tf.int32)
        y1 = y0 + 1

        # clip to range [0, H/W] to not violate img boundaries
        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)

        # get pixel value at corner coords
        Ia = _get_pixel_value(img, x0, y0)
        Ib = _get_pixel_value(img, x0, y1)
        Ic = _get_pixel_value(img, x1, y0)
        Id = _get_pixel_value(img, x1, y1)
        # recast as float for delta calculation
        x0 = tf.cast(x0, tf.float32)
        x1 = tf.cast(x1, tf.float32)
        y0 = tf.cast(y0, tf.float32)
        y1 = tf.cast(y1, tf.float32)

        # calculate deltas
        wa = (x1-x) * (y1-y)
        wb = (x1-x) * (y-y0)
        wc = (x-x0) * (y1-y)
        wd = (x-x0) * (y-y0)

        # add dimension for addition
        wa = tf.expand_dims(wa, axis=3)
        wb = tf.expand_dims(wb, axis=3)
        wc = tf.expand_dims(wc, axis=3)
        wd = tf.expand_dims(wd, axis=3)

        # compute output
        out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
        return out

def count_parameters():
    total_num_parameters = 0
    for variable in tf.trainable_variables():
        total_num_parameters += np.array(variable.get_shape().as_list()).prod()
    return total_num_parameters

def record_time(start_time, remain_ratio):
    now_time = time.time()
    elapsed_time = (now_time - start_time)/3600
    estimated_time_arrival = elapsed_time*remain_ratio
    return elapsed_time, estimated_time_arrival