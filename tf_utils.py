import tensorflow as tf
import tensorflow_addons as tfa
import utils
import numpy as np

PI = np.math.pi

def rgb2gray(rgb):
    red, green, blue = tf.split(rgb, num_or_size_splits=3, axis=-1)
    gray = 0.2627*red + 0.6780*green + 0.0593*blue
    return gray

def bgr2gray(bgr):
    blue, green, red = tf.split(bgr, num_or_size_splits=3, axis=-1)
    gray = 0.2627*red + 0.6780*green + 0.0593*blue
    return gray

def rgb2bgr(rgb):
    red, green, blue = tf.split(rgb, num_or_size_splits=3, axis=-1)
    bgr = tf.concat([blue, green, red], axis=3)
    return bgr

def bgr2rgb(bgr):
    blue, green, red = tf.split(bgr, num_or_size_splits=3, axis=-1)
    rgb = tf.concat([red, green, blue], axis=3)
    return rgb

def sphere2world(sunpose, h, w, skydome = True):
    x, y = sunpose
    
    unit_w = tf.divide(2 * PI, w)
    unit_h = tf.divide(PI, h * 2 if skydome else h)
    
    # degree in xy coordinate to radian
    theta = (x - 0.5 * w) * unit_w
    phi   = (h - y) * unit_h if skydome else (h * 0.5 - y) * unit_h

    x_u = tf.math.cos(phi) * tf.math.cos(theta)
    y_u = tf.math.sin(phi)
    z_u = tf.math.cos(phi) * tf.math.sin(theta)
    p_u = [x_u, y_u, z_u]

    return tf.convert_to_tensor(p_u)

def pano2world(i, h, w):
    # xy coord to degree
    # gap value + init (half of the gap value)

    x = ((i+1.) - tf.floor(i/w) * w - 1.) * (360.0/w) + (360.0/(w*2.)) 
    y = (tf.floor(i/w)) * (90./h) + (90./(2.*h))

    # deg2rad
    phi = (y) * (PI / 180.)
    theta = (x - 180.0) * (PI / 180.)

    # rad2xyz
    x_u = tf.math.cos(phi) * tf.math.cos(theta)
    y_u = tf.math.sin(phi)
    z_u = tf.math.cos(phi) * tf.math.sin(theta)
    p_u = [x_u, y_u, z_u]
    
    return tf.convert_to_tensor(p_u)

def positional_encoding(_input, with_r=False):
    # coord conv
    b, h, w = _input.get_shape()[0:3]

    w_range = tf.linspace(-1., 1., w)
    h_range = tf.linspace(-1., 1., h)
    x, y = tf.meshgrid(w_range, h_range)
    x, y = [tf.reshape(i, [1, h, w, 1]) for i in [x, y]]
    normalized_coord = tf.concat([x,y], axis=-1)

    if with_r:
        half_h = h * 0.5
        half_w = w * 0.5
        r = tf.sqrt(tf.square(x - half_w) + tf.square(y - half_h))
        normalized_coord = tf.concat([normalized_coord, r], axis=-1)

    normalized_coord = tf.tile(normalized_coord, [b,1,1,1])
    pose_aware_input = tf.concat([_input, normalized_coord], axis=-1)

    return pose_aware_input

def log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator
    
def get_tensor_shape(x):
    a = x.get_shape().as_list()
    b = [tf.shape(x)[i] for i in range(len(a))]
    def _select_one(aa, bb):
        if type(aa) is int:
            return aa
        else:
            return bb
    return [_select_one(aa, bb) for aa, bb in zip(a, b)]

def sample_1d(
    img,   # [b, h, c]
    y_idx, # [b, n], 0 <= pos < h, dtpye=int32
):
    b, h, c = get_tensor_shape(img)
    b, n    = get_tensor_shape(y_idx)
    
    b_idx = tf.range(b, dtype=tf.int32) # [b]
    b_idx = tf.expand_dims(b_idx, -1)   # [b, 1]
    b_idx = tf.tile(b_idx, [1, n])      # [b, n]
    
    y_idx = tf.clip_by_value(y_idx, 0, h - 1) # [b, n]
    a_idx = tf.stack([b_idx, y_idx], axis=-1) # [b, n, 2]
    
    return tf.gather_nd(img, a_idx)

def interp_1d(
    img, # [b, h, c]
    y,   # [b, n], 0 <= pos < h, dtype=float32
):
    b, h, c = get_tensor_shape(img)
    b, n    = get_tensor_shape(y)
    
    y_0 = tf.floor(y) # [b, n]
    y_1 = y_0 + 1    
    
    _sample_func = lambda y_x: sample_1d(
        img,
        tf.cast(y_x, tf.int32)
    )
    y_0_val = _sample_func(y_0) # [b, n, c]
    y_1_val = _sample_func(y_1)
    
    w_0 = y_1 - y # [b, n]
    w_1 = y - y_0
    
    w_0 = tf.expand_dims(w_0, -1) # [b, n, 1]
    w_1 = tf.expand_dims(w_1, -1)
    
    return w_0*y_0_val + w_1*y_1_val

def hdr_logCompression(x, validDR = 10.):
    # 0~1
    # disentangled way
    x = tf.math.multiply(validDR, x)
    numerator = tf.math.log(1.+ x)
    denominator = tf.math.log(1.+validDR)
    output = tf.math.divide(numerator, denominator)

    return output

def hdr_logDecompression(x, validDR = 10.):
    # 0~1
    denominator = tf.math.log(1.+validDR)
    x = tf.math.multiply(x, denominator)
    x = tf.math.exp(x)
    output = tf.math.divide(x-1., validDR)
    
    return output

def createDirectories(path, name="name", dir="dir"):
    
    path = utils.createNewDir(path, dir)
    root_logdir = utils.createNewDir(path, name)
    logdir = utils.createNewDir(root_logdir)

    if dir=="tensorboard":
        train_logdir, test_logdir = utils.createTrainValidationDirpath(logdir, createDir=False)
        train_summary_writer = tf.summary.create_file_writer(train_logdir)
        test_summary_writer = tf.summary.create_file_writer(test_logdir)
        return train_summary_writer, test_summary_writer, logdir

    if dir=="outputImg":
        train_logdir, test_logdir = utils.createTrainValidationDirpath(logdir, createDir=True)
        return train_logdir, test_logdir

def checkpoint_initialization(model_name : str,
                                pretrained_dir : str,
                                checkpoint_path : str,
                                model="model",
                                optimizer="optimizer",
                                ):
    if pretrained_dir is None:
        checkpoint_path = utils.createNewDir(checkpoint_path, model_name)
    else: checkpoint_path = pretrained_dir
    
    ckpt = tf.train.Checkpoint(
                            epoch = tf.Variable(0),
                            lin=model,
                           optimizer=optimizer,)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    #  if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest {} checkpoint has restored!!'.format(model_name))

    return ckpt, ckpt_manager

def metric_initialization(model_name : str, lr = "lr"):
    
    optimizer = tf.keras.optimizers.Adam(lr)
    train_loss = tf.keras.metrics.Mean(name= 'train_loss_{}'.format(model_name), dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean(name='test_loss_{}'.format(model_name), dtype=tf.float32)

    return optimizer, train_loss, test_loss