import tensorflow as tf
import os
import glob
import configparser
import cv2
import gzip
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelBinarizer
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

slash = '/'


def clear_path(_path):

    try:
        shutil.rmtree(_path + slash)
    except:
        pass

    try:
        os.mkdir(_path + slash)
    except:
        pass


def unpack_mnist(_path_in, _path_out):

    def load_mnist(_path, _name):

        labels_path = _path + slash + _name + '-labels-idx1-ubyte.gz'
        images_path = _path + slash + _name + '-images-idx3-ubyte.gz'

        with gzip.open(labels_path, 'rb') as lbpath:
            lbpath.read(8)
            buffer = lbpath.read()
            labels = np.frombuffer(buffer, dtype=np.uint8)

        with gzip.open(images_path, 'rb') as imgpath:
            imgpath.read(16)
            buffer = imgpath.read()
            images = np.frombuffer(buffer, dtype=np.uint8).reshape(len(labels), 28, 28)

        return images, labels

    clear_path(_path_out + slash + 'annotations')
    clear_path(_path_out + slash + 'images')

    imgs_train, lbls_train = load_mnist(_path_in, 'train')
    imgs_test, lbls_test = load_mnist(_path_in, 't10k')
    imgs = np.concatenate((imgs_train, imgs_test))
    lbls = np.concatenate((lbls_train, lbls_test))

    label_binarizer = LabelBinarizer()
    label_binarizer.fit(range(max(lbls) + 1))
    bin_lbls = label_binarizer.transform(lbls)

    digits = len(str(len(imgs_train)))

    with open(_path_out + slash + 'annotations' + slash + 'labels.txt', 'w') as txt_file:

        txt_file.write(str(len(bin_lbls)) + '\n')
        txt_file.write('name_img / label / one_hot_label' + '\n')

        for i in range(len(imgs)):
            name_img = str(i + 1).zfill(digits) + '.jpg'

            im = Image.fromarray(imgs[i])
            im.save(_path_out + slash + 'images' + slash + name_img)

            str_line = str(name_img) + '  ' + str(lbls[i]) + '  ' + ''.join(map(str, bin_lbls[i]))
            txt_file.write(str_line + '\n')


# set bar into end of string path file
def clearslash(_path: str):

    # if last character is not a bar
    if _path[len(_path) - 1:] == slash:
        _path = _path[:-1]

    # return path
    return _path


# get number of shards from split based on size of shards
def get_shard_num(_split_size: int, _shard_size: int):

    # get number of shards
    _shard_num = _split_size // _shard_size

    # increase number of shards if rest of division is greater than zero
    if (_split_size % _shard_size) != 0:
        _shard_num += 1

    return _shard_num


# get all images from input path dataset
def get_all_file_paths(_dir, _ext):

    def find_tree_sub_folders(_dir):

        _sub_folders = []
        if os.path.isdir(_dir):

            _sub_folders.append(_dir)

            _folders = os.listdir(_dir)
            _folders.sort()

            for _folder in _folders:
                _sub_dir = _dir + slash + _folder

                if os.path.isdir(_sub_dir):
                    _next_subfolders = find_tree_sub_folders(_sub_dir)

                    _sub_folders = _sub_folders + _next_subfolders

        return _sub_folders

    def find_img_files(_folders, _ext):

        _img_paths = []
        for _folder in _folders:

            _files = os.listdir(_folder)
            _files.sort()

            for _file in _files:
                if _file.endswith(_ext):
                    _img_paths.append(_folder + slash + _file)

        return _img_paths

    return find_img_files(find_tree_sub_folders(_dir), _ext)


def read_train_cfg_file(_path: str):

    configfile_name = ''
    try:
        # get image files paths in current directory
        configfile_name = glob.glob(_path + slash + '*.ini')
    except:
        pass

    configfile_name.sort()

    cfgfile = configparser.ConfigParser()
    cfgfile.read(configfile_name[0])

    return str(cfgfile['params']['dataset_path']), str(cfgfile['params']['model_name']), \
           int(cfgfile['params']['epochs']), int(cfgfile['params']['batch_size']), \
           float(cfgfile['params']['learning_rate']), float(cfgfile['params']['decay_rate']), \
           int(cfgfile['params']['ckpt_freq']), int(cfgfile['params']['ckpt_max_to_keep']), \
           int(cfgfile['params']['val_freq']), int(cfgfile['params']['max_digits_loss']), \
           int(cfgfile['params']['cpu_count']), bool(cfgfile['params']['resume_train']), \
           str(cfgfile['params']['total_time_train'])


def update_train_cfg_file(_out_path: str, _dataset_path: str, _model_name: str, _epochs: int, _batch_size: int,
                          _learning_rate: float, _decay_rate: float, _ckpt_freq: int, _val_freq: int,
                          _ckpt_max_to_keep: int, _max_digits_loss: int, _cpu_count: int, _resume_train: bool,
                          _total_time_train: str, _create_file: bool):

    n_files = len(get_all_file_paths(_out_path + slash, '.ini'))

    if _create_file: n_files += 1

    path_cfg = _out_path + slash + 'info' + str(n_files) + '.ini'

    filecfg = open(path_cfg, 'w')

    # Add content to the file
    Config = configparser.ConfigParser()

    Config.add_section('params')

    Config.set('params', 'dataset_path', str(_dataset_path))
    Config.set('params', 'model_name', str(_model_name))
    Config.set('params', 'epochs', str(_epochs))
    Config.set('params', 'batch_size', str(_batch_size))
    Config.set('params', 'learning_rate', str(_learning_rate))
    Config.set('params', 'decay_rate', str(_decay_rate))
    Config.set('params', 'ckpt_freq', str(_ckpt_freq))
    Config.set('params', 'val_freq', str(_val_freq))
    Config.set('params', 'ckpt_max_to_keep', str(_ckpt_max_to_keep))
    Config.set('params', 'max_digits_loss', str(_max_digits_loss))
    Config.set('params', 'cpu_count', str(_cpu_count))
    Config.set('params', 'resume_train', str(_resume_train))
    Config.set('params', 'total_time_train', str(_total_time_train))

    Config.write(filecfg)

    filecfg.close()


def create_list_partition(_path_out: str, _folder_name: str, train_split_paths: list, val_split_paths: list,
                              test_split_paths: list):

    with open(_path_out + slash + _folder_name + slash + 'list_partition.txt', 'w') as txt_file:

        for train_split_path in train_split_paths:
            txt_file.write(train_split_path + ' train' + '\n')

        for val_split_path in val_split_paths:
            txt_file.write(val_split_path + ' val' + '\n')

        for test_split_path in test_split_paths:
            txt_file.write(test_split_path + ' test' + '\n')


def create_dataset_cfg_file(_path_in: str, _path_out: str, _train_split_prct: float, _val_split_prct: float,
                            _test_split_prct: float, _dataset_size: int, _train_split_size: int, _val_split_size: int,
                            _test_split_size: int, _shard_size: int, _shard_train_num: int, _shard_val_num: int,
                            _shard_test_num: int, _img_size: int, _img_depth: int, _folder_name: str):

    path_cfg = _path_out + slash + _folder_name + slash + 'info.ini'

    try:
        os.remove(path_cfg)
    except:
        pass

    filecfg = open(path_cfg, 'w')

    # Add content to the file
    Config = configparser.ConfigParser()

    Config.add_section('params')

    Config.set('params', 'path_in', str(_path_in))
    Config.set('params', 'path_out', str(_path_out) + slash + str(_folder_name))
    Config.set('params', 'train_split_prct', str(_train_split_prct))
    Config.set('params', 'val_split_prct', str(_val_split_prct))
    Config.set('params', 'test_split_prct', str(_test_split_prct))
    Config.set('params', 'dataset_size', str(_dataset_size))
    Config.set('params', 'train_split_size', str(_train_split_size))
    Config.set('params', 'val_split_size', str(_val_split_size))
    Config.set('params', 'test_split_size', str(_test_split_size))
    Config.set('params', 'shard_size', str(_shard_size))
    Config.set('params', 'shard_train_num', str(get_shard_num(_train_split_size, _shard_size)))
    Config.set('params', 'shard_val_num', str(get_shard_num(_val_split_size, _shard_size)))
    Config.set('params', 'shard_test_num', str(get_shard_num(_test_split_size, _shard_size)))
    Config.set('params', 'img_size', str(_img_size))
    Config.set('params', 'img_depth', str(_img_depth))

    Config.write(filecfg)

    filecfg.close()


def read_dataset_cfg_file(_dataset_path: str):

    configfile_name = ''
    try:
        # get image files paths in current directory
        configfile_name = glob.glob(_dataset_path + slash + '*.ini')
    except:
        pass

    cfgfile = configparser.ConfigParser()
    cfgfile.read(configfile_name)

    return float(cfgfile['params']['train_split_prct']), float(cfgfile['params']['val_split_prct']), float(
        cfgfile['params']['test_split_prct']), int(cfgfile['params']['dataset_size']), int(
        cfgfile['params']['train_split_size']), int(cfgfile['params']['val_split_size']), int(
        cfgfile['params']['test_split_size']), int(cfgfile['params']['shard_size']), int(
        cfgfile['params']['shard_train_num']), int(cfgfile['params']['shard_val_num']), int(
        cfgfile['params']['shard_test_num']), int(cfgfile['params']['img_size']), int(cfgfile['params']['img_depth'])


# preprocess and encode image raw into tensor binary string bytes
def encode_img(_img_path, _img_size, _img_depth):

    # read image from disk
    img = cv2.imread(_img_path)

    # resize image to size especified
    img = cv2.resize(img, (_img_size, _img_size), interpolation=cv2.INTER_NEAREST)

    if _img_depth == 3:
        # convert image to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    else:
        # convert image to gray
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # convert pixels values to int
    img = img.astype(int)

    # compress back to jpg bytes
    _, img_bytes = cv2.imencode('.jpg', img)

    # return string bytes jpg
    return img_bytes.tostring()


def decode_img(_file_raw, _img_size: int, _img_depth: int):

    # Create a feature
    features = {'raw': tf.FixedLenFeature([], tf.string)}

    parsed_example = tf.parse_single_example(serialized=_file_raw, features=features)

    img_raw = parsed_example['raw']

    # Convert the image data from string back to the numbers
    img_decoded = tf.image.decode_jpeg(img_raw)

    # Reshape image data into the original shape
    img_decoded = tf.reshape(img_decoded, [_img_size, _img_size, _img_depth])

    if _img_depth == 3:

        # convert BGR to RGB
        img_decoded = tf.reverse(img_decoded, axis=[-1])

    # converto to float32
    img_decoded = tf.cast(img_decoded, tf.float32)

    # scale pixel values to [0, 1]
    img_decoded = img_decoded / 255

    return img_decoded


def train_pipeline(_path_tfrecords, _img_size: int, _img_depth: int, _epochs: int, _batch_size: int, _cpu_count: int,
                   _shard_size: int, _shard_num: int):

    with tf.device('/cpu:0'):

        dataset = tf.data.TFRecordDataset.list_files(file_pattern=_path_tfrecords)
        dataset = dataset.shuffle(buffer_size=_shard_num // _shard_size, seed=False, reshuffle_each_iteration=True)
        dataset = dataset.interleave(map_func=lambda x: tf.data.TFRecordDataset(x).map(
            lambda _file_raw: decode_img(_file_raw, _img_size, _img_depth), num_parallel_calls=_cpu_count),
                                     cycle_length=_cpu_count, block_length=1, num_parallel_calls=_cpu_count)
        dataset = dataset.shuffle(buffer_size=_shard_num // _shard_size, seed=False, reshuffle_each_iteration=True)
        dataset = dataset.batch(_batch_size)
        dataset = dataset.shuffle(buffer_size=_batch_size, seed=False, reshuffle_each_iteration=True)
        dataset = dataset.repeat(_epochs)

    return dataset


def val_pipeline(_path_tfrecords, _img_size: int, _img_depth: int, _batch_size: int, _cpu_count: int):

    with tf.device('/cpu:0'):

        dataset = tf.data.TFRecordDataset(_path_tfrecords, num_parallel_reads=_cpu_count)
        dataset = dataset.map(map_func=lambda _file_raw: decode_img(_file_raw, _img_size, _img_depth), num_parallel_calls=_cpu_count)
        dataset = dataset.batch(batch_size=_batch_size)
        dataset = dataset.prefetch(buffer_size=1)

    return dataset


def test_pipeline(_path_tfrecords, _img_size: int, _img_depth: int, _batch_size: int):

    with tf.device('/cpu:0'):

        dataset = tf.data.TFRecordDataset(_path_tfrecords)
        dataset = dataset.map(map_func=lambda _file_raw: decode_img(_file_raw, _img_size, _img_depth))
        dataset = dataset.batch(batch_size=_batch_size)

    return dataset