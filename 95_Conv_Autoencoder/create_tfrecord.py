from random import shuffle
from datetime import datetime
import multiprocessing as mp
from libs import *

#==================================== PATH INPUT =========================================#

#path_in = '/media/lucianorosaserver/SSD_SATA_3/datasets/01_DeepFashion/Category and Attribute Prediction Benckmark/Img/img_highres'
#path_in = '/media/lucianorosaserver/SSD_SATA_3/datasets/25_Fashion_MNIST/fashion-mnist-master/data/fashion/images'

#path_in = '/media/lucianorosaserver/SSD_SATA_3/datasets/43_dataset/aux_25000'
#path_in = '/media/lucianorosaserver/SSD_SATA_3/datasets/43_dataset/aux_50000'
#path_in = '/media/lucianorosaserver/SSD_SATA_3/datasets/43_dataset/aux_100000'
path_in = '/media/lucianorosaserver/SSD_SATA_3/datasets/43_dataset/aux_200000'

#=========================================================================================#


#=================================== PATH OUTPUT =========================================#

#path_out = '/home/lucianorosaserver/data_ml/tfrecords/01_DeepFashion'
#path_out = '/home/lucianorosaserver/data_ml/tfrecords/02_DeepFashion2'
#path_out = '/home/lucianorosaserver/data_ml/tfrecords/03_Street2shop'
#path_out = '/home/lucianorosaserver/data_ml/tfrecords/25_Fashion_MNIST'
path_out = '/home/lucianorosaserver/data_ml/tfrecords/43_dataset'

#=========================================================================================#

#shard_size = 64
#shard_size = 32
shard_size = 16

img_size = 512
#img_size = 256
#img_size = 128
#img_size = 28

img_depth = 3
#img_depth = 1

train_split_prct = 0.85
val_split_prct = 0.05
test_split_prct = 0.1

#train_split_prct = 0.8
#val_split_prct = 0.1
#test_split_prct = 0.1


# convert int64 value to tensor int64 list feature
def _int64_feature(_value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[_value]))


# convert bytes list value to tensor bytes list feature
def _bytes_feature(_value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[_value]))


# generate tfrecords files from shards
def create_tfrecord(_split_name: str, _split_paths: list, _split_size, _shard_id: int, _shard_size: int,
                    _folder_name: str, _split_digits: int, _shard_num: int, _shard_digits: int, _path_out: str,
                    _img_size: int = img_size, _img_depth: int = img_depth):

    # get id end of shard
    _end_id = _shard_id * _shard_size

    # get id ini of shard
    _ini_id = _end_id - _shard_size

    # get shard paths from range of shard id
    _shard_paths = _split_paths[_ini_id: _end_id]

    # get shard size
    _shard_size_aux = len(_shard_paths)

    # pad string shard id with leading zeros
    _shard_id_aux = str(_shard_id).zfill(_shard_digits)

    # pad string id ini with leading zeros
    _ini_id_aux = str(_ini_id + 1).zfill(_split_digits)

    # pad string id end with leading zeros
    _end_id_aux = str(_end_id).zfill(_split_digits)

    # generate path and name of file tfrecord (current shard)
    _path_file = str(_path_out) + slash + str(_folder_name) + slash + str(_split_name) + slash + str(_split_name) + '_' + str(
        _shard_id_aux) + '_' + str(_ini_id_aux) + '_' + str(_end_id_aux) + '_' + str(_shard_size_aux) + '.tfrecords'

    # open the TFRecords file
    _writer = tf.python_io.TFRecordWriter(_path_file)

    # loop number os shards (based on length of array _shard_size)
    for _shard_path in _shard_paths:

        try:
            # load, pre-process and encode image
            _img_raw = encode_img(_shard_path, _img_size, _img_depth)

            # Create a feature
            _feature = {'raw': _bytes_feature(_img_raw),
                        'path': _bytes_feature(_shard_path.encode('utf-8'))}

            # Create an example protocol buffer
            _example = tf.train.Example(features=tf.train.Features(feature=_feature))

            # Serialize to string and write on the file
            _writer.write(_example.SerializeToString())

        except:
            break

    # close the TFRecords file
    _writer.close()

    print(_split_name + ' shard: ' + str(_shard_id_aux) + slash + str(_shard_num) + ' (' + str(_ini_id_aux) + '-' + str(_end_id_aux) + ') - ' + str(
        _shard_size_aux))


# prepare splits for separation in shards and generate rfrecords
def process_split(_split_name: str, _split_paths: list, _shard_size: int, _shard_num: int, _folder_name: str, _path_out: str):

    # get time ini (current time)
    _ini_time_split = datetime.now()

    # create paths or empty directory
    if os.path.exists(_path_out + slash + _folder_name + slash + _split_name + slash):

        files = glob.glob(_path_out + slash + _folder_name + slash + _split_name + slash + '*.tfrecords')
        for f in files:
            os.remove(f)
    else:
        os.mkdir(_path_out + slash + _folder_name + slash + _split_name + slash)

    # get size of split array (train, val, test)
    _split_size = len(_split_paths)

    # get the number digits of split size
    _split_digits = len(str(abs(_split_size)))

    # get the number digits of number of shards
    _shard_digits = len(str(abs(_shard_num)))

    print('\nini process ' + _split_name + '... ')

    # create pool and define number os cpus
    pool = mp.Pool(mp.cpu_count())

    # exec async process of shards of split array
    pool.starmap_async(create_tfrecord, [(_split_name, _split_paths, _split_size, _shard_id + 1, _shard_size,
                                          _folder_name, _split_digits, _shard_num, _shard_digits, _path_out) for _shard_id in
                                         range(_shard_num)]).get()

    # close pool
    pool.close()

    # calculate time total
    _total_time_split = datetime.now() - _ini_time_split

    print('end process ' + _split_name + '...')

    return _total_time_split


if __name__ == '__main__':

    # clear slashes end of path
    path_in = clearslash(path_in)
    path_out = clearslash(path_out)

    # read image files from the dataset folder
    img_paths = get_all_file_paths(path_in, '.jpg')

    # abort process if no files founded
    if len(img_paths) == 0:
        print('\nprocess aborted, no such files in path: ' + str(path_in) + slash)
        exit(0)

    # get size of list image file paths
    dataset_size = len(img_paths)

    # shuffle data
    shuffle(img_paths)

    # get number of shards total
    shard_num_total = dataset_size // shard_size

    # get number of shards in train, val and split sets
    shard_num_train = round((shard_num_total * (train_split_prct * 100)) / 100)
    shard_num_val = round((shard_num_total * (val_split_prct * 100)) / 100)
    shard_num_test = round((shard_num_total * (test_split_prct * 100)) / 100)

    # split data into train, validation, and test sets
    train_split_paths = img_paths[0:round(shard_num_train * shard_size)]
    val_split_paths = img_paths[round(shard_num_train * shard_size):round((shard_num_train + shard_num_val) * shard_size)]
    test_split_paths = img_paths[round((shard_num_train + shard_num_val) * shard_size):]

    # get size of train, val and test split sets
    train_split_size = len(train_split_paths)
    val_split_size = len(val_split_paths)
    test_split_size = len(test_split_paths)

    # concatenate name of new folder from store .tfrecords files
    folder_name = str(dataset_size) + '_' + str(int(train_split_prct * 100)) + '_' + str(
        int(val_split_prct * 100)) + '_' + str(int(test_split_prct * 100)) + '_' + str(shard_size) + '_' + str(
        img_size) + '_' + str(img_depth)

    # calc num of shards (train, val and test)
    shard_train_num = get_shard_num(train_split_size, shard_size)
    shard_val_num = get_shard_num(val_split_size, shard_size)
    shard_test_num = get_shard_num(test_split_size, shard_size)

    print('\npath_in.............: ' + str(path_in))
    print('path_out............: ' + str(path_out) + slash + str(folder_name))

    print('\ntrain_split_prct....: ' + str(round(train_split_prct * 100)) + ' %')
    print('val_split_prct......: ' + str(round(val_split_prct * 100)) + ' %')
    print('test_split_prct.....: ' + str(round(test_split_prct * 100)) + ' %')

    print('\ndataset_size........: ' + str(dataset_size))
    print('train_split_size....: ' + str(dataset_size) + ' * ' + str(train_split_prct) + ' = ' + str(train_split_size))
    print('val_split_size......: ' + str(dataset_size) + ' * ' + str(val_split_prct) + ' = ' + str(val_split_size))
    print('test_split_size.....: ' + str(dataset_size) + ' * ' + str(test_split_prct) + ' = ' + str(test_split_size))

    print('\nshard_size..........: ' + str(shard_size))
    print('shard_train_num.....: ' + str(train_split_size) + slash + str(shard_size) + ' = ' + str(shard_train_num))
    print('shard_val_num.......: ' + str(val_split_size) + slash + str(shard_size) + ' = ' + str(shard_val_num))
    print('shard_test_num......: ' + str(test_split_size) + slash + str(shard_size) + ' = ' + str(shard_test_num))

    print('\nimg_size............: ' + str(img_size) + ' * ' + str(img_size))
    print('img_depth...........: ' + str(img_depth))

    # create path files
    if not os.path.exists(path_out + slash + folder_name + slash):
        os.mkdir(path_out + slash + folder_name + slash)

    # create config file
    create_dataset_cfg_file(path_in, path_out, train_split_prct, val_split_prct, test_split_prct, dataset_size,
                            train_split_size, val_split_size, test_split_size, shard_size, shard_train_num,
                            shard_val_num, shard_test_num, img_size, img_depth, folder_name)

    # create list of splits partitions
    create_list_partition(path_out, folder_name, train_split_paths, val_split_paths, test_split_paths)

    # get time ini (current time)
    ini_time = datetime.now()

    # process split train
    total_time_split_train = process_split('train', train_split_paths, shard_size, shard_train_num, folder_name, path_out)

    # process split val
    total_time_split_val = process_split('val', val_split_paths, shard_size, shard_val_num, folder_name, path_out)

    # process split test
    total_time_split_test = process_split('test', test_split_paths, shard_size, shard_test_num, folder_name, path_out)

    # calculate time total
    total_time = datetime.now() - ini_time

    print('\ntime total..........: ' + str(total_time))
    print('\ntrain...............: ' + str(total_time_split_train))
    print('val.................: ' + str(total_time_split_val))
    print('test................: ' + str(total_time_split_test))
