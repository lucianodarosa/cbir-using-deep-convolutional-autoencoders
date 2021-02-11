import matplotlib.pyplot as plt
import random
from libs import *

#==================================== DATASET PATH =========================================#

#dataset_path = '/home/lucianorosaserver/data_ml/tfrecords/01_DeepFashion/289212_85_5_10_32_512_3'
#dataset_path = '/home/lucianorosaserver/data_ml/tfrecords/25_Fashion_MNIST/70000_80_10_10_64_28_1'

#dataset_path = '/home/lucianorosaserver/data_ml/tfrecords/43_dataset/24703_85_5_10_16_512_3'
#dataset_path = '/home/lucianorosaserver/data_ml/tfrecords/43_dataset/49401_85_5_10_16_512_3'
#dataset_path = '/home/lucianorosaserver/data_ml/tfrecords/43_dataset/98620_85_5_10_16_512_3'
dataset_path = '/home/lucianorosaserver/data_ml/tfrecords/43_dataset/196651_85_5_10_16_512_3'


#===========================================================================================#


def show_images(_path_dataset, _img_size, _img_depth):

    with tf.Session() as sess:

        # Create a feature
        feature = {'raw': tf.FixedLenFeature([], tf.string),
                   'path': tf.FixedLenFeature([], tf.string)}

        # Create a list of filenames and pass it to a queue
        filename_queue = tf.train.string_input_producer([_path_dataset], num_epochs=1)

        # Define a reader and read the next record
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        # Decode the record read by the reader
        features = tf.parse_single_example(serialized_example, features=feature)

        # Convert the image data from string back to the numbers
        img = tf.image.decode_jpeg(features['raw'])
        path = tf.cast(features['path'], dtype=tf.string)

        if _img_depth == 3:

            # Reshape image data into the original shape
            img = tf.reshape(img, [_img_size, _img_size, _img_depth])

            # convert BGR to RGB
            img = tf.reverse(img, axis=[-1])

        else:
            # Reshape image data into the original shape
            img = tf.reshape(img, [_img_size, _img_size])

        # Creates batches by randomly shuffling tensors
        images = tf.train.shuffle_batch([img, path], batch_size=10, capacity=30, num_threads=1,
                                        min_after_dequeue=10)

        # Initialize all global and local variables
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        sess.run(init_op)

        # Create a coordinator and run all QueueRunner objects
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        img = sess.run([images])

        rows, cols = 2, 5
        f, axarr = plt.subplots(rows, cols, figsize=(18, 8))

        count = 0
        for i in range(rows):
            for j in range(cols):

                img_raw = img[0][0][count]
                img_path = img[0][1][count]

                axarr[i, j].imshow(img_raw, cmap='gray')
                print(os.path.basename(img_path).decode('utf-8'))
                axarr[i, j].title.set_text(os.path.basename(img_path).decode('utf-8'))
                print(str(_img_size) + "," + str(_img_size) + ", " + str(_img_depth))

                count += 1

        print()

        plt.show()
        plt.close()

        # Stop the threads
        coord.request_stop()

        # Wait for threads to stop
        coord.join(threads)
        sess.close()


if __name__ == '__main__':

    # clear slashes end of path
    dataset_path = clearslash(dataset_path)

    # get size and depth of images inside tfrecords
    img_size, img_depth = read_dataset_cfg_file(dataset_path)[11:]

    # get all tfrecord files into path
    path_tfrecords_train = glob.glob(dataset_path + slash + 'train' + slash + '*.tfrecords')
    path_tfrecords_val = glob.glob(dataset_path + slash + 'val' + slash + '*.tfrecords')
    path_tfrecords_test = glob.glob(dataset_path + slash + 'test' + slash + '*.tfrecords')

    # choice random tfrecord file
    path_tfrecord_train = random.choice(path_tfrecords_train)
    path_tfrecord_val = random.choice(path_tfrecords_val)
    path_tfrecord_test = random.choice(path_tfrecords_test)

    # show images from tfrecord files
    show_images(path_tfrecord_train, img_size, img_depth)
    show_images(path_tfrecord_val, img_size, img_depth)
    show_images(path_tfrecord_test, img_size, img_depth)
