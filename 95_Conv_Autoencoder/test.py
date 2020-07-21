from libs import *
from Model import *
import pickle

#===================================== TRAIN PATH ==========================================#

#train_path = '/home/lucianorosaserver/data_ml/trainings/43_Dafiti_dataset/2020-05-19_01:53:16_AM'
train_path = '/home/lucianorosaserver/data_ml/trainings/43_Dafiti_dataset/2020-05-19_07:22:44_AM'

#===========================================================================================#


#==================================== OUTPUT PATH ==========================================#

embeddings_path = './embeddings.pickle'

#===========================================================================================#

if __name__ == '__main__':

    # clear slashes end of path
    train_path = clearslash(train_path)

    dataset_path, _, _, _, learning_rate, decay_rate, _, _, _, max_digits_loss = read_train_cfg_file(train_path)[:10]

    test_split_size, _, _, _, _, img_size, img_depth = read_dataset_cfg_file(dataset_path)[6:]

    # get the number digits of number of batches
    batches_digits = len(str(test_split_size))

    # get path of tfrecors files of splits
    path_tfrecords_test = glob.glob(pathname=dataset_path + slash + 'test' + slash + '*.tfrecords')

    # sort data paths
    path_tfrecords_test.sort()

    with tf.Graph().as_default():

        # create dataset input pipelines of splits
        dataset_test = test_pipeline(_path_tfrecords=path_tfrecords_test, _img_size=img_size, _img_depth=img_depth,
                                     _batch_size=1)

        # create iterators for distribute data into model
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, dataset_test.output_types, dataset_test.output_shapes)
        next_batch = iterator.get_next()

        # create inializators por iterators datasets
        test_iterator = dataset_test.make_one_shot_iterator()

        is_training = tf.placeholder_with_default(False, (), 'is_training')
        global_step = tf.Variable(0, trainable=False, dtype=tf.int64)

        model = Model(next_batch, learning_rate, decay_rate, test_split_size, global_step, is_training)

        # create saver for save/restore model architecture and model weights
        saver = tf.train.Saver(max_to_keep=None)

        print()

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            test_handle = sess.run(test_iterator.string_handle())

            saver.restore(sess, tf.train.latest_checkpoint(train_path + slash + 'checkpoints' + slash))

            embeddings_arr = []

            test_loss_avg = 0
            count_test = 0

            while True:

                try:
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

                    test_loss, embeddings, _ = sess.run([model.loss, model.embeddings, update_ops], feed_dict={handle: test_handle, is_training: False})

                    test_loss_avg += test_loss

                    for i in embeddings:
                        embeddings_arr.append(i)

                    print("image: {}/{},   test_loss: {}".format(str(count_test + 1).rjust(batches_digits, '0'),
                                                                 test_split_size,
                                                                 str(round(test_loss * 100, 6)).ljust(max_digits_loss,
                                                                                                      '0')))
                    count_test += 1

                except tf.errors.OutOfRangeError:
                    break

            embeddings_arr = np.array(embeddings_arr)

            indexes = list(range(0, len(embeddings_arr)))
            data = {"indexes": indexes, "features": embeddings_arr}

            f = open('embeddings.pickle', "wb")
            f.write(pickle.dumps(data))
            f.close()

            print("\ntest_loss_avg: {}".format(str(round((test_loss_avg / test_split_size) * 100, 6)).ljust(max_digits_loss, '0')))