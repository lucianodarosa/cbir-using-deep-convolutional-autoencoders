from random import shuffle
from datetime import datetime
from libs import *
from Model import Model
import sys

#==================================== DATASET PATH =========================================#

#dataset_path = '/home/lucianorosaserver/data_ml/tfrecords/01_DeepFashion/289212_85_5_10_32_512_3'
#dataset_path = '/home/lucianorosaserver/data_ml/tfrecords/02_DeepFashion2/'
#dataset_path = '/home/lucianorosaserver/data_ml/tfrecords/03_Street2shop/'
#dataset_path = '/home/lucianorosaserver/data_ml/tfrecords/25_Fashion_MNIST/'

#dataset_path = '/home/lucianorosaserver/data_ml/tfrecords/43_Dafiti_dataset/24703_85_5_10_32_512_3'
#dataset_path = '/home/lucianorosaserver/data_ml/tfrecords/43_Dafiti_dataset/49401_85_5_10_32_512_3'
#dataset_path = '/home/lucianorosaserver/data_ml/tfrecords/43_Dafiti_dataset/98620_85_5_10_32_512_3'

dataset_path = '/home/lucianorosaserver/data_ml/tfrecords/43_Dafiti_dataset/24703_85_5_10_16_512_3'
#dataset_path = '/home/lucianorosaserver/data_ml/tfrecords/43_Dafiti_dataset/49401_85_5_10_16_512_3'
#dataset_path = '/home/lucianorosaserver/data_ml/tfrecords/43_Dafiti_dataset/98620_85_5_10_16_512_3'
#dataset_path = '/home/lucianorosaserver/data_ml/tfrecords/43_Dafiti_dataset/196651_85_5_10_16_512_3'

#===========================================================================================#


#==================================== OUTPUT PATH ==========================================#

#out_path = '/home/lucianorosaserver/data_ml/trainings/01_DeepFashion'
#out_path = '/home/lucianorosaserver/data_ml/trainings/02_DeepFashion2'
#out_path = '/home/lucianorosaserver/data_ml/trainings/03_Street2shop'
#out_path = '/home/lucianorosaserver/data_ml/trainings/25_Fashion_MNIST'
out_path = '/home/lucianorosaserver/data_ml/trainings/43_Dafiti_dataset'

#===========================================================================================#

epochs = 30
#batch_size = 64
batch_size = 19
#batch_size = 16
learning_rate = 0.1
decay_rate = 0.96
#ckpt_freq = 20
ckpt_freq = batch_size * 50
val_freq = batch_size * 50
#val_freq = batch_size * 10
ckpt_max_to_keep = 10
max_digits_loss = 9
cpu_count = 12
resume_train = False
latest_train_path = ''

if __name__ == '__main__':

    # clear slashes end of path
    dataset_path = clearslash(dataset_path)
    out_path = clearslash(out_path)
    latest_train_path = clearslash(latest_train_path)

    train_split_prct, val_split_prct, _, dataset_size, train_split_size, val_split_size, \
    _, shard_size, shard_train_num, shard_val_num, _, img_size, img_depth = \
        read_dataset_cfg_file(dataset_path)

    # get number of batches of splits
    batches_train = train_split_size // batch_size
    batches_val = val_split_size // batch_size

    # get number of global steps
    global_steps = epochs * batches_train

    # get the number digits of number of batches
    batches_digits = len(str(batches_train))

    # get the number digits of number of epochs
    epochs_digits = len(str(epochs))

    # get the number digits of global steps
    global_steps_digits = len(str(global_steps))

    # get path of tfrecors files of splits
    path_tfrecords_train = glob.glob(pathname=dataset_path + slash + 'train' + slash + '*.tfrecords')
    path_tfrecords_val = glob.glob(pathname=dataset_path + slash + 'val' + slash + '*.tfrecords')

    # sort data paths
    path_tfrecords_train.sort()
    path_tfrecords_val.sort()

    # shuffle data paths
    shuffle(path_tfrecords_train)
    shuffle(path_tfrecords_val)

    # create tf graph
    with tf.Graph().as_default():

        # create train dataset input pipeline
        dataset_train = train_pipeline(_path_tfrecords=path_tfrecords_train, _img_size=img_size, _img_depth=img_depth,
                                       _epochs=epochs, _batch_size=batch_size, _cpu_count=cpu_count,
                                       _shard_size=shard_size, _shard_num=shard_train_num)

        # create val dataset input pipeline
        dataset_val = val_pipeline(_path_tfrecords=path_tfrecords_val, _img_size=img_size, _img_depth=img_depth,
                                   _batch_size=batch_size, _cpu_count=cpu_count)

        # create iterators for distribute data into model
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, dataset_train.output_types, dataset_train.output_shapes)
        next_batch = iterator.get_next()

        # create inializators por iterators datasets
        train_iterator = dataset_train.make_initializable_iterator()
        val_iterator = dataset_val.make_initializable_iterator()

        is_training = tf.placeholder_with_default(False, (), 'is_training')
        global_step = tf.Variable(0, trainable=False, dtype=tf.int64)

        model = Model(next_batch, learning_rate, decay_rate, batches_train, global_step, is_training)

        #[print(n.name) for n in tf.get_default_graph().as_graph_def().node]
        #exit(0)

        '''
        total_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.all_variables()])
        print('Total params: ' + str(total_params))

        trainable_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print('Trainable params: ' + str(trainable_params))

        print('Non-trainable params: ' + str(total_params - trainable_params))
        '''

        # create summary for saving checkpoints of training
        summary_train_loss = tf.summary.scalar('train_loss', model.loss * 100)
        summary_eval_loss = tf.summary.scalar('eval_loss', model.loss * 100)
        summary_merged = tf.summary.merge_all()

        # create saver for save/restore model architecture and model weights
        saver = tf.train.Saver(max_to_keep=ckpt_max_to_keep)

        print()

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            training_handle = sess.run(train_iterator.string_handle())
            validation_handle = sess.run(val_iterator.string_handle())

            if resume_train and bool(latest_train_path.strip()) and os.path.isdir(out_path + slash + latest_train_path):

                train_path = latest_train_path

                try:
                    saver.restore(sess, tf.train.latest_checkpoint(out_path + slash + train_path + slash + 'checkpoints'))

                    epoch_count = global_step.eval() // batches_train
                    batch_count = abs(global_step.eval() - (epoch_count * batches_train))
                except:
                    epoch_count = 0
                    batch_count = 0
            else:

                train_path = datetime.now().strftime("%Y-%m-%d_%I:%M:%S_%p")
                os.mkdir(out_path + slash + train_path)

                epoch_count = 0
                batch_count = 0

            os.system("gnome-terminal --tab -e 'bash -c \""
                      "export WORKON_HOME=$HOME/.virtualenvs && "
                      "export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3 && "
                      "source /usr/local/bin/virtualenvwrapper.sh && "
                      "workon venv1 && "
                      "tensorboard --logdir " + (out_path + slash + train_path + slash + 'summaries') + " \" '")

            # initialize dataset iterator
            sess.run(train_iterator.initializer)

            # export metagraph
            saver.export_meta_graph(filename=out_path + slash + train_path + slash + 'checkpoints' + slash + "model.meta")

            # create train and val summaries writers
            writer_train = tf.summary.FileWriter(out_path + slash + train_path + slash + 'summaries' + slash + 'train')
            writer_val = tf.summary.FileWriter(out_path + slash + train_path + slash + 'summaries' + slash + 'val')

            # create training config file
            update_train_cfg_file(out_path + slash + train_path, dataset_path, 'model_1', epochs, batch_size, learning_rate, decay_rate,
                                  ckpt_freq, ckpt_max_to_keep, val_freq, max_digits_loss, cpu_count, resume_train, 0, True)

            ini_time_train = datetime.now()

            while True:

                try:

                    ini_time_batch_train = datetime.now()

                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

                    _, _, train_loss_batch, summary_train_loss = sess.run(
                        [update_ops, model.optimizer, model.loss, summary_merged],
                        feed_dict={handle: training_handle, is_training: True})
                    writer_train.add_summary(summary_train_loss, global_step.eval())

                    total_time_batch_train = datetime.now() - ini_time_batch_train

                    print(
                        'epoch: {}/{},   batch: {}/{},   global_step: {}/{},   train_loss: {},   batch_time: {}'.format(
                            str(epoch_count + 1).rjust(epochs_digits, '0'), epochs,
                            str(batch_count + 1).rjust(batches_digits, '0'), batches_train,
                            str(global_step.eval()).rjust(global_steps_digits, '0'), global_steps,
                            str(round(train_loss_batch * 100, 6)).ljust(max_digits_loss, '0'),
                            total_time_batch_train))

                    if global_step.eval() % val_freq == 0:

                        loss_avg_val = 0

                        sess.run(val_iterator.initializer)

                        ini_time_val = datetime.now()

                        while True:

                            try:

                                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

                                _, val_loss_batch, summary_eval_loss = sess.run([update_ops, model.loss, summary_merged],
                                                                                 feed_dict={handle: validation_handle,
                                                                                                    is_training: True})
                                writer_val.add_summary(summary_eval_loss, global_step.eval())

                                loss_avg_val += val_loss_batch


                            except tf.errors.OutOfRangeError:
                                break
                            except Exception as inst:
                                print('error val: ' + str(sys.exc_info()[0]))
                                break

                        total_time_val = datetime.now() - ini_time_val

                        print('\nval_loss: {},   val_time: {}'.format(
                            str(round((loss_avg_val / batches_val) * 100, 6)).ljust(max_digits_loss, '0'),
                            total_time_val) + '\n')

                    if global_step.eval() % ckpt_freq == 0:

                        # save model and weights
                        save_path = saver.save(sess=sess,
                                               save_path=out_path + slash + train_path + slash + 'checkpoints' + slash + "model.ckpt",
                                               write_meta_graph=False, global_step=global_step)

                        update_train_cfg_file(out_path + slash + train_path, dataset_path, 'model_1', epochs,
                                              batch_size, learning_rate, decay_rate, ckpt_freq, ckpt_max_to_keep,
                                              val_freq, max_digits_loss, cpu_count, resume_train,
                                              datetime.now() - ini_time_train, False)

                    batch_count += 1
                    if batch_count == batches_train:
                        batch_count = 0

                        epoch_count += 1
                        if epoch_count == epochs:
                            break

                except tf.errors.OutOfRangeError:
                    break
                except Exception as inst:
                    print('error train: ' + str(sys.exc_info()[0]))
                    break

            total_time_train = datetime.now() - ini_time_train

            update_train_cfg_file(out_path + slash + train_path, dataset_path, 'model_1', epochs, batch_size,
                                  learning_rate, decay_rate, ckpt_freq, ckpt_max_to_keep, val_freq, max_digits_loss,
                                  cpu_count, resume_train, total_time_train, False)

            print('\n' + str(total_time_train))

            writer_train.close()
            writer_val.close()