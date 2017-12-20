import tensorflow as tf
import numpy as np
import math


from os import listdir, path
from sklearn.metrics import roc_auc_score

from enhancersdata import EnhancersData
from sacred import Experiment
from sacred.observers import FileStorageObserver

ex = Experiment()
ex_log_dir = 'brain_logs'
data_dir = '/cs/grad/pazbu/paz/dev/projects/dnanet-v3/data/brain'

ex.observers.append(FileStorageObserver.create(ex_log_dir))


@ex.config
def general_config():
    general_cfg = {
                    "seq_length": 1000,
                    "num_outs": 16,
                    "batch_size": 64,
                    "num_runs": 30,
                    "num_epochs": 10
                    }

@ex.config
def cnn_config():
    conv1_cfg = {"num_filters": 32, "filter_size": [4, 7]}
    conv2_cfg = {"num_filters": 64, "filter_size": [1, 5]}
    conv3_cfg = {"num_filters": 64, "filter_size": [1, 5]}
    pool1_cfg = {"kernel_size": 5, "stride": 5}
    pool2_cfg = {"kernel_size": 5, "stride": 5}
    pool3_cfg = {"kernel_size": 5, "stride": 5}
    dense1_cfg = {"size": 100}
    dense2_cfg = {"size": 50}
    dropout_keep_prob = 0.5


def conv_rev_max(name_scope, input_tensor, num_filters, kernel_size=[5, 5]):
    """Returns a convolutions layer"""
    input_shape = input_tensor.get_shape().as_list()
    input_channels = input_shape[-1]
    with tf.name_scope(name_scope):
        initer = tf.truncated_normal(kernel_size + [input_channels, num_filters],
                                     stddev=math.sqrt(2 / float(input_channels)))
        weights = tf.Variable(initer, name='weights')
        num_kernels = weights.get_shape()[3]
        biases = tf.Variable(tf.zeros([num_kernels]), name='biases')

        forward = tf.nn.conv2d(input_tensor, weights, strides=[1, 1, 1, 1], padding='VALID', name="forward")
        # forward = conv2dSparseMax(input_tensor, weights, padding=padding, name="forward")
        weights_rev = tf.reverse(weights, [0, 1])

        backward = tf.nn.conv2d(input_tensor, weights_rev, strides=[1, 1, 1, 1], padding='VALID', name="backward")
        # backward = conv2dSparseMax(input_tensor, weights_rev, padding=padding, name="backward")
        max_conv = tf.maximum(forward, backward) + biases

        return tf.nn.relu(max_conv)

# model
@ex.capture
def CNN(x, general_cfg, conv1_cfg, conv2_cfg, conv3_cfg, pool1_cfg, pool2_cfg, pool3_cfg, dense1_cfg, dense2_cfg, dropout_keep_prob):
    x_seq = tf.reshape(x, [-1, 4, general_cfg["seq_length"], 1])
    conv1 = tf.layers.conv2d(inputs=x_seq, filters=conv1_cfg["num_filters"], kernel_size=conv1_cfg["filter_size"],
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(), name="conv1")

    # conv1 = conv_rev_max("conv1", x_seq, num_filters=conv1_cfg["num_filters"], kernel_size=conv1_cfg["filter_size"])

    max_pool1 = tf.nn.max_pool(conv1, ksize=[1, 1, pool1_cfg["kernel_size"], 1],
                               strides=[1, 1, pool1_cfg["stride"], 1], padding='VALID')
    # conv2
    conv2 = tf.layers.conv2d(inputs=max_pool1, filters=conv2_cfg["num_filters"], kernel_size=conv2_cfg["filter_size"],
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(), name="conv2")
    max_pool2 = tf.nn.max_pool(conv2, ksize=[1, 1, pool2_cfg["kernel_size"], 1],
                               strides=[1, 1, pool2_cfg["stride"], 1], padding='VALID')
    # conv3
    conv3 = tf.layers.conv2d(inputs=max_pool2, filters=conv3_cfg["num_filters"], kernel_size=conv3_cfg["filter_size"],
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(), name="conv3")
    max_pool3 = tf.nn.max_pool(conv3, ksize=[1, 1, pool3_cfg["kernel_size"], 1],
                                      strides=[1, 1, pool3_cfg["stride"], 1], padding='VALID')
    # flatten
    # bn1 = tf.layers.batch_normalization(max_pool3)

    conv3_flat = tf.contrib.layers.flatten(max_pool3)
    # two affine (fully-connected) layers with dropout in between

    dense1 = tf.layers.dense(conv3_flat, dense1_cfg["size"], activation=tf.nn.relu, name="dense1")
    # dropout = tf.nn.dropout(dense1, keep_prob=dropout_keep_prob)

    dense2 = tf.layers.dense(dense1, dense2_cfg["size"], activation=tf.nn.relu, name="dense2")
    # output layer
    dense_out = tf.layers.dense(dense2, general_cfg["num_outs"], activation=None, name="dense_out")
    # we're returning the unscaled output so we can use the safe: tf.nn.softmax_cross_entropy_with_logits
    return dense_out


def log_files(dir_path):
    for filename in listdir(dir_path):
        ex.add_resource(path.join(dir_path, filename))


def batched_eval(op, batch_iter, x, y_, keep_prob):
    vals = []
    for x_batch, y_batch in batch_iter:
        vals.extend(op.eval(feed_dict={x: x_batch, y_: y_batch, keep_prob: 1.0}))
    return vals


def calc_auc_rocs(y_true, y_pred):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    num_outs = y_true.shape[1]
    auc_rocs = []
    for idx in range(num_outs):
        y_true_feat = (y_true[:, idx]).astype(np.uint8)
        auc_rocs.append(roc_auc_score(y_true_feat, y_pred[:, idx]))
    return auc_rocs

@ex.automain
def run_experiment(general_cfg, dropout_keep_prob, seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)
    ds = EnhancersData(data_dir)
    log_files(data_dir)

    x = tf.placeholder(tf.float32, shape=[None, 4, general_cfg["seq_length"]], name="x")
    y_ = tf.placeholder(tf.float32, shape=[None, general_cfg["num_outs"]], name="y_")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    y_conv = CNN(x, dropout_keep_prob=keep_prob)

    # loss_priors = np.load('prior.npy')
    # cross_entropy_no_reduce = tf.losses.sigmoid_cross_entropy(multi_class_labels=y_, logits=y_conv, reduction=tf.losses.Reduction.NONE)
    # cross_entropy_no_reduce = tf.matmul(loss_priors, tf.transpose(cross_entropy_no_reduce))
    # cross_entropy = tf.reduce_sum(cross_entropy_no_reduce)

    cross_entropy = tf.losses.sigmoid_cross_entropy(multi_class_labels=y_, logits=y_conv)


    # attach update ops used for the batch normalization
    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # with tf.control_dependencies(update_ops):
    #     train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

    y_pred_sig = tf.sigmoid(y_conv, name="sigmoid_out")
    correct_prediction = tf.equal(y_, tf.round(y_pred_sig))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    num_runs = general_cfg["num_runs"]
    num_epochs = general_cfg["num_epochs"]
    mini_batch_size = general_cfg["batch_size"]

    for var in tf.trainable_variables():
        tf.summary.histogram(var.name, var)
    merged_summary_op = tf.summary.merge_all()

    saver = tf.train.Saver()
    best_val_acc = 0
    ex_id = ex.current_run._id
    class_weights = np.sum(ds.test.labels, axis=0) / np.sum(ds.test.labels)
    with tf.Session() as sess:
        for run_idx in range(num_runs):
            chkp_dir = path.join(ex_log_dir, str(ex_id), "run" + str(run_idx))
            ds.reset()
            sess.run(tf.global_variables_initializer())
            summary_writer = tf.summary.FileWriter("/cs/grad/pazbu/paz/dev/projects/dnanet-v3/summaries", graph=sess.graph)
            current_step = 0
            while ds.train.epochs_completed < num_epochs:
                current_step += 1
                batch = ds.train.next_batch(mini_batch_size)
                summary_str,_ , l = sess.run([merged_summary_op, train_step, cross_entropy],
                                             feed_dict={x: batch[0], y_: batch[1], keep_prob: dropout_keep_prob})
                l_sum = np.sum(l)
                # skip to next epoch if loss is very small:
                if l_sum < 0.001:
                    print('loss < 0.001. skipping to next epoch...')
                    curr_epoch = ds.train.epochs_completed
                    while curr_epoch == ds.train.epochs_completed:
                        ds.train.next_batch(mini_batch_size)
                        current_step += 1
                    continue

                # summary_writer.add_summary(summary_str, current_step)
                if current_step % 200 == 0:
                    train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                    y_vals = batched_eval(y_pred_sig, ds.validation.single_pass_batch_iter(1000), x, y_, keep_prob)
                    valid_accuracy = accuracy.eval(feed_dict={y_: ds.validation.labels, y_pred_sig: y_vals})
                    if best_val_acc < valid_accuracy:
                        saver.save(sess, save_path=path.join(chkp_dir, str(valid_accuracy)), global_step=current_step)
                        best_val_acc = valid_accuracy
                        ex.info["best_validation_accuracy"] = best_val_acc
                    print('run: %d, epoch: %d, iteration: %d, train accuracy: %g, validation accuracy: %g, loss: %g' %
                          (run_idx, ds.train.epochs_completed, current_step, train_accuracy, valid_accuracy, l_sum))

            y_pred = batched_eval(y_pred_sig, ds.test.single_pass_batch_iter(1000), x, y_, keep_prob)
            test_accuracy = accuracy.eval(feed_dict={y_: ds.test.labels, y_pred_sig: y_pred})
            print('test accuracy: ', test_accuracy)
            auc_rocs = calc_auc_rocs(ds.test.labels, y_pred)
            for i, auc_roc in enumerate(auc_rocs):
                print('roc-auc ' + str(i) + ":", auc_roc)
            auc_rocs_weighted_sum = np.sum(np.multiply(auc_rocs, class_weights))
            print('roc-auc weighted sum:', auc_rocs_weighted_sum)
            ex.info["final_auc_rocs_run" + str(run_idx)] = auc_rocs
