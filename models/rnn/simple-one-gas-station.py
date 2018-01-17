import os
import pandas as pd
import tensorflow as tf
from sklearn import model_selection
from time import time
import argparse

GAS_STATIONS_PATH = os.path.join('..','..', 'data', 'raw', 'input_data', 'Eingabedaten', 'Tankstellen.csv')
GAS_PRICE_PATH = os.path.join('..','..', 'data', 'raw', 'input_data', 'Eingabedaten', 'Benzinpreise')

gas_stations_df = pd.read_csv(GAS_STATIONS_PATH, sep=';', names=['id', 'Name', 'Company', 'Street', 'House_Number', 'Postalcode', 'City', 'Lat', 'Long'],index_col='id')

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gas_station",
                    type=int,
                    help="Number of gas station to learn on")
parser.add_argument("--batch_size",
                    type=int,
                    default=128,
                    help="Batch size")
parser.add_argument("--layers",
                    type=int,
                    default=1,
                    help="Number of LSTM layers")
parser.add_argument("--lstm_size",
                    type=int,
                    help="Size of LSTM hidden state")
parser.add_argument( "--length_of_sequence",
                    default=200,
                    type=int,
                    help="Length of prediction sequences")
parser.add_argument( "--future_prediction",
                    default=1,
                    type=int,
                    help="Choose how much into the future should be predicted, '1' means the next value")
parser.add_argument( "--dense_hidden_units",
                    type=int,
                    help="Number of hidden units in dense layer before output layer")
parser.add_argument( "-m", "--memory",
                    default=1,
                    type=float,
                    help="Percentage of VRAM for this process, written as 0.")
parser.add_argument( "-n", "--name",
                    type=str,
                    help="Name of training")
args = parser.parse_args()
    parser.add_argument("--chkpt-path",
                        type=str,
                        help="Path for saving the checkpoints")
    parser.add_argument("--log-path",
                        type=str,
                        help="Path for saving the logs")

gas_station_id = args.gas_station

gas_station = pd.read_csv(os.path.join(GAS_PRICE_PATH,'{}.csv'.format(gas_station_id)), names=['Timestamp', 'Price'],  index_col='Timestamp',parse_dates=['Timestamp'],sep=';')
gas_station_resampled = gas_station.resample('1T').bfill()

lstm_size = args.lstm_size
number_of_layers = args.layers
batch_size = args.batch_size
length_of_each_sequence = args.length_of_sequence
dense_hidden_units = args.dense_hidden_units

features = []
sequence = []
for value in gas_station_resampled.values:
    if len(sequence) < length_of_each_sequence - 1:
        sequence.append([value[0]])
    elif len(sequence) < length_of_each_sequence:
        sequence.append([value[0]])
        assert len(sequence) == length_of_each_sequence
        features.append(sequence)
        sequence = []
    else:
        raise NotImplementedError()
print("Number of training samples: " + str(len(features)))

X_train, X_test, _, _ = model_selection.train_test_split(features, features, test_size=0.4, shuffle=False,
                                                         random_state=42)
# Train is here validation set
X_val, X_test, _, _ = model_selection.train_test_split(X_test, X_test, test_size=0.75, shuffle=False, random_state=42)


features_placeholder = tf.placeholder(tf.float32, [None, length_of_each_sequence, 1], name='features_placeholder')

seed = tf.placeholder(tf.int64, shape=[])
dataset = tf.data.Dataset.from_tensor_slices(X_train)
dataset = dataset.shuffle(buffer_size=batch_size*10, seed=seed)
dataset = dataset.batch(batch_size)
iterator = dataset.make_initializable_iterator()
next_elem = iterator.get_next()

def lstm_cell():
  return tf.nn.rnn_cell.LSTMCell(lstm_size)

stacked_lstm = [lstm_cell() for _ in range(number_of_layers)]

# create a RNN cell composed sequentially of a number of RNNCells
multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(stacked_lstm)

# 'outputs' is a tensor of shape [batch_size, max_time, 256]
# 'state' is a N-tuple where N is the number of LSTMCells containing a
# tf.contrib.rnn.LSTMStateTuple for each cell
outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                   inputs=features_placeholder[:,:-args.future_prediction],
                                   dtype=tf.float32)


#final_state = state[-1]

dense = tf.layers.dense(outputs[:,-1], dense_hidden_units, activation=tf.nn.relu, name='dense', kernel_initializer=tf.truncated_normal_initializer(mean=0.001, stddev=0.1), bias_initializer=tf.truncated_normal_initializer(mean=0.01, stddev=0.1))
y_ = tf.layers.dense(dense, 1, activation=None, name='y_', kernel_initializer=tf.truncated_normal_initializer(mean=0.001, stddev=0.1), bias_initializer=tf.truncated_normal_initializer(mean=0.01, stddev=0.1))

learning_rate =1e-4
optimizer = tf.train.AdamOptimizer(learning_rate)

predicted_labels = y_
# TODO: Are we sure with this?
true_labels = features_placeholder[:,-1]

assert predicted_labels.shape[1] == true_labels.shape[1], str(predicted_labels.shape[1]) + ' ' + str(true_labels.shape[1])

cost = tf.reduce_sum(tf.squared_difference(predicted_labels, true_labels), name='cost')
tf.summary.scalar(name='cost', tensor=cost)
grads_and_vars = optimizer.compute_gradients(loss=cost)  # list of (gradient, variable) tuples
train_step = optimizer.apply_gradients(grads_and_vars)

mse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(true_labels, predicted_labels))))
tf.summary.scalar(tensor=mse, name='MSE')

init = tf.global_variables_initializer()

merged = tf.summary.merge_all()

# Add ops to save and restore all the variables.
saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.10)
name_of_training = args.name
log_path = '/home/informaticup2018/sebastian/checkpoints/seq2seq/' + name_of_training
chkpt_path = '/home/informaticup2018/sebastian/checkpoints/seq2seq/' + name_of_training

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.memory)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    train_writer = tf.summary.FileWriter(log_path + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(log_path + '/test')
    val_writer = tf.summary.FileWriter(log_path + '/val')
    sess.run(iterator.initializer, feed_dict={seed: int(time())})
    sess.run(init)

    training_cycles = 0
    while True:
        try:
            elem = sess.run(next_elem)
        except tf.errors.OutOfRangeError:
            # TODO: Save trained modell
            sess.run(iterator.initializer, feed_dict={seed: int(time())})
            elem = sess.run(next_elem)
        if training_cycles == 0:
            summary = sess.run(merged, feed_dict={features_placeholder: elem})
            train_writer.add_summary(summary, training_cycles)
        if training_cycles % 50 == 0:
            summary_test = sess.run(merged, feed_dict={features_placeholder: X_test})
            test_writer.add_summary(summary_test, training_cycles)

            summary_val = sess.run(merged, feed_dict={features_placeholder: X_val})
            val_writer.add_summary(summary_val, training_cycles)
        summary, _ = sess.run([merged, train_step], feed_dict={features_placeholder: elem})
        if training_cycles % 5000 == 0:
            if not os.path.exists(chkpt_path + '/'):
                os.makedirs(chkpt_path + '/')
            save_path = saver.save(sess, chkpt_path + '-' + str(training_cycles) + '.ckpt')

        training_cycles += 1
        train_writer.add_summary(summary, training_cycles)