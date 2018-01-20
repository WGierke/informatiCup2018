import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import model_selection
from time import time
import argparse

GAS_STATIONS_PATH = os.path.join('..','..', 'data', 'raw', 'input_data', 'Eingabedaten', 'Tankstellen.csv')
GAS_PRICE_PATH = os.path.join('..','..', 'data', 'raw', 'input_data', 'Eingabedaten', 'Benzinpreise')

SIMPLE_RNN_MODEL = 'resampled'
EVENT_RNN_MODEL = 'event'

gas_stations_df = pd.read_csv(GAS_STATIONS_PATH, sep=';', names=['id', 'Name', 'Company', 'Street', 'House_Number', 'Postalcode', 'City', 'Lat', 'Long'],index_col='id')

def parse_arguments():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        type=str,
                        default=SIMPLE_RNN_MODEL,
                        help="Name of model, can be one of {}".format(str([SIMPLE_RNN_MODEL,EVENT_RNN_MODEL])))
    parser.add_argument("-g", "--gas_station",
                        type=int,
                        required=True,
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
                        default=80,
                        help="Size of LSTM hidden state")
    parser.add_argument("--length_of_sequence",
                        default=200,
                        type=int,
                        help="Length of prediction sequences")
    parser.add_argument("--future_prediction",
                        default=1,
                        type=int,
                        help="Choose how much into the future should be predicted, '1' means the next value")
    parser.add_argument("--dense_hidden_units",
                        type=int,
                        default=100,
                        help="Number of hidden units in dense layer before output layer")
    parser.add_argument("-m", "--memory",
                        default=1,
                        type=float,
                        help="Percentage of VRAM for this process, written as 0.")
    parser.add_argument("-n", "--name",
                        type=str,
                        required=True,
                        help="Name of training")
    parser.add_argument("--chkpt_path",
                        type=str,
                        required=True,
                        help="Path for saving the checkpoints")
    parser.add_argument("--log_path",
                        type=str,
                        required=True,
                        help="Path for saving the logs")
    parser.add_argument("--resampling",
                        type=str,
                        default='1D',
                        help="Resampling of data frame")
    parser.add_argument('--additional_gas_stations',
                        nargs='+',
                        type=int,
                        help="Additional gas stations which are feed into the model",
                        default=[])
    parser.add_argument("--sequence_stride",
                        type=int,
                        help="At every sequence stride a new sequence is extracted for the training and test data.",
                        default=1)
    return parser.parse_args()


def train(X_test, X_val, chkpt_path, features_placeholder, gpu_options, init, iterator, log_path, merged, next_elem,
          saver, seed, train_step, name_of_training):
    log_path = log_path + "/" + name_of_training
    chkpt_path = chkpt_path + "/" + name_of_training + "/"
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        if tf.train.latest_checkpoint(chkpt_path) is not None:
            saver.restore(sess, tf.train.latest_checkpoint(chkpt_path))
            print("Restored model from " + tf.train.latest_checkpoint(chkpt_path))
        else:
            print("No model was loaded")

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
                if not os.path.exists(chkpt_path):
                    os.makedirs(chkpt_path)
                save_path = saver.save(sess, chkpt_path + name_of_training + "-" + str(training_cycles) + '.ckpt')

            training_cycles += 1
            train_writer.add_summary(summary, training_cycles)


def define_simple_rnn_model(dense_hidden_units, future_prediction, length_of_each_sequence, number_of_layers, lstm_size, number_of_additional_gas_stations=0):

    features_placeholder = tf.placeholder(tf.float32, [None, length_of_each_sequence, 1 + number_of_additional_gas_stations], name='features_placeholder')
    seed = tf.placeholder(tf.int64, shape=[])

    def lstm_cell():
        return tf.nn.rnn_cell.LSTMCell(lstm_size)

    stacked_lstm = [lstm_cell() for _ in range(number_of_layers)]
    # create a RNN cell composed sequentially of a number of RNNCells
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(stacked_lstm)
    # 'outputs' is a tensor of shape [batch_size, max_time, 256]
    # 'state' is a N-tuple where N is the number of LSTMCells containing a
    # tf.contrib.rnn.LSTMStateTuple for each cell
    outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                       inputs=features_placeholder[:, :-future_prediction],
                                       dtype=tf.float32)
    # final_state = state[-1]
    dense = tf.layers.dense(outputs[:, -1], dense_hidden_units, activation=tf.nn.relu, name='dense',
                            kernel_initializer=tf.truncated_normal_initializer(mean=0.001, stddev=0.1),
                            bias_initializer=tf.truncated_normal_initializer(mean=0.01, stddev=0.1))
    y_ = tf.layers.dense(dense,  1 + number_of_additional_gas_stations, activation=None, name='y_',
                         kernel_initializer=tf.truncated_normal_initializer(mean=0.001, stddev=0.1),
                         bias_initializer=tf.truncated_normal_initializer(mean=0.01, stddev=0.1))
    learning_rate = 1e-4
    optimizer = tf.train.AdamOptimizer(learning_rate)
    predicted_labels = y_
    true_labels = features_placeholder[:, -1]
    assert predicted_labels.shape[1] == true_labels.shape[1], str(predicted_labels.shape[1]) + ' ' + str(
        true_labels.shape[1])
    cost = tf.reduce_sum(tf.squared_difference(predicted_labels, true_labels), name='cost')
    tf.summary.scalar(name='cost', tensor=cost)
    grads_and_vars = optimizer.compute_gradients(loss=cost)  # list of (gradient, variable) tuples
    train_step = optimizer.apply_gradients(grads_and_vars)
    mse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(true_labels, predicted_labels))))
    tf.summary.scalar(tensor=mse, name='MSE')
    return features_placeholder, seed, train_step

def define_event_rnn_model(dense_hidden_units, length_of_each_sequence, number_of_layers, lstm_size):
    """
    Placeholder expects first the price and the time to the next change as input.

    :param dense_hidden_units:
    :param future_prediction:
    :param length_of_each_sequence:
    :param number_of_layers:
    :param lstm_size:
    :return:
    """
    # [price, time_to_last_event]
    features_placeholder = tf.placeholder(tf.float32, [None, length_of_each_sequence, 2], name='features_placeholder')
    seed = tf.placeholder(tf.int64, shape=[])

    def lstm_cell():
        return tf.nn.rnn_cell.LSTMCell(lstm_size)

    stacked_lstm = [lstm_cell() for _ in range(number_of_layers)]
    # create a RNN cell composed sequentially of a number of RNNCells
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(stacked_lstm)
    # 'outputs' is a tensor of shape [batch_size, max_time, 256]
    # 'state' is a N-tuple where N is the number of LSTMCells containing a
    # tf.contrib.rnn.LSTMStateTuple for each cell
    outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                       inputs=features_placeholder[:, :-1],
                                       dtype=tf.float32)
    # final_state = state[-1]
    dense = tf.layers.dense(outputs[:, -1], dense_hidden_units, activation=tf.nn.relu, name='dense',
                            kernel_initializer=tf.truncated_normal_initializer(mean=0.001, stddev=0.1),
                            bias_initializer=tf.truncated_normal_initializer(mean=0.01, stddev=0.1))

    dense_time_of_change = tf.layers.dense(dense, dense_hidden_units, activation=tf.nn.relu, name='dense_time_of_change',
                            kernel_initializer=tf.truncated_normal_initializer(mean=0.001, stddev=0.1),
                            bias_initializer=tf.truncated_normal_initializer(mean=0.01, stddev=0.1))
    y_time_of_change = tf.layers.dense(dense_time_of_change, 1, activation=None, name='y_time_of_change',
                         kernel_initializer=tf.truncated_normal_initializer(mean=0.001, stddev=0.1),
                         bias_initializer=tf.truncated_normal_initializer(mean=0.01, stddev=0.1))

    dense_price = tf.layers.dense(dense, dense_hidden_units, activation=tf.nn.relu,
                                           name='dense_price',
                                           kernel_initializer=tf.truncated_normal_initializer(mean=0.001, stddev=0.1),
                                           bias_initializer=tf.truncated_normal_initializer(mean=0.01, stddev=0.1))
    y_price = tf.layers.dense(dense_price, 1, activation=None, name='y_dense_price',
                                       kernel_initializer=tf.truncated_normal_initializer(mean=0.001, stddev=0.1),
                                       bias_initializer=tf.truncated_normal_initializer(mean=0.01, stddev=0.1))


    learning_rate = 1e-4
    optimizer = tf.train.AdamOptimizer(learning_rate)
    predicted_labels = tf.concat([y_price, y_time_of_change], 1)
    true_labels = features_placeholder[:, -1]
    assert predicted_labels.shape[1] == true_labels.shape[1], str(predicted_labels.shape[1]) + ' ' + str(
        true_labels.shape[1])
    # We could introduce linear coefficients for the cost function
    cost = tf.reduce_sum(tf.squared_difference(predicted_labels, true_labels), name='cost')
    tf.summary.scalar(name='cost', tensor=cost)
    grads_and_vars = optimizer.compute_gradients(loss=cost)  # list of (gradient, variable) tuples
    train_step = optimizer.apply_gradients(grads_and_vars)
    mse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(true_labels, predicted_labels))))
    tf.summary.scalar(tensor=mse, name='MSE')
    return features_placeholder, seed, train_step


def calculate_samples(gas_station_resampled, length_of_each_sequence):
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
    return features

def combine_channels_and_generate_sequences(gas_station_series, sequence_length,sequence_stride=1):
    # input: list of gas stations [[t1,t2,t3,...],[b1,b2,v3...]]
    assert sequence_length >= 1, "Must use a positive sequence length, used {}.".format(sequence_length)
    assert sequence_stride > 0, "Must use a positive stride, used {}.".format(sequence_stride)

    features = []
    gas_stations = np.array(gas_station_series)
    available_timesteps = gas_stations.T.shape[0]
    assert sequence_length <= available_timesteps, "Sequence length must be longer than the available time steps in the sequence."
    assert sequence_stride < available_timesteps, "Stride must be smaller than the length of the time series"

    for i in range(available_timesteps - sequence_length)[::sequence_stride]:
        features.append(gas_stations.T[i:i + sequence_length])
    return np.array(features)


def build_dataset(X_train, batch_size, seed):
    dataset = tf.data.Dataset.from_tensor_slices(X_train)
    dataset = dataset.shuffle(seed=seed, buffer_size=len(X_train)+1)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    next_elem = iterator.get_next()
    return iterator, next_elem

def load_station(gas_station_id):
    p = os.path.join(GAS_PRICE_PATH,'{}.csv'.format(gas_station_id))
    try:
        return pd.read_csv(p, names=['Timestamp', 'Price'],  index_col='Timestamp',parse_dates=['Timestamp'],sep=';')
    except FileNotFoundError:
        raise ValueError('You tried to retrieve the history for gas station with id {}, but file {} was no found.'.format(gas_station_id,p))


def build_df_for_all_gas_stations():
    raise NotImplementedError()

    min_start_date = gas_stations_df.min
    max_end_date = gas_stations_df.max
    for gas_station_row in gas_stations_df.iterrows():
        try:
            gas_station = pd.read_csv(os.path.join(GAS_PRICE_PATH, '{}.csv'.format(gas_station_row)),
                                      names=['Timestamp', 'Price'], index_col='Timestamp', parse_dates=['Timestamp'],
                                      sep=';')
            if min_start_date > gas_station.index.min:
                min_start_date = gas_station.index.min
            if max_end_date < gas_station.index.max:
                max_end_date = gas_station.index.max
        except FileNotFoundError:
            pass
    gas_stations = []

def main():
    args = parse_arguments()
    model = args.model
    gas_station_id = args.gas_station
    gas_station_df = load_station(gas_station_id)

    # overall model params
    lstm_size = args.lstm_size
    number_of_layers = args.layers
    batch_size = args.batch_size
    dense_hidden_units = args.dense_hidden_units
    length_of_each_sequence = args.length_of_sequence
    frequency = args.resampling
    sequence_stride = args.sequence_stride

    if model == EVENT_RNN_MODEL:

        if len(args.additional_gas_stations) > 0:
            raise NotImplementedError("Close gas stations not available for this model.")

        if args.future_prediction != 1:
            raise NotImplementedError("This model will only predict one next event.")

        deltas = pd.Series(gas_station_df.index[1:] - gas_station_df.index[:-1])
        deltas_in_minutes = deltas.apply(lambda x: x.round(frequency).total_seconds() / pd.Timedelta('1D').total_seconds())
        event_deltas = np.append([0], deltas_in_minutes.values.flatten())
        price = gas_station_df.values.flatten()
        features = combine_channels_and_generate_sequences([price, event_deltas],sequence_length=length_of_each_sequence,sequence_stride=sequence_stride)
        print("Number of training samples: {}".format(features.shape[0]))
        features_placeholder, seed, train_step = define_event_rnn_model(dense_hidden_units, length_of_each_sequence, number_of_layers, lstm_size)

    elif model == SIMPLE_RNN_MODEL:

        future_prediction = args.future_prediction
        additional_gas_station_ids = args.additional_gas_stations

        gas_station_resampled = gas_station_df.resample(frequency).bfill()

        additional_gas_stations_resampled = []
        for id in additional_gas_station_ids:
            additional_df = load_station(id).resample(frequency).bfill()
            additional_df_aligned = gas_station_resampled.align(additional_df)[1].fillna(0)[gas_station_resampled.index.min():gas_station_resampled.index.max()]
            additional_gas_stations_resampled.append(additional_df_aligned.values.flatten())

        print("Using {} as supplementary gas stations".format(', '.join(map(str,additional_gas_station_ids))))

        features = combine_channels_and_generate_sequences([gas_station_resampled.values.flatten()] + additional_gas_stations_resampled, length_of_each_sequence,sequence_stride=sequence_stride)
        print("Number of training samples: {}".format(features.shape[0]))

        features_placeholder, seed, train_step = define_simple_rnn_model(dense_hidden_units,
                                                                         future_prediction=future_prediction,
                                                                         length_of_each_sequence=length_of_each_sequence,
                                                                         number_of_layers=number_of_layers,
                                                                         lstm_size=lstm_size,
                                                                         number_of_additional_gas_stations=len(additional_gas_station_ids))
    else:
        raise NotImplementedError("The model you wished for is not available, go implement it yourself.")

    X_train, X_intermediate, _, _ = model_selection.train_test_split(features, features, test_size=0.4, shuffle=False,
                                                                    random_state=42)
    # Train is here validation set
    X_val, X_test, _, _ = model_selection.train_test_split(X_intermediate, X_intermediate, test_size=0.75, shuffle=False, random_state=42)

    iterator, next_elem = build_dataset(X_train, batch_size, seed)

    init = tf.global_variables_initializer()

    merged = tf.summary.merge_all()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.10)
    name_of_training = args.name
    log_path = args.log_path
    chkpt_path = args.chkpt_path

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.memory)

    train(X_test, X_val, chkpt_path, features_placeholder, gpu_options, init, iterator, log_path, merged, next_elem,
          saver, seed, train_step, name_of_training)

if __name__ == "__main__":
    main()