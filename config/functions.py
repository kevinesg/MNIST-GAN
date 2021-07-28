from config import config as C
import tensorflow as tf

def parse_values(line):
    defs = [tf.constant([], dtype=tf.float32)] + [0.] * 784
    values = tf.io.decode_csv(line, record_defaults=defs)
    X = tf.stack(values[1:])
    X = tf.reshape(X, (28, 28, 1))
    y = tf.stack(values[:1])
    y = tf.one_hot(tf.cast(y, tf.int32), depth=10)
    y = tf.reshape(y, (10,))
        
    return X, y

def csv_reader_dataset(filepaths, n_readers=10, n_threads=12):
    dataset = tf.data.Dataset.list_files(filepaths)
    dataset = dataset.interleave(
        lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
        cycle_length=n_readers, num_parallel_calls=n_readers)
    dataset = dataset.map(parse_values, num_parallel_calls=n_threads)
    dataset = dataset.shuffle(10).repeat(1)
    dataset = dataset.batch(C.BATCH_SIZE).prefetch(1)

    return dataset