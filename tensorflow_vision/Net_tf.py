import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, optimizers, datasets, Sequential, losses


class NetWork(tf.keras.Model):

    def __init__(self, total_words, embedding_len, max_review_len, units, num_class):
        super(NetWork, self).__init__()

        self.embedding = layers.Embedding(total_words, embedding_len, input_length=max_review_len)
        self.lstm1 = layers.LSTM(units, dropout=0.5, return_sequences=True)
        self.lstm2 = layers.LSTM(units, dropout=0.5)

        self.dense1 = layers.Dense(units)
        self.dropout1 = layers.Dropout(rate=0.5)
        self.relu = layers.ReLU()
        self.dense2 = layers.Dense(num_class)
        self.softmax = layers.Softmax()

    def call(self, inputs):

        x = self.embedding(inputs)
        x = self.lstm1(x)
        x = self.lstm2(x)

        x = self.dense1(x)
        x = self.relu(self.dropout1(x))
        x = self.dense2(x)

        y = self.softmax(x)

        return y

    def evaluate(self, inputs_):

        x_ = self.embedding(inputs_, training=False)
        x_ = self.lstm1(x_, training=False)
        x_ = self.lstm2(x_, training=False)

        x_ = self.dense1(x_)
        x_ = self.relu(self.dropout1(x_, training=False))
        x_ = self.dense2(x_)

        y_ = self.softmax(x_)

        return y_


if __name__ == '__main__':

    total_words = 152410 + 3
    max_review_len = 600
    embedding_len = 256
    units = 128
    num_class = 9

    input_ = tf.convert_to_tensor(np.random.randint(0, 152410, (64, 600)))
    label_ = tf.convert_to_tensor(np.random.randint(0, 10,  (64, )), dtype=tf.int32)
    label_ = tf.one_hot(label_, depth=num_class)

    print("input.shape", input_.shape)
    print("label_.shape", label_.shape)

    model = NetWork(total_words, embedding_len, max_review_len, units, num_class)
    model.build(input_shape=(64, 600))
    model.summary()
    loss_fn = losses.CategoricalCrossentropy()

    output_ = model(input_)
    loss = loss_fn(y_pred=output_, y_true=label_)

    print("output_.shape", output_.shape)
    print("loss", loss)



