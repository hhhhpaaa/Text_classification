import math
import os
import h5py
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from Net_tf import NetWork
from tensorflow.keras import layers, optimizers, callbacks, losses
from tensorflow.keras.optimizers import schedules


def get_data(path_, train_test_proportion, num_class):

    h5 = h5py.File(path_, 'r')
    data_x = h5['data_x'][:]
    data_y = h5['data_y'][:]
    label_weight = h5['label_weight'][:]
    h5.close()

    index_ = data_x.shape[0] - int(data_x.shape[0] / train_test_proportion)

    np.random.seed(1)
    np.random.shuffle(data_x)
    np.random.seed(1)
    np.random.shuffle(data_y)

    data_x = tf.convert_to_tensor(data_x, dtype=tf.float32)
    data_y = tf.convert_to_tensor(data_y, dtype=tf.int32)
    data_y = tf.one_hot(data_y, depth=num_class)

    train_dataset = tf.data.Dataset.from_tensor_slices((data_x[:index_], data_y[:index_]))
    test_dataset = tf.data.Dataset.from_tensor_slices((data_x[index_:], data_y[index_:]))

    train_dataset = train_dataset.shuffle(1000).batch(batch_size)
    test_dataset = test_dataset.shuffle(1000).batch(batch_size)

    train_shape = data_x[:index_].shape
    test_shape = data_x[index_:].shape

    return train_dataset, test_dataset, label_weight, train_shape, test_shape


def acc_fn(y_pred, y_true):

    y_true = tf.argmax(y_true, axis=1)
    y_pred = tf.argmax(y_pred, axis=1)

    acc_vector = tf.equal(y_true, y_pred)
    acc_vector = tf.cast(acc_vector, dtype=tf.float32)

    acc_ = tf.reduce_sum(acc_vector)

    return acc_


def train(epoch):
    global best_loss_train, best_acc_train
    loss_train = 0.0
    acc_train = 0.0
    for step, (input_, label_) in tqdm(enumerate(train_loader), total=len(train_loader)):
        with tf.GradientTape() as tape:
            output_ = model(input_)
            loss_ = loss_fn(y_pred=output_, y_true=label_)

            grads = tape.gradient(loss_, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        acc_ = acc_fn(y_pred=output_, y_true=label_)
        loss_train += loss_.numpy()
        acc_train += acc_.numpy()

    # Save checkpoint. best_train_loss
    if save_weights_loss_train and (loss_train / len(train_loader) < best_loss_train):
        print('Saving best train loss...')
        best_loss_train = loss_train / len(train_loader)
        a, b = math.modf(best_loss_train)
        model_path = os.path.join(save_weights_path,
                                  '{}train_ckpt_loss{:0>4d}{:0>4d}.pth'.format(epoch, int(b), int(a * 10000)))
        model.save_weights(model_path)

    # Save checkpoint. best_train_acc
    if save_weights_acc_train and (acc_train / train_shape[0] > best_acc_train):
        print('Saving best train acc...')
        best_acc_train = acc_train / train_shape[0]
        a, b = math.modf(best_acc_train * 100)
        model_path = os.path.join(save_weights_path,
                                  '{}train_ckpt_acc{:0>4d}{:0>4d}.pth'.format(epoch, int(b), int(a * 10000)))
        model.save_weights(model_path)

    # print('Train Loss: %.3f | Acc-1: %.3f |' % (loss_train / len(train_loader), acc_train / train_shape[0]))
    return loss_train / len(train_loader), acc_train / train_shape[0]


def test(epoch):
    global best_loss_test, best_acc_test
    loss_test = 0.0
    acc_test = 0.0
    for step, (input_, label_) in tqdm(enumerate(test_loader), total=len(test_loader)):
        output_ = model.evaluate(input_)
        loss_ = loss_fn(y_pred=output_, y_true=label_)
        acc_ = acc_fn(y_pred=output_, y_true=label_)

        loss_test += loss_.numpy()
        acc_test += acc_.numpy()

    # Save checkpoint. best_test_loss
    if save_weights_loss_test and (loss_test / len(test_loader) < best_loss_test):
        print('Saving best test loss...')
        best_loss_test = loss_test / len(test_loader)
        a, b = math.modf(best_loss_test)
        model_path = os.path.join(save_weights_path,
                                  '{}test_ckpt_loss{:0>4d}{:0>4d}.pth'.format(epoch, int(b), int(a * 10000)))
        model.save_weights(model_path)

    # Save checkpoint. best_test_acc
    if save_weights_acc_test and (acc_test / test_shape[0] > best_acc_test):
        print('Saving best test acc...')
        best_acc_test = acc_test / test_shape[0]
        a, b = math.modf(best_acc_test * 100)
        model_path = os.path.join(save_weights_path,
                                  '{}test_ckpt_acc{:0>4d}{:0>4d}.pth'.format(epoch, int(b), int(a * 10000)))
        model.save_weights(model_path)

    # print('Test Loss: %.3f | Acc-1: %.3f |' % (loss_test / len(test_loader), acc_test / test_shape[0]))
    return loss_test / len(test_loader), acc_test / test_shape[0]


def save_plot_result(dict_result):

    fig1 = plt.figure(figsize=(20, 8), dpi=80)
    plt.plot(range(len(dict_result["loss_train"])), dict_result["loss_train"], label='train_loss')
    plt.plot(range(len(dict_result["loss_test"])), dict_result["loss_test"], color='red', label='test_loss')
    plt.legend()
    plt.title('loss')
    fig1.savefig('loss.pdf')
    plt.show()

    fig2 = plt.figure(figsize=(20, 8), dpi=80)
    plt.plot(range(len(dict_result["acc_train"])), dict_result["acc_train"], label='train_acc')
    plt.plot(range(len(dict_result["acc_test"])), dict_result["acc_test"], color='red', label='test_acc')
    plt.legend()
    plt.title('acc')
    fig2.savefig('acc.pdf')
    plt.show()


if __name__ == '__main__':

    save_weights_loss_train = False
    save_weights_acc_train = False
    save_weights_acc_test = False
    save_weights_loss_test = False
    load_weights = False
    best_loss_train = 10.0
    best_loss_test = 10.0
    best_acc_train = 0.0
    best_acc_test = 0.0

    Epoch = 200
    batch_size = 64
    train_test_proportion = 10
    total_words = 152410 + 3
    embedding_len = 256
    max_review_len = 600
    units = 128
    num_class = 9

    save_weights_path = './checkpoint_LSTM'
    weight_path = './checkpoint_LSTM/ckpt_loss00000010.pth'
    path_data = r'data_train.h5'

    train_loader, test_loader, label_weight, train_shape, test_shape = get_data(path_data,
                                                                                train_test_proportion, num_class)

    model = NetWork(total_words=total_words, embedding_len=embedding_len,
                    max_review_len=max_review_len, num_class=num_class, units=units)
    if load_weights:
        model.load_weights(weight_path)
        print("loading weights success")
    model.build(input_shape=(64, 300))
    model.summary()
    loss_fn = losses.CategoricalCrossentropy()
    lr = schedules.ExponentialDecay(initial_learning_rate=0.1, decay_steps=1000, decay_rate=0.96, staircase=True)
    optimizer = optimizers.Adam(learning_rate=lr)
    # optimizer = optimizers.Adam(learning_rate=0.001)

    if save_weights_loss_train or save_weights_acc_train or save_weights_acc_test or save_weights_loss_test:
        if not os.path.isdir(save_weights_path):
            os.mkdir(save_weights_path)

    dict_save = {"Epoch": [], "loss_train": [], "acc_train": [], "loss_test": [], "acc_test": []}
    for epoch in range(Epoch):
        loss_train, acc_train = train(epoch)
        loss_test, acc_test = test(epoch)

        print('epoch: {}, train_loss: {:.4f}, train_acc: {:.4f}, test_loss: {:.4f}, test_acc: {:.4f}'.format(
            epoch, loss_train, acc_train, loss_test, acc_test))
        print('lr: {:.8f}'.format(optimizer._decayed_lr(tf.float32).numpy()))

        dict_save["Epoch"].append(epoch)
        dict_save["loss_train"].append(loss_train)
        dict_save["acc_train"].append(acc_train)
        dict_save["loss_test"].append(loss_test)
        dict_save["acc_test"].append(acc_test)

        if acc_train > 0.8 and acc_test > 0.8:
            a1, b1 = math.modf(acc_train * 100)
            a2, b2 = math.modf(acc_test * 100)
            if not os.path.isdir(save_weights_path):
                os.mkdir(save_weights_path)
            model_path = os.path.join(save_weights_path,
                                      '{}train_acc{:0>4d}{:0>4d}test_acc{:0>4d}{:0>4d}.pth'.format(
                                          epoch, int(b1), int(a1 * 10000), int(b2), int(a2 * 10000)))
            model.save_weights(model_path)
            print('save weights success')

    save_plot_result(dict_save)
    df_history = pd.DataFrame(dict_save)
    df_history.to_csv("./history_LSTM.csv", index=False)