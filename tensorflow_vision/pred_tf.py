import tensorflow as tf
import re
import jieba
import json
from Net_tf import NetWork


def file_read(words_index_path, label_dict_path):

    with open(words_index_path, 'r', encoding='utf8') as fp:
        words_index = json.load(fp)

    with open(label_dict_path, 'r', encoding='utf8') as fp:
        label_dict = json.load(fp)

    return words_index, label_dict


def data_process(string_list, words_index):

    data_pred = []
    for string_ in string_list:

        string_.strip('\n')
        string_ = re.sub("[A-Za-z0-9\：\·\—\，\。\“ \”]", "", string_)
        seq_list = jieba.lcut(string_, cut_all=False)

        seq_vector = []
        for i in seq_list:
            if i in words_index:
                seq_vector.append(words_index[i])
            else:
                continue
        seq_vector.append(2)
        if len(seq_vector) >= 600:
            seq_vector = seq_vector[:599]
            seq_vector.append(2)
        else:
            seq_vector = seq_vector + (600 - len(seq_vector)) * [0]

        data_pred.append(seq_vector)

    return data_pred


def load_model(weight_path):

    total_words = 152410 + 3
    embedding_len = 256
    max_review_len = 600
    units = 128
    num_class = 9

    model = NetWork(total_words=total_words, embedding_len=embedding_len,
                    max_review_len=max_review_len, num_class=num_class, units=units)
    model.load_weights(weight_path)
    model.build(input_shape=(64, 300))
    model.summary()


    return model


def pred_(data_pred, label_dict, model):

    input_ = tf.convert_to_tensor(data_pred)
    output_ = model.evaluate(input_)

    y_pred = tf.argmax(output_, axis=1)
    label_list = []
    for i in y_pred.numpy().tolist():
        label_list.append(label_dict[str(i)])

    return label_list


if __name__ == '__main__':

    string_list = ['多名乌克兰政要被曝在迪拜度假，前总理季莫申科被拍后“匆忙掩面逃走”',
                   'EA经典游戏将从GOG下架 《辛迪加战争》将绝版',
                   '马斯克每次年底的预言都是非常准确。马斯克在圣诞节的时候说过2023年经济会大衰退。结果在一月份的时候，特斯拉MODEL 3,MODEL '
                   'Y直接大幅优惠降价。几天时间疯狂收割3万辆订单。这就是一个明显的在衰退之前回笼资金的手段。那2023年因为什么会导致经济大衰退呢？']
    words_index_path = 'words_index.json'
    label_dict_path = 'label_dict.json'
    weight_path = './checkpoint_LSTM/89train_acc00998471test_acc00924982.pth'

    model = load_model(weight_path)
    words_index, label_dict = file_read(words_index_path, label_dict_path)
    data_pred = data_process(string_list, words_index)
    label_list = pred_(data_pred, label_dict, model)

    print(label_list)