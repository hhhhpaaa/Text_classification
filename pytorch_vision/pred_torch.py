import torch
import re
import jieba
import json
from Net_torch import NetWork


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
    units = 128
    num_class = 9

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = NetWork(total_words=total_words, embedding_len=embedding_len, units=units, num_class=num_class)
    model.load_state_dict(torch.load(weight_path))
    model = model.to(device)

    return model, device


def pred_(data_pred, label_dict, model, device):

    input_ = torch.tensor(data_pred).to(device)
    model.eval()
    output_ = model(input_)

    _, label_pred = output_.squeeze(dim=0).max(-1)
    label_list = []
    for i in label_pred.tolist():
        label_list.append(label_dict[str(i)])

    return label_list


if __name__ == '__main__':

    string_list = ['俄外长就落实纳卡地区停火分别与阿亚两国外长会谈',
                   'EA经典游戏将从GOG下架 《辛迪加战争》将绝版',
                   '无房北漂昨夜无眠!北京住户与政府共有产权房新政征求意见啦,第一时间带你锁定申购三大关键环节……']
    words_index_path = 'words_index.json'
    label_dict_path = 'label_dict.json'
    weight_path = './checkpoint_LSTM/190train_acc00957342test_acc00865794.pth'

    model, device = load_model(weight_path)
    words_index, label_dict = file_read(words_index_path, label_dict_path)
    data_pred = data_process(string_list, words_index)
    label_list = pred_(data_pred, label_dict, model, device)

    print(label_list)

