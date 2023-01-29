import json
import h5py
import pandas as pd
import numpy as np
import jieba
import re
from collections import Counter


def get_data(path_):

    df_data = pd.read_excel(path_, sheet_name=None)
    df_data.pop('其他')
    keys = df_data.keys()

    return df_data, keys


def stop_words(path_stop_words):

    with open(path_stop_words, "r", encoding='utf-8') as f:  # 打开文件
        word_stop = []
        for line in f.readlines():
            line = line.strip('\n')  # 去掉列表中每一个元素的换行符
            word_stop.append(str(line))
        f.close()

    return word_stop


def cut_word(line_list):

    words_count = Counter()  # 创建计数器，用于统计词频
    word_line_list = []  # 存放不同种类的词语列表
    list_weight = []  # 标签权重

    for line_list_ in line_list:  # 循环遍历不同种类的字符串列表

        list_weight.append(len(line_list_))
        words_list = []  # 存放切分后的词语列表
        all_words = []
        for line in line_list_:  # 遍历该类别内的字符串数据
            line.strip('\n')  # 移除字符串中的空格
            # 将字母数字标点替换为空格
            line = re.sub("[A-Za-z0-9\：\·\—\，\。\“ \”]", "", line)
            # 中文字符串分词
            seq_list = jieba.lcut(line, cut_all=False)
            # 将分词后的词语列表seg_list加入到全部单词列表all_words中
            all_words = all_words + seq_list
            words_list.append(seq_list)
        word_line_list.append(words_list)
        # 遍历all_words列表，更新计数器
        for x in all_words:
            if len(x) > 1 and x != '\r\n':
                words_count[x] += 1

    return word_line_list, words_count, np.max(list_weight) / np.array(list_weight)


def data_process(data, index_):
    label_dict = dict(zip(range(len(index_)), index_))
    line_list = []
    for i, j in label_dict.items():
        data_ = data[j]
        data_['text'] = data_['title'] + data_['content']
        data_.dropna(subset=['text'], inplace=True)
        data_.drop(data_[data_['text'] == ""].index.tolist(), axis=0, inplace=True)
        if len(data_) == 0:
            continue
        else:
            line_list.append(data_.text.values.tolist())

    word_line_list, words_count, label_weight = cut_word(line_list)

    return word_line_list, words_count, label_dict, label_weight


def get_vector(word_line_list, words_count, stop_word):

    for i in stop_word:  # 删除停止词
        del words_count[i]

    words_counts = dict(sorted(words_count.items(), key=lambda x: x[1]))  # 按照词频排序
    # 按照词频进行编码，从3开始编码，0、1、2不参与编码
    words_index = dict(zip(words_counts.keys(), range(3, len(words_counts.keys()) + 3)))

    word_vector = []  # 存放编码完成的词语列表
    words_label = []  # 存放label
    for lines_, label in zip(word_line_list, range(len(word_line_list))):
        for line in lines_:  # 遍历每种类别
            words = []
            words.append(1)  # 文章开始用1作为标志
            for i in line:  # 遍历词语列表
                if i in words_index:  # 按照words_index进行编码，如果当前词不在words_index编码表中，则直接跳过
                    words.append(words_index[i])
                else:
                    continue
            words.append(2)  # 文章结束用2作为标志
            if len(words) >= 600:  # 对词语列表进行截取，只取前600个词，如果当前列表不足600则用0补充
                words = words[:599]
                words.append(2)
            else:
                words = words + (600 - len(words))*[0]

            word_vector.append(words)
            words_label.append(label)

    return words_index, word_vector, words_label


if __name__ == "__main__":

    path_train = r'.\Data\1617241934831197.xlsx'
    path_stop_words = r".\stop_word.txt"

    stop_word = stop_words(path_stop_words)
    df_train, index_train = get_data(path_train)
    word_line_list, words_count, label_dict, label_weight = data_process(df_train, index_train)
    words_index, word_vector, words_label = get_vector(word_line_list, words_count, stop_word)

    print('index_train', index_train)
    print('words_index', len(words_index))
    print('label_weight', label_weight)

    y_train = np.array(words_label)
    x_train = np.array(word_vector)

    print('x_train', x_train.shape)
    print('y_train', y_train.shape)

    with h5py.File(r'data_train.h5', 'w') as hf:
        hf.create_dataset("data_x",  data=x_train)
        hf.create_dataset("data_y",  data=y_train)
        hf.create_dataset("label_weight", data=label_weight)

    words_index = json.dumps(words_index)
    with open('words_index.json', 'w') as json_file:
        json_file.write(words_index)

    label_dict = json.dumps(label_dict)
    with open('label_dict.json', 'w') as json_file:
        json_file.write(label_dict)

    print("write json success")




