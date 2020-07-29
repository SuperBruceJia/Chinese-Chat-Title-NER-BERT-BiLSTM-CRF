# encoding=utf-8
# 加载所需要的必需包
# pip install bert-base==0.0.7 -i https://pypi.python.org/simple
import tensorflow as tf
import numpy as np
import codecs
import pickle
import os
import pandas as pd

from bert_base.train.models import create_model, InputFeatures
from bert_base.bert import tokenization, modeling
from bert_base.train.train_helper import get_args_parser

args = get_args_parser()

# 训练好的模型路径：final_output文件夹下
model_dir = r'/Users/shuyuej/Desktop/Python-Files/BERT-BiLSTM-CRF-NER/final_output'

# 原始的bert模型：chinese_L-12_H-768_A-12文件夹下
bert_dir = '/Users/shuyuej/Desktop/Python-Files/BERT-BiLSTM-CRF-NER/init_checkpoint/chinese_L-12_H-768_A-12'

# 读取测试的数据集：Test-set.xlsx
test_data = pd.read_excel('/Users/shuyuej/Desktop/Python-Files/BERT-BiLSTM-CRF-NER/Test-set.xlsx')
test_data = np.array(test_data)

# 设置参数
is_training = False
use_one_hot_embeddings = False
batch_size = 1

# 建立词典供后面使用
Data_base = ['哥', '姐', '小', '老', '妹', '大', '长', '师', '兄', '女', '书', '记', '总', '先', '生', '同', '学', '板', '处', '局']

# 是否使用GPU加速
gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
sess = tf.Session(config=gpu_config)
model = None

global graph
input_ids_p, input_mask_p, label_ids_p, segment_ids_p = None, None, None, None

if not os.path.exists(os.path.join(model_dir, "checkpoint")):
    raise Exception("failed to get checkpoint. going to return")

# 加载label->id的词典
with codecs.open(os.path.join(model_dir, 'label2id.pkl'), 'rb') as rf:
    label2id = pickle.load(rf)
    id2label = {value: key for key, value in label2id.items()}

with codecs.open(os.path.join(model_dir, 'label_list.pkl'), 'rb') as rf:
    label_list = pickle.load(rf)
num_labels = len(label_list) + 1

graph = tf.get_default_graph()
with graph.as_default():
    # print("going to restore checkpoint")
    # sess.run(tf.global_variables_initializer())
    input_ids_p  = tf.placeholder(tf.int32, [batch_size, args.max_seq_length], name="input_ids")
    input_mask_p = tf.placeholder(tf.int32, [batch_size, args.max_seq_length], name="input_mask")

    # 配置BERT
    bert_config = modeling.BertConfig.from_json_file(os.path.join(bert_dir, 'bert_config.json'))
    (total_loss, logits, trans, pred_ids) = create_model(
        bert_config=bert_config, is_training=False, input_ids=input_ids_p, input_mask=input_mask_p,
        segment_ids=None, labels=None, num_labels=num_labels, use_one_hot_embeddings=False, dropout_rate=1.0)

    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))

tokenizer = tokenization.FullTokenizer(vocab_file=os.path.join(bert_dir, 'vocab.txt'), do_lower_case=args.do_lower_case)


def predict_online():
    """
    do online prediction. each time make prediction for one instance.
    you can change to a batch if you want.

    :param line: a list. element is: [dummy_label,text_a,text_b]
    :return:
    """
    score = 0

    def convert(line):
        feature = convert_single_example(0, line, label_list, args.max_seq_length, tokenizer, 'p')
        input_ids = np.reshape([feature.input_ids], (batch_size, args.max_seq_length))
        input_mask = np.reshape([feature.input_mask], (batch_size, args.max_seq_length))
        segment_ids = np.reshape([feature.segment_ids], (batch_size, args.max_seq_length))
        label_ids = np.reshape([feature.label_ids], (batch_size, args.max_seq_length))

        return input_ids, input_mask, segment_ids, label_ids

    global start_point
    global graph
    with graph.as_default():
        for i in range(len(test_data)):
            # 读取测试的数据集
            sentence = test_data[i][0]
            print('The Print Sentence is', sentence)
            start_point = i
            if len(sentence) < 2:
                print(sentence)
                continue
            sentence = tokenizer.tokenize(sentence)
            input_ids, input_mask, segment_ids, label_ids = convert(sentence)
            feed_dict = {input_ids_p: input_ids, input_mask_p: input_mask}
            # run session get current feed_dict result
            pred_ids_result = sess.run([pred_ids], feed_dict)
            pred_label_result = convert_id_to_label(pred_ids_result, id2label)
            final_output = strage_combined_link_org_loc(sentence, pred_label_result[0])

            # 测试模型输出的结果，若与label相同，score则+1
            if list(final_output) == list(test_data[start_point][2]):
                score += 1
            print("Current score is ", score)
            print('Test ', i+1)
            print('\n')

        # 计算测试集的准确率
        accuracy = score / len(test_data)
        print('Final Accuracy is ', accuracy)


def convert_id_to_label(pred_ids_result, idx2label):
    """
    将id形式的结果转化为真实序列结果
    :param pred_ids_result:
    :param idx2label:
    :return:
    """
    result = []
    for row in range(batch_size):
        curr_seq = []
        for ids in pred_ids_result[row][0]:
            if ids == 0:
                break
            curr_label = idx2label[ids]
            if curr_label in ['[CLS]', '[SEP]']:
                continue
            curr_seq.append(curr_label)
        result.append(curr_seq)
    return result


def strage_combined_link_org_loc(tokens, tags):
    """
    组合策略
    :param pred_label_result:
    :param types:
    :return:
    """

    # Step One: 通过模型实现人名（这里主要是大名与聊天称谓）的NER命名实体识别
    def print_output(data, type):
        line = []
        line.append(type)
        for i in data:
            line.append(i.word)

        final_result = line[1:]
        for m in range(len(final_result)):
            if '老' == final_result[m]:
                final_result[m] += final_result[m+1]
                final_result.remove(final_result[m+1])
                break

        for m in range(len(final_result)):
            if '老师' == final_result[m]:
                final_result[m-1] += final_result[m]
                final_result.remove(final_result[m])
                break

        for m in range(len(final_result)):
            if '老板' == final_result[m]:
                final_result[m-1] += final_result[m]
                final_result.remove(final_result[m])
                break

        for m in range(len(final_result)):
            if '处长' == final_result[m]:
                final_result[m-1] += final_result[m]
                final_result.remove(final_result[m])
                break

        for m in range(len(final_result)):
            if '局长' == final_result[m]:
                final_result[m-1] += final_result[m]
                final_result.remove(final_result[m])
                break

        for m in range(len(final_result)):
            if '将军' == final_result[m]:
                final_result[m-1] += final_result[m]
                final_result.remove(final_result[m])
                break

        for m in range(len(final_result)):
            if '司令' == final_result[m]:
                final_result[m-1] += final_result[m]
                final_result.remove(final_result[m])
                break

        for m in range(len(final_result)):
            if '主席' == final_result[m]:
                final_result[m-1] += final_result[m]
                final_result.remove(final_result[m])
                break

        for m in range(len(final_result)):
            if '先生' == final_result[m]:
                final_result[m-1] += final_result[m]
                final_result.remove(final_result[m])
                break

        for m in range(len(final_result)):
            if '女士' == final_result[m]:
                final_result[m-1] += final_result[m]
                final_result.remove(final_result[m])
                break

        for m in range(len(final_result)):
            if '师兄' == final_result[m]:
                final_result[m-1] += final_result[m]
                final_result.remove(final_result[m])
                break

        for m in range(len(final_result)):
            if '师姐' == final_result[m]:
                final_result[m-1] += final_result[m]
                final_result.remove(final_result[m])
                break

        for m in range(len(final_result)):
            if '同学' == final_result[m]:
                final_result[m-1] += final_result[m]
                final_result.remove(final_result[m])
                break

        for m in range(len(final_result)):
            if '校长' == final_result[m]:
                final_result[m-1] += final_result[m]
                final_result.remove(final_result[m])
                break

        for m in range(len(final_result)):
            if '兄' == final_result[m]:
                final_result[m-1] += final_result[m]
                final_result.remove(final_result[m])
                break

        for m in range(len(final_result)):
            if '弟' == final_result[m]:
                final_result[m-1] += final_result[m]
                final_result.remove(final_result[m])
                break

        for m in range(len(final_result)):
            if '姐' == final_result[m]:
                final_result[m-1] += final_result[m]
                final_result.remove(final_result[m])
                break

        for m in range(len(final_result)):
            if '妹' == final_result[m]:
                final_result[m-1] += final_result[m]
                final_result.remove(final_result[m])
                break

        print('Stage One - Find the Title:', final_result)

        # 将所有NER识别到的名称给合并，比如说['小贾', '老张'] 拆分为 ['小', '贾', '老', '张']
        concat_name = []
        for i in range(len(line[1:])):
            concat_name += list(line[1:][i])

        # 输入人的全称大名
        name = test_data[start_point][1]
        print('The Input Person Name is', name)
        # 将输入的人名给拆分为单个字，例如['贾舒越'] 拆分为 ['贾', '舒', '越']
        input_name = []
        for j in range(len(name)):
            input_name += list(name[j])

        # 开始做匹配，如果匹配成功就输出，否则no match
        # 设置两个定时器，timer用于是否有匹配成功的；
        # timer_two用于输出匹配的第一个匹配结果
        timer = 0
        for i in range(len(input_name)):
            for j in range(len(concat_name)):
                # NER识别词的字至少出现在了输入的全称人名中一次的话
                if input_name[i] == concat_name[j]:
                    timer += 1

                else:
                    continue

        # NER识别词的字一次都没有出现在输入的全称人名中的话
        # 就输出There is no match for （输入的全称人名）
        if timer == 0:
            print('Stage Two - Print the Model Final Result:', 'There is no match for', name)
            final_output = str("There is no match for %s" % name)

        # 至少出现一次之后继续往下执行
        else:
            # 设置定时器2，用于
            timer_two = 0

            # 将识别出来的词分开并拆分
            # 例如['小贾', '老张'] 分开为 ['小贾']、 ['老张']
            for y in range(len(final_result)):
                # 设置定时器3，用于判断拆分后的NER的识别词中是否至少有一个相同的拆分后的全称人名
                timer_three = 0

                # 判断拆分后的NER的识别词中是否至少有一个相同的拆分后的全称人名
                for i in range(len(input_name)):
                    for j in range(len(final_result[y])):
                        # 如果有二者有相同的字，timer_three自增1
                        if input_name[i] == final_result[y][j]:
                            timer_three += 1

                # 此时，拆分后的NER的识别词中第一次出现相同的全称人名，程序继续进行
                if timer_three != 0:
                    final_result_single = final_result[y]
                    # 第一次 ['小贾'] 拆分为['小', '贾'], 第二次 ['老张'] 拆分为['老', '张']
                    final_result_cut_single = []
                    for j in range(len(final_result_single)):
                        final_result_cut_single += list(final_result_single[j])

                    # 求取NER识别出来的词与输入的人名之间的交集intersection
                    # 比如上例，比如说针对于第一次，intersection_of_two为['贾']
                    intersection_of_two = list(set(input_name).intersection(set(final_result_cut_single)))

                    # 除去NER识别词与全称人名之间的交集（即识别词除二者交集之外的其余部分）
                    # 比如上例，对于第一个['小', '贾']，除去与['贾', '舒', '越']的交集后
                    # final_result_except_inter为['小']
                    final_result_except_inter = final_result_cut_single
                    for t in range(len(intersection_of_two)):
                        if intersection_of_two[t] in final_result_except_inter:
                            final_result_except_inter.remove(intersection_of_two[t])

                        else:
                            continue

                    # 建立 [字典库 + 全称大名]
                    Data_final_base = list(Data_base + input_name)

                    # 下方为Step 2：通过编写规则实现匹配
                    # 若 NER识别词包含于全称大名内
                    # 或者 NER识别词除去二者交集后的词在[字典库 + 全称大名]之内
                    # 则认为NER提取出来的词为全称大名的一个称谓
                    if set(final_result_cut_single) <= set(input_name):
                        timer_two += 1
                        if timer_two == 1:
                            final_key_result = final_result[y]
                            print('Stage Two - Print the Model Final Result:', final_key_result, 'is', name)
                            final_output = str("%s is %s" % (final_key_result, name))
                        else:
                            continue

                    else:
                        if list(set(final_result_except_inter) & set(Data_final_base)) != []:
                            timer_two += 1
                            if timer_two == 1:
                                final_key_result = final_result[y]
                                print('Stage Two - Print the Model Final Result:', final_key_result, 'is', name)
                                final_output = str("%s is %s" % (final_key_result, name))
                            else:
                                continue

                        else:
                            print('Stage Two - Print the Model Final Result:', 'There is no match for', name)
                            final_output = str("There is no match for %s" % name)

                else:
                    continue

        return final_output

    params = None
    eval = Result(params)
    if len(tokens) > len(tags):
        tokens = tokens[:len(tags)]
    person, loc, org = eval.get_result(tokens, tags)

    # 输出句子中识别出的人名
    final_output = print_output(person, 'PER')

    return final_output


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode):
    """
    将一个样本进行分析，然后将字转化为id, 标签转化为id,然后结构化到InputFeatures对象中
    :param ex_index: index
    :param example: 一个样本
    :param label_list: 标签列表
    :param max_seq_length:
    :param tokenizer:
    :param mode:
    :return:
    """
    label_map = {}
    # 1表示从1开始对label进行index化
    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i
    # 保存label->index 的map
    if not os.path.exists(os.path.join(model_dir, 'label2id.pkl')):
        with codecs.open(os.path.join(model_dir, 'label2id.pkl'), 'wb') as w:
            pickle.dump(label_map, w)

    tokens = example
    # tokens = tokenizer.tokenize(example.text)
    # 序列截断
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")  # 句子开始设置CLS 标志
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    label_ids.append(label_map["[CLS]"])  # O OR CLS 没有任何影响，不过我觉得O 会减少标签个数,不过拒收和句尾使用不同的标志来标注，使用LCS 也没毛病
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(0)
    ntokens.append("[SEP]")  # 句尾添加[SEP] 标志
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)  # 将序列中的字(ntokens)转化为ID形式
    input_mask = [1] * len(input_ids)

    # padding, 使用
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")
        # label_mask.append(0)

    # print(len(input_ids))
    assert len(input_ids)   == max_seq_length
    assert len(input_mask)  == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids)   == max_seq_length
    # assert len(label_mask) == max_seq_length

    # 结构化为一个类
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        # label_mask = label_mask
    )
    return feature


class Pair(object):
    def __init__(self, word, start, end, type, merge=False):
        self.__word = word
        self.__start = start
        self.__end = end
        self.__merge = merge
        self.__types = type

    @property
    def start(self):
        return self.__start

    @property
    def end(self):
        return self.__end

    @property
    def merge(self):
        return self.__merge

    @property
    def word(self):
        return self.__word

    @property
    def types(self):
        return self.__types

    @word.setter
    def word(self, word):
        self.__word = word

    @start.setter
    def start(self, start):
        self.__start = start

    @end.setter
    def end(self, end):
        self.__end = end

    @merge.setter
    def merge(self, merge):
        self.__merge = merge

    @types.setter
    def types(self, type):
        self.__types = type

    def __str__(self) -> str:
        line = []
        line.append('entity:{}'.format(self.__word))
        line.append('start:{}'.format(self.__start))
        line.append('end:{}'.format(self.__end))
        line.append('merge:{}'.format(self.__merge))
        line.append('types:{}'.format(self.__types))
        return '\t'.join(line)


class Result(object):
    def __init__(self, config):
        self.config = config
        self.person = []
        self.loc = []
        self.org = []
        self.others = []

    def get_result(self, tokens, tags, config=None):
        # 先获取标注结果
        self.result_to_json(tokens, tags)
        return self.person, self.loc, self.org

    def result_to_json(self, string, tags):
        """
        将模型标注序列和输入序列结合 转化为结果
        :param string: 输入序列
        :param tags: 标注结果
        :return:
        """
        item = {"entities": []}
        entity_name = ""
        entity_start = 0
        idx = 0
        last_tag = ''

        for char, tag in zip(string, tags):
            if tag[0] == "S":
                self.append(char, idx, idx+1, tag[2:])
                item["entities"].append({"word": char, "start": idx, "end": idx+1, "type":tag[2:]})

            elif tag[0] == "B":
                if entity_name != '':
                    self.append(entity_name, entity_start, idx, last_tag[2:])
                    item["entities"].append({"word": entity_name, "start": entity_start, "end": idx, "type": last_tag[2:]})
                    entity_name = ""
                entity_name += char
                entity_start = idx

            elif tag[0] == "I":
                entity_name += char

            elif tag[0] == "O":
                if entity_name != '':
                    self.append(entity_name, entity_start, idx, last_tag[2:])
                    item["entities"].append({"word": entity_name, "start": entity_start, "end": idx, "type": last_tag[2:]})
                    entity_name = ""

            else:
                entity_name = ""
                entity_start = idx
            idx += 1
            last_tag = tag

        if entity_name != '':
            self.append(entity_name, entity_start, idx, last_tag[2:])
            item["entities"].append({"word": entity_name, "start": entity_start, "end": idx, "type": last_tag[2:]})

        return item

    def append(self, word, start, end, tag):
        if tag == 'LOC':
            self.loc.append(Pair(word, start, end, 'LOC'))

        elif tag == 'PER':
            self.person.append(Pair(word, start, end, 'PER'))

        elif tag == 'ORG':
            self.org.append(Pair(word, start, end, 'ORG'))

        else:
            self.others.append(Pair(word, start, end, tag))


if __name__ == "__main__":
    max_time = 0
    score = 0
    if max_time <= len(test_data):
        predict_online()
        max_time += 1


