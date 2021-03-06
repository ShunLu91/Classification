import os
import sys
import jieba
import gensim
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchnet import meter
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


class CONFIG:
    vocab_size = 146214  # 词汇量，与word2id中的词汇量一致
    n_class = 2  # 分类数：分别为pos和neg
    max_sen_len = 30  # 句子最大长度 # train:187 # val:192 # test:156
    embedding_dim = 50  # 词向量维度
    batch_size = 50  # 批处理尺寸
    n_hidden = 256  # 隐藏层节点数
    n_epoch = 100  # 训练迭代周期，即遍历整个训练样本的次数
    opt = 'adam'  # 训练优化器：adam
    learning_rate = 0.025
    drop_prob = 0.1  # dropout层，参数keep的比例
    num_filters = 512  # 卷积层filter的数量
    filter_sizes = '3,4,5'
    kernel_size = 4
    LSTM_layers = 3
    save_dir = './checkpoints/'
    train_path = '../dataset/sentiment/train.txt'
    dev_path = '../dataset/sentiment/dev.txt'
    test_path = '../dataset/sentiment/test.txt'
    word2id_path = '../dataset/sentiment/word2id.txt'
    pre_word2vec_path = '../dataset/sentiment/wiki_word2vec_50.bin'
    corpus_word2vec_path = '../dataset/sentiment/word2vec.txt'
    data_save_dir = '../dataset/sentiment/'


def build_word2id():
    """
    :param file: word2id保存地址
    构建词汇表并存储
    """
    word2id = {'_PAD_': 0}
    path = [CONFIG.train_path, CONFIG.dev_path, CONFIG.test_path]
    # 给每个词编号然后存入word2id这个字典中

    for _path in path:
        step = 0
        with open(_path, encoding='utf-8') as f:
            for line in f.readlines():
                sp = line.strip().split()[1] + line.strip().split()[2]
                sp = jieba.cut(sp)
                words = list(sp)
                for word in words:
                    if word not in word2id.keys():
                        word2id[word] = len(word2id)

                step += 1
                if step % 10000 == 0:
                    print('current step:{}'.format(step))

    with open(CONFIG.word2id_path, 'w', encoding='utf-8') as f:
        for w in word2id:
            f.write(w + '\t')
            f.write(str(word2id[w]))
            f.write('\n')


def build_word2vec(fname, word2id, save_to_path=None):
    """
    :param fname: 预训练的word2vec.
    :param word2id: 语料文本中包含的词汇集.
    基于预训练好的word2vec和词汇表构建向量
    :param save_to_path: 保存训练语料库中的词组对应的word2vec到本地
    :return: 语料文本中词汇集对应的word2vec向量{id: word2vec}.
    """
    n_words = max(word2id.values()) + 1
    model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True)
    word_vecs = np.array(np.random.uniform(-1., 1., [n_words, model.vector_size]))
    for word in word2id.keys():
        try:
            word_vecs[word2id[word]] = model[word]
        except KeyError:
            pass
    if save_to_path:
        with open(save_to_path, 'w', encoding='utf-8') as f:
            for vec in word_vecs:
                vec = [str(w) for w in vec]
                f.write(' '.join(vec))
                f.write('\n')
    return word_vecs


def read_data(path, word_dict, mode):
    step = 0
    data = {'comment': [], 'label': [], 'max_len': 0}
    with open(path, 'r') as f:
        for line in f:
            sp = line.strip().split()[1] + line.strip().split()[2]
            sp = jieba.cut(sp)
            words = list(sp)
            if words:
                line_vec = []
                for _w in words:
                    line_vec.append(word_dict.get(_w))
                # 每条评论不等长的处理
                if len(line_vec) > data['max_len']:
                    data['max_len'] = len(line_vec)
                while len(line_vec) < CONFIG.max_sen_len:
                    line_vec.append(0)
                if len(line_vec) > CONFIG.max_sen_len:
                    line_vec = line_vec[:CONFIG.max_sen_len]
                data['comment'].append(line_vec)
                data['label'].append(int(line.strip().split()[0]))

            step += 1
            if step % 10000 == 0:
                print('current step:{}'.format(step))
    print('max_len: {}'.format(data['max_len']))

    # save
    np.save(os.path.join(CONFIG.data_save_dir, '{}_max{}_data.npy'.format(mode, CONFIG.max_sen_len)), data)

    return data


class Set(Dataset):
    def __init__(self, data, mode):
        super(Set, self).__init__()
        self.mode = mode
        self.data = data['comment']
        self.label = data['label']

    def __getitem__(self, index):
        _data = np.array(self.data[index])
        _label = np.array(self.label[index])

        return torch.from_numpy(_data).long(), torch.from_numpy(_label).long()

    def __len__(self):
        return len(self.label)


def accuracy(output, label, topk=(1,)):
    maxk = max(topk)
    batch_size = label.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(label.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# 模型构建
class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.args = args
        label_num = args.n_class
        filter_num = args.num_filters
        filter_sizes = [int(fsz) for fsz in args.filter_sizes.split(',')]

        embedding_dim = args.embedding_dim
        word_vec = torch.from_numpy(self.args.word2_vec)
        self.embedding = nn.Embedding(self.args.vocab_size, self.args.embedding_dim, _weight=word_vec)

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, filter_num, (fsz, embedding_dim)) for fsz in filter_sizes])
        # self.dropout = nn.Dropout(args.drop_keep_prob)
        self.linear = nn.Linear(len(filter_sizes) * filter_num, label_num)

    def forward(self, x):
        # 经过embedding,x的维度为(batch_size, max_len, embedding_dim)
        x = self.embedding(x)
        # 经过view函数x的维度变为(batch_size, input_chanel=1, w=max_len, h=embedding_dim)
        x = x.view(x.size(0), 1, x.size(1), self.args.embedding_dim).float()
        # 经过卷积运算,x中每个运算结果维度为(batch_size, out_chanel, w, h=1)
        x = [F.relu(conv(x)) for conv in self.convs]
        # 经过最大池化层,维度变为(batch_size, out_chanel, w=1, h=1)
        x = [F.max_pool2d(input=x_item, kernel_size=(x_item.size(2), x_item.size(3))) for x_item in x]
        # 将不同卷积核运算结果维度（batch，out_chanel,w,h=1）展平为（batch, outchanel*w*h）
        x = [x_item.view(x_item.size(0), -1) for x_item in x]
        x = torch.cat(x, 1)  # 将不同卷积核提取的特征组合起来
        # x = self.dropout(x)
        logits = self.linear(x)
        return logits


# def pre_weight(vocab_size):
#     weight = torch.zeros(vocab_size, CONFIG.embedding_dim)
#     #初始权重
#     for i in range(len(word2vec_model.index2word)):#预训练中没有word2ix，所以只能用索引来遍历
#         try:
#             index = word2ix[word2vec_model.index2word[i]]#得到预训练中的词汇的新索引
#         except:
#             continue
#         weight[index, :] = torch.from_numpy(word2vec_model.get_vector(
#             ix2word[word2ix[word2vec_model.index2word[i]]]))#得到对应的词向量
#     return weight


class SentimentModel(nn.Module):
    def __init__(self):
        super(SentimentModel, self).__init__()
        self.hidden_dim = CONFIG.n_hidden

        embedding_dim = CONFIG.embedding_dim
        word_vec = torch.from_numpy(CONFIG.word2_vec)
        self.embedding = nn.Embedding(CONFIG.vocab_size, CONFIG.embedding_dim, _weight=word_vec)

        # requires_grad指定是否在训练过程中对词向量的权重进行微调
        self.embedding.weight.requires_grad = True
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=CONFIG.LSTM_layers,
                            batch_first=True, dropout=CONFIG.drop_prob, bidirectional=False)
        self.dropout = nn.Dropout(CONFIG.drop_prob)
        self.fc1 = nn.Linear(7680, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 2)

    #   self.linear = nn.Linear(self.hidden_dim, vocab_size)# 输出的大小是词表的维度，

    def forward(self, input, hidden=None):

        embeds = self.embedding(input).float()  # [batch, seq_len] => [batch, seq_len, embed_dim]
        # embeds = pack_padded_sequence(embeds, CONFIG.max_sen_len, batch_first=True)
        batch_size, seq_len = input.size()
        if hidden is None:
            h_0 = input.data.new(CONFIG.LSTM_layers * 1, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(CONFIG.LSTM_layers * 1, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        # print('embeds.shape:', embeds.shape)
        output, hidden = self.lstm(embeds, (h_0, c_0))  # hidden 是h,和c 这两个隐状态
        # output, _ = pad_packed_sequence(output, batch_first=True)
        # print('output.shape:', output.shape)
        x = output.reshape((output.shape[0], -1))
        # print('x.shape:', x.shape)
        output = self.dropout(torch.tanh(self.fc1(x)))
        output = torch.tanh(self.fc2(output))
        output = self.fc3(output)
        # last_outputs = self.get_last_output(output, CONFIG.max_sen_len)
        # output = output.reshape(batch_size * seq_len, -1)
        return output

    def get_last_output(self, output, batch_seq_len):
        last_outputs = torch.zeros((output.shape[0], output.shape[2]))
        for i in range(len(batch_seq_len)):
            last_outputs[i] = output[i][batch_seq_len[i] - 1]  # index 是长度 -1
        last_outputs = last_outputs.to(output.device)
        return last_outputs


if __name__ == '__main__':
    # 首先生成word2id
    # build_word2id()

    # 读word2id
    word2id_dict = {}
    with open(CONFIG.word2id_path, 'r') as f:
        for line in f:
            line = line.strip().split()
            word2id_dict[line[0]] = int(line[1])

    # 根据word2id生成word2vec
    # word2_vec = build_word2vec(
    #     fname=CONFIG.pre_word2vec_path,
    #     word2id=word2id_dict,
    #     save_to_path=CONFIG.corpus_word2vec_path)

    CONFIG.word2_vec = np.loadtxt(CONFIG.corpus_word2vec_path)

    # word2vec加载
    # word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(CONFIG.word2_vec, binary=True)

    # train_data = read_data(CONFIG.train_path, word2id_dict, mode='train')
    # valid_data = read_data(CONFIG.dev_path, word2id_dict, mode='valid')
    # test_data = read_data(CONFIG.test_path, word2id_dict, mode='test')

    train_data = np.load(os.path.join(CONFIG.data_save_dir, '{}_max{}_data.npy'.format('train', CONFIG.max_sen_len)),
                         allow_pickle=True).item()
    valid_data = np.load(os.path.join(CONFIG.data_save_dir, '{}_max{}_data.npy'.format('valid', CONFIG.max_sen_len)),
                         allow_pickle=True).item()
    test_data = np.load(os.path.join(CONFIG.data_save_dir, '{}_max{}_data.npy'.format('test', CONFIG.max_sen_len)),
                        allow_pickle=True).item()

    train_set = Set(train_data, mode='train')
    valid_set = Set(valid_data, mode='valid')
    test_set = Set(test_data, mode='test')
    train_queue = DataLoader(train_set, batch_size=CONFIG.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valid_queue = DataLoader(valid_set, batch_size=CONFIG.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_queue = DataLoader(test_set, batch_size=CONFIG.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    print('Dataset: Train={}, Val={}, Test={}'.format(len(train_set), len(valid_set), len(test_set)))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = TextCNN(CONFIG)
    model = SentimentModel()
    # model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=CONFIG.n_class)
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model = model.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG.learning_rate, weight_decay=3e-4)

    for epoch in range(CONFIG.n_epoch):
        train_acc = 0
        n = 0
        model.train()
        train_loss = meter.AverageValueMeter()
        acc = meter.AverageValueMeter()
        for step, (inputs, targets) in enumerate(train_queue):
            n += 1
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # print('outputs', outputs.shape)
            loss = criterion(outputs, targets)
            loss.backward()
            prec = accuracy(outputs, targets, topk=(1,))
            acc.add(prec[0].cpu())
            optimizer.step()
            train_loss.add(loss.item())
            # train_acc += accuracy_score(torch.argmax(outputs.cpu().data, dim=1), targets.cpu())
            sys.stdout.write(
                '\r(Epoch: {epoch}/{n_epoch} | Batch: {batch}/{size}) | Loss: {loss:.4f} | train_acc: {acc: .4f}'.format(
                    epoch=epoch,
                    n_epoch=CONFIG.n_epoch,
                    batch=step + 1,
                    size=len(train_queue),
                    loss=train_loss.mean,
                    acc=acc.mean)
            )
            sys.stdout.flush()

        model.eval()
        val_loss = meter.AverageValueMeter()
        val_acc = meter.AverageValueMeter()
        with torch.no_grad():
            for step, (inputs, targets) in enumerate(valid_queue):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss.add(loss.item())
                prec = accuracy(outputs, targets, topk=(1,))
                val_acc.add(prec[0].cpu())
        print('\nval_acc: {:.3f}, val_loss: {:.3f}'.format(val_acc.mean, val_loss.mean))

        model.eval()
        test_loss = meter.AverageValueMeter()
        test_acc = meter.AverageValueMeter()
        test_label = []
        test_pre = []
        with torch.no_grad():
            for step, (inputs, targets) in enumerate(test_queue):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss.add(loss.item())
                prec = accuracy(outputs, targets, topk=(1,))
                test_acc.add(prec[0].cpu())
                test_label.extend(targets.cpu().numpy().tolist())
                test_pre.extend(outputs.cpu().numpy().argmax(axis=1).tolist())
        print('test_acc: {:.3f}, test_loss: {:.3f}'.format(test_acc.mean, test_loss.mean))
        # print('confusion matrix: \n', confusion_matrix(test_label, test_pre))
        # print('recall_score:', metrics.recall_score(test_label, test_pre))
        # print('f1_score:', metrics.f1_score(test_label, test_pre))
