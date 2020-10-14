import time
import argparse

from utils import *
from model import Network

train_batch_loss_list = []
train_batch_acc_list = []
train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []

parser = argparse.ArgumentParser('Main')
parser.add_argument('--data_dir', type=str, default='./dataset/MNIST/raw', help='dataset dir')
parser.add_argument('--epochs', type=int, default=10, help='num of epochs')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--print_freq', type=int, default=50, help='print frequency')
parser.add_argument('--best_acc', type=float, default=0.0, help='best test acc')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--layers', type=int, default=3, help='network layers')
parser.add_argument('--dimension', nargs='+', type=int, help='network fc dimension')
parser.add_argument('--dropout_rate', type=float, default=0.0, help='dropout rate')
parser.add_argument('--noise_rate', type=float, default=0.0, help='dropout rate')
args = parser.parse_args()
print(args)


if __name__ == '__main__':
    train_data, train_target = get_data(args.data_dir)
    test_data, test_target = get_data(args.data_dir, 't10k')

    model = Network(
        batchsize=args.batch_size,
        layers=args.layers,
        dimension=args.dimension,
        dropout_rate=args.dropout_rate
    )
    for epoch in range(args.epochs):
        # train
        model.training = True
        train_acc = 0
        train_loss = 0
        for step in range(train_data.shape[0] // args.batch_size):
            batch_loss = 0
            batch_acc = 0
            data = train_data[step * args.batch_size:(step + 1) * args.batch_size].reshape(
                [args.batch_size, 28, 28, 1])
            target = train_target[step * args.batch_size:(step + 1) * args.batch_size]
            if args.noise_rate > 0:
                data = random_noise(data, args.noise_rate)
            output = model.forward(data)
            batch_loss += model.softmax.cal_loss(output, np.array(target))
            train_loss += model.softmax.cal_loss(output, np.array(target))
            for j in range(args.batch_size):
                if np.argmax(model.softmax.softmax[j]) == target[j]:
                    batch_acc += 1
                    train_acc += 1
            model.optimize(args.learning_rate, args.weight_decay)
            if step % args.print_freq == 0:
                print(
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) +
                    "  epoch: %d  step: %5d  avg_batch_acc: %.4f  avg_batch_loss: %.4f  learning_rate %f" % (
                        epoch, step, batch_acc / float(args.batch_size), batch_loss / args.batch_size, args.learning_rate
                    )
                )
                train_batch_loss_list.append(batch_loss / args.batch_size)
                train_batch_acc_list.append(batch_acc / float(args.batch_size))
        train_loss_list.append(train_loss / train_data.shape[0])
        train_acc_list.append(train_acc / float(train_data.shape[0]))
        print(
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) +
            "  epoch: %d  train_acc: %.4f  avg_train_loss: %.4f" % (
                epoch, train_acc / float(train_data.shape[0]), train_loss / train_data.shape[0])
        )

        # test
        model.training = False
        val_acc = 0
        val_loss = 0
        for step in range(int(test_data.shape[0] / args.batch_size)):
            data = test_data[step * args.batch_size:(step + 1) * args.batch_size].reshape([args.batch_size, 28, 28, 1])
            target = test_target[step * args.batch_size:(step + 1) * args.batch_size]
            output = model.forward(data)
            val_loss += model.softmax.cal_loss(output, np.array(target))
            for j in range(args.batch_size):
                if np.argmax(model.softmax.softmax[j]) == target[j]:
                    val_acc += 1
        if args.best_acc < val_acc / float(test_data.shape[0]):
            args.best_acc = val_acc / float(test_data.shape[0])
        test_loss_list.append(val_loss / test_data.shape[0])
        test_acc_list.append(val_acc / float(test_data.shape[0]))
        print(
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) +
            "  epoch: %d  val_acc: %.4f  avg_val_loss: %.4f best_val_acc: %.4f" % (
                epoch, val_acc / float(test_data.shape[0]), val_loss / test_data.shape[0], args.best_acc)
        )

    print(train_batch_loss_list)
    print(train_batch_acc_list)
    print(train_loss_list)
    print(train_acc_list)
    print(test_loss_list)
    print(test_acc_list)
