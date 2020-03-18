from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
import os

import pickle
import random
import numpy as np
import math
import csv
import statistics as stat
from InstanceLoader import *
from transform import  *
from torch.utils.data import DataLoader
import torch.optim as optim
from cnn import select_model, select_criterion


def process_one_our_instance(data, time_out = 900.0, num_runs = 30):
    instance_id = data[0][0]

    """
    if dataset == 'rue':
        return None, None, None
    """
    best_runtime = time_out
    best_algorithm = 'all'
    algorithm_to_result = {}
    for i in range(num_runs):
        ins_id, algorithm, repeat, runstatus, runtime = \
        data[i][0], data[i][1], data[i][2], data[i][3], float(data[i][4])
        if ins_id != instance_id:
            return None, None, None
        if algorithm not in algorithm_to_result.keys():
            algorithm_to_result[algorithm] = list([runtime])
        else:
            algorithm_to_result[algorithm].append(runtime)

    algorithm_to_median = {} # algorithm-> median runtime
    for key, value in algorithm_to_result.items():
        median = stat.median(value)
        algorithm_to_median[key] = median
        if median < time_out:
            if median < best_runtime:
                best_runtime = median # median of the test performance
                best_algorithm = key
        else:
            algorithm_to_median[key] = time_out * 10
    return instance_id, best_algorithm, algorithm_to_median

def load_labels(filename = '/home/kfzhao/data/our_instances/algorithm_runs.csv', time_out = 900.0, num_runs = 30):
    data = []
    labels = {}
    with open(filename, 'r') as label_file:
        csvreader = csv.reader(label_file)
        next(csvreader)
        for _ in range(0 * num_runs):
            next(csvreader)

        for line in csvreader:
            data.append(line)
    label_file.close()
    num_instances = 0
    best_algo_cnt = {} # algo-> best instance count
    timeout_algo_cnt = {} # algo-> timeout instance count
    single_run_time = {} # algo-> average run time
    num_timeout = 0
    best_run_time = 0
    for i in range(3000):
        instance_id, best_algorithm, algorithm_to_median = \
            process_one_our_instance(data[i * num_runs: i * num_runs + num_runs], time_out, num_runs)
        if instance_id is None:
            continue
        num_instances += 1
        if best_algorithm not in best_algo_cnt.keys():
            best_algo_cnt[best_algorithm] = 0
        best_algo_cnt[best_algorithm] += 1
        best_run_time += algorithm_to_median[best_algorithm]
        if algorithm_to_median[best_algorithm] >= time_out:
            num_timeout += 1

        for algorithm, runtime in algorithm_to_median.items():
            if algorithm not in single_run_time.keys():
                single_run_time[algorithm] = 0.0
            single_run_time[algorithm] += runtime
            if algorithm not in timeout_algo_cnt.keys():
                timeout_algo_cnt[algorithm] = 0
            if runtime >= time_out:
                timeout_algo_cnt[algorithm] += 1
        # save the label
        instance_id = os.path.splitext(instance_id)[0] # remove the extent '.tsp'
        labels[instance_id] = (best_algorithm, algorithm_to_median)
    """
    for algorithm, cnt in best_algo_cnt.items():
        print("{} best instance number : {}".format(algorithm, cnt))

    for algorithm, cnt in timeout_algo_cnt.items():
        print("{} timeout instance number : {}".format(algorithm, cnt))

    print("average best run time={}".format(best_run_time / num_instances))
    for algorithm, runtime in single_run_time.items():
        print("{} average runtime : {}".format(algorithm, runtime / num_instances))
    print("# virtual best solver timeout {}".format(num_timeout))
    """
    return labels

def validate(args, model, dataloader):
    model.eval()
    device = args.device
    num_instances = 0
    correct = 0
    for i, data in enumerate(dataloader):
        if args.cuda:
            data.to(device)

        outputs = model(data)
        outputs = outputs.squeeze()
        #print(outputs)
        idx = torch.argmax(outputs)
        #print(idx)
        if torch.equal(idx, data.y.squeeze()):
            correct  = correct + 1
        num_instances = num_instances + 1
    accuracy = float(correct) / num_instances
    return accuracy

def compute_pred_performance(idx, run_time):
    idx, run_time = idx.cpu().numpy(), run_time.numpy()
    res = 0.0
    best_runtime = 0.0
    improve_cnt = 0
    for i in range(idx.shape[0]): # for one batch
        #print(run_time[i][idx[i]], run_time[i][1])
        res += run_time[i][idx[i]]
        best_runtime += np.min(run_time[i])
        if run_time[i][idx[i]] <= run_time[i][1]:
            improve_cnt += 1
    return res, improve_cnt, best_runtime


def cnn_validate(args, model, dataloader):
    loss_type = args.loss_type
    model.eval()
    num_instances = 0
    correct = 0
    improve = 0
    pred_performance= 0.0
    best_performance = 0.0
    single_best_performance = 0.0
    for i, (data, label, _, run_time) in enumerate(dataloader):
        if args.cuda:
            data, label = data.cuda(), label.cuda()

        outputs = model(data)

        if loss_type == 'nll':
            outputs = torch.nn.functional.log_softmax(outputs, dim=1)
            outputs = outputs.squeeze()
            idx = torch.argmax(outputs, dim=1)
        elif loss_type == 'sce':
            outputs = torch.nn.functional.log_softmax(outputs, dim=1)
            idx = torch.argmax(outputs, dim=1)
            label = torch.argmax(label, dim=1)
        elif loss_type == 'bce':
            outputs = torch.nn.functional.sigmoid(outputs)
            idx = torch.argmax(outputs, dim=1)
            label = torch.argmax(label, dim=1)
        elif loss_type == 'mse':
            #outputs = torch.nn.functional.sigmoid(outputs)
            idx = torch.argmin(outputs, dim=1)
            label = torch.argmin(label, dim=1)


        #label = torch.argmax(label, dim = 1) # for soft cross entory, bce
        #idx = torch.argmin(outputs, dim = 1) # for mse loss
        #label = torch.argmin(label, dim = 1)
        #idx = torch.argmax(outputs, dim=1) # for nll loss

        correct += (idx == label.squeeze()).sum().item()
        num_instances += label.shape[0]
        # compute the real performance
        res, improve_cnt, best_runtime = compute_pred_performance(idx, run_time)
        #print(idx, label)
        pred_performance += res
        best_performance += best_runtime
        improve += improve_cnt
        single_best_performance += (run_time[:, 1].sum().item())  # 1->eax.restart

    accuracy = float(correct) / num_instances
    improve_rate = float(improve) / num_instances

    if args.verbose:
        print("pred performance={}".format(pred_performance))
        print("single best performance={}".format(single_best_performance))
        print("best performance={}".format(best_performance))
        print("improve rate={}".format(improve_rate))

    return accuracy, pred_performance


def batch_train(args, model, train_dataloader, val_dataloader, optimizer, scheduler = None):
    device = args.device
    if args.cuda:
        model.to(device)

    for epoch in range(args.epoches):
        model.train()
        for i, data in enumerate(train_dataloader):
            if args.cuda:
                data.to(device)

            outputs = model(data)
            print(outputs.shape)
            print('label:', data.y.shape)
            #print(data.x.shape, data.edge_index.shape)
            loss = torch.nn.functional.nll_loss(outputs, data.y) / args.batch_size
            loss.backward()


            if (i + 1) % args.batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()

            if args.verbose and (i + 1) % 100 * args.batch_size == 0:
                print('epoch:{} loss: {:^10}'.format(epoch, loss.item()))

        train_accuracy = validate(args, model, train_dataloader)
        val_accuracy = validate(args, model, val_dataloader)
        if args.verbose:
            print('epoch:{} train accuracy: {:^10}'.format(epoch, train_accuracy))
            print('epoch:{} val accuracy: {:^10}'.format(epoch, val_accuracy))

    print("finish training.")
    return model

def cnn_train(args, model, train_dataloader, val_dataloader, optimizer, criterion, scheduler = None):
    device = args.device
    if args.cuda:
        model.to(device)

    max_train_acc, max_val_acc = 0.0, 0.0
    best_train_performance, best_val_performance = float('inf'), float('inf')
    for epoch in range(args.epoches):
        total_loss = 0.0
        model.train()
        for i, (data, label, weights, _) in enumerate(train_dataloader):
            if args.cuda:
                data, label, weights = data.cuda(), label.cuda(), weights.cuda()

            optimizer.zero_grad()
            outputs = model(data)

            loss = criterion(outputs, label, weights)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        #train_accuracy, train_performance = cnn_validate(args, model, train_dataloader)
        val_accuracy, val_performance = cnn_validate(args, model, val_dataloader)
        #max_train_acc = max(train_accuracy, max_train_acc)
        max_val_acc = max(val_accuracy, max_val_acc)
        #best_train_performance = min(train_performance, best_train_performance)
        best_val_performance = min(val_performance, best_val_performance)

        # decay the learning rate
        if (epoch + 1) % args.decay_patience == 0:
            scheduler.step()
        if args.verbose:
            print('epoch:{} loss: {:^10}'.format(epoch, total_loss))
            #print('epoch:{} train accuracy: {:^10}'.format(epoch, train_accuracy))
            print('epoch:{} val accuracy: {:^10}'.format(epoch, val_accuracy))
    if args.verbose:
        print("finish training.")
    return model, max_train_acc, max_val_acc, best_train_performance, best_val_performance


'''
def main(args):
    instances_path = args.instances_path
    batch_size = args.batch_size
    num_node_feats = args.in_ch
    hid_ch = args.hid_ch
    out_ch = args.out_ch
    num_edge_hid = args.num_edge_hid
    num_hid = args.num_hid
    dropout = args.dropout
    weight_decay = args.weight_decay
    batch_norm = args.batch_norm
    lr = args.learning_rate

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda' if args.cuda else 'cpu')

    labels = load_labels()
    # split the dataset
    keys = list(labels.keys())
    random.shuffle(keys)
    train_keys, val_keys = keys[:1500], keys[1500:]
    train_labels = { k : labels[k] for k in train_keys}
    val_labels = {k : labels[k] for k in val_keys}


    train_dataset = GeoInstanceDataset(num_node_feats, instances_path, train_labels)
    val_dataset = GeoInstanceDataset(num_node_feats, instances_path, val_labels)

    train_dataloader = GeoDataLoader(train_dataset, batch_size = 1, shuffle = True)
    val_dataloader = GeoDataLoader(val_dataset, batch_size=1, shuffle=True)

    """
    model = MPNN(in_ch= num_node_feats, hid_ch = hid_ch,
                 out_ch = out_ch, num_edge_feats = 1,
                 num_edge_hid= num_edge_hid, num_hid = num_hid,
                 num_classes= 5, dropout = dropout, batch_norm = batch_norm)
    """

    model = CGCNN(in_ch= num_node_feats, num_edge_feats = 1, num_hid = num_hid,
                  num_classes= 5, dropout = dropout, batch_norm = batch_norm)

    optimizer = optim.Adam(model.parameters(),
                           lr=lr, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, factor=0.8)

    model = batch_train(args, model, train_dataloader, val_dataloader, optimizer, scheduler)
'''


def build_transform(args):
    num_rotate = args.num_rotate
    num_grid = args.num_grid
    scale_factor = args.scale_factor
    flip = args.flip
    bt = BuildTransformation(num_rotate, num_grid, scale_factor, flip)

    return bt.get_train_transform(), bt.get_val_transform()

def main_cnn(args, train_dataset, val_dataset):
    batch_size = args.batch_size
    weight_decay = args.weight_decay
    decay_factor = args.decay_factor
    num_workers = args.num_workers

    lr = args.learning_rate

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda' if args.cuda else 'cpu')

    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle= True, num_workers = num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size = 32, shuffle= False, num_workers = num_workers)

    model, criterion = select_model(args), select_criterion(args)

    optimizer = optim.Adam(model.parameters(),
                           lr=lr, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_factor)

    _, max_train_acc, max_val_acc, best_train_performance, best_val_performance = \
        cnn_train(args, model, train_dataloader, val_dataloader, optimizer, criterion, scheduler)
    if args.verbose:
        print("max_train_accuracy={}".format(max_train_acc))
    print("max_val_accuracy={}".format(max_val_acc))
    print("best_train_performance={}".format(best_train_performance))
    print("best_val_performance={}".format(best_val_performance))
    return best_val_performance

def cross_validation(args, num_fold = 5):

    instances_path = args.instances_path
    label_path = os.path.join(instances_path, 'algorithm_runs.csv')
    labels = load_labels(label_path)

    # split the dataset
    keys = list(labels.keys())
    random.shuffle(keys)
    num_instances = int(len(keys))
    num_fold_instances = num_instances / num_fold
    train_transforms, val_transforms = build_transform(args)
    val_performance = 0.0
    for i in range(num_fold):
        start = int(i * num_fold_instances)
        end = num_instances if i == num_fold - 1 else int(i * num_fold_instances + num_fold_instances)

        val_keys = keys[start: end]
        train_keys = keys[: start] + keys[end:]
        train_labels = {k: labels[k] for k in train_keys}
        val_labels = {k: labels[k] for k in val_keys}
        train_dataset = ArgumentDataset(args, instances_path, train_labels, train_transforms)
        val_dataset = ArgumentDataset(args, instances_path, val_labels, val_transforms)
        if args.verbose:
            print("Fold {}: # training images: {}".format(i,train_dataset.num))
            print("Fold {}: # validation images: {}".format(i,val_dataset.num))
        val_performance += main_cnn(args, train_dataset, val_dataset)
        if args.verbose:
            print("Fold {} finished.".format(i))
    avg_run_time = 0 - val_performance / num_instances
    print("avg_run_time={}".format(avg_run_time))

if __name__ == "__main__":
    parser = ArgumentParser("TSP Selector", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler="resolve")
    """
    # Model Settings (ONLY FOR MPNN MODEL)
    parser.add_argument("--in_ch", default=2, type=int,
                        help="input features dim of nodes")
    parser.add_argument("--hid_ch", default=32, type=int,
                        help="number of hidden units of MPNN")
    parser.add_argument("--out_ch", default=32, type=int,
                        help="number of output units of MPNN")
    parser.add_argument("--num_edge_hid", default=32, type=int,
                        help="number of hidden units of edge network")
    parser.add_argument("--num_hid", default=32, type=int,
                        help="number of hidden units of MLP")
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument("--batch_norm", default=False, type=bool)
    """
    # Simple CNN Settings
    parser.add_argument("--num_cov_layer", default=5, type=int,
                        help="number of convolution layers")
    parser.add_argument("--num_mlp_layer", default=3, type=int,
                        help="number of mlp layers")
    parser.add_argument("--channels", type=str, default='64 192 384 256 256',
                        help="number of channels of each cov layers")
    parser.add_argument("--mlp_hids", type=str, default='4096 4096 512',
                        help="number of hidden units of the mlp layer")
    parser.add_argument("--kernel_size", default=3, type=int,
                        help="convolution kernel size ")
    parser.add_argument("--stride", default=2, type=int,
                        help="convolution stride")
    parser.add_argument("--adp_output_size", default=6, type=int,
                        help="adaptive pooling output size ")
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='Dropout rate (1 - keep probability).')
    # Data Argument Settings
    parser.add_argument("--num_rotate", default=17, type=int,
                        help="number of rotation in 2*pi")
    parser.add_argument("--num_grid", default=64, type=int,
                        help="number of grid in the tsp image")
    parser.add_argument("--scale_factor", default=4, type=int,
                        help="reduce the image resolution by scale_factor")
    parser.add_argument("--flip", default=True, type=bool)
    # Model Settings (ONLY FOR CNN)
    parser.add_argument("--model_type", type=str, default='resnet34')
    # loss type: nll, sce, bce, mse
    parser.add_argument("--loss_type", type=str, default='sce')
    # Training settings
    parser.add_argument("--num_classes", default=6, type=int,
                        help="number of classes")
    parser.add_argument("--num_fold", default=5, type=int,
                        help="number of fold for cross validation")
    parser.add_argument("--epoches", default=10, type=int)
    parser.add_argument("--batch_size", default= 16, type=int)
    parser.add_argument("--learning_rate", default= 2e-4, type=float)
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--decay_factor', type=float, default=0.8,
                        help='decay rate of (gamma).')
    parser.add_argument('--decay_patience', type=int, default=50,
                        help='num of epoches for one lr decay.')
    parser.add_argument('--weight_exp', type=float, default=2.0,
                        help='loss weight exp factor.')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--num_workers', type = int, default= 16,
                        help='number of workers for Dataset.')
    # Other
    parser.add_argument("--instances_path", type=str, default="/home/kfzhao/data/our_instances")
    parser.add_argument("--verbose", default=True, type=bool)
    args = parser.parse_args()
    if args.verbose:
        print(args)

    cross_validation(args, args.num_fold)
