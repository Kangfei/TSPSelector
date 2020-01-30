from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
import os

import pickle
import random
import numpy as np
from torchvision.models import  alexnet, resnet18, vgg11, vgg11_bn, vgg16, vgg16_bn
from InstanceLoader import *

from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader as GeoDataLoader
import torch.optim as optim
from mpnn import MPNN, CGCNN, SimpleCNN


def process_one_instance(data):
    instance_id = data[0][0]
    best_runtime = 36000
    best_algorithm = 'all'
    algorithm_to_result = {}
    for i in range(50):
        ins_id, repeat, algorithm, runtime, runstatus = \
        data[i][0], data[i][1], data[i][2], data[i][3], data[4]
        if ins_id != instance_id:
            return None, None
        if algorithm not in algorithm_to_result.keys():
            algorithm_to_result[algorithm] = list([runtime])
        else:
            algorithm_to_result[algorithm].append(runtime)

    for key, value in algorithm_to_result.items():
        value.sort()
        if value[5] < 3600:
            if (value[4] + value[5]) / 2.0 < best_runtime:
                best_runtime = (value[4] + value[5]) / 2.0 # median of the test performance
                best_algorithm = key
    return instance_id, best_algorithm

def load_labels(filename = '/home/kfzhao/data/ECJ_instances/algorithm_runs.arff.txt'):

    file = open(filename, 'r')
    data = []
    labels = {}
    for line in file:
        if line.strip().startswith('@') or line.strip() == '':
            continue
        line = line.strip().split(',')
        ins_id, repeat, algorithm, runtime, runstatus = \
        str(line[0]), int(line[1]), str(line[2]), float(line[3]), str(line[4])
        data.append((ins_id, repeat, algorithm, runtime, runstatus))
    file.close()
    print("num of record:", len(data))
    for i in range(int(len(data) / 50)):
        instance_id, best_algorithm = process_one_instance(data[i * 50: i * 50 + 50])
        #print(instance_id, best_algorithm)
        if instance_id is not None:
            labels[instance_id] = best_algorithm
    print("num of labels:", len(labels))
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


def cnn_validate(args, model, dataloader):
    model.eval()
    num_instances = 0
    correct = 0
    for i, (data, label) in enumerate(dataloader):
        if args.cuda:
            data, label = data.cuda(), label.cuda()

        outputs = model(data)
        outputs = torch.nn.functional.log_softmax(outputs, dim=1)  # for resnet
        outputs = outputs.squeeze()

        idx = torch.argmax(outputs, dim = 1)
        correct += (idx == label.squeeze()).sum().item()
        num_instances += label.shape[0]

    accuracy = float(correct) / num_instances
    return accuracy

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


def cnn_train(args, model, train_dataloader, val_dataloader, optimizer, scheduler = None):
    device = args.device
    if args.cuda:
        model.to(device)

    max_accuracy = 0.0
    for epoch in range(args.epoches):
        model.train()
        for i, (data, label) in enumerate(train_dataloader):
            if args.cuda:
                data, label = data.cuda(), label.cuda()

            optimizer.zero_grad()
            outputs = model(data)
            outputs = torch.nn.functional.log_softmax(outputs, dim=1)  # for cnn
            label = label.reshape((label.shape[0]))

            #print('output:',outputs.shape)
            #print('label:', label.shape)

            loss = torch.nn.functional.nll_loss(outputs, label)
            loss.backward()
            optimizer.step()

            if args.verbose and (i + 1) % 50 == 0:
                print('epoch:{} loss: {:^10}'.format(epoch, loss.item()))

        train_accuracy = cnn_validate(args, model, train_dataloader)
        val_accuracy = cnn_validate(args, model, val_dataloader)
        max_accuracy = max(val_accuracy, max_accuracy)
        # decay the learning rate
        scheduler.step(val_accuracy)
        if args.verbose:
            print('epoch:{} train accuracy: {:^10}'.format(epoch, train_accuracy))
            print('epoch:{} val accuracy: {:^10}'.format(epoch, val_accuracy))
    if args.verbose:
        print("finish training.")
    return model, max_accuracy


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

def select_model(args):
    model_type = args.model_type
    kwargs = {"num_classes" : 5}

    if model_type is 'resnet18':
        model = resnet18(pretrained=False, progress=True, **kwargs)
    elif model_type is 'vgg11':
        model = vgg11(pretrained=False, progress=True, **kwargs)
    elif model_type is 'vgg11_bn':
        model = vgg11_bn(pretrained=False, progress=True, **kwargs)
    elif model_type is 'vgg16:':
        model = vgg16(pretrained=False, progress=True, **kwargs)
    elif model_type is 'vgg16_bn':
        model = vgg16_bn(pretrained=False, progress=True, **kwargs)
    else:
        model = alexnet(pretrained=False, progress=True, **kwargs)

    if args.verbose:
        print(model)
    return model


def main_cnn(args):
    instances_path = args.instances_path
    batch_size = args.batch_size
    num_node_feats = args.in_ch
    weight_decay = args.weight_decay
    decay_factor = args.decay_factor

    lr = args.learning_rate

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda' if args.cuda else 'cpu')

    label_path = os.path.join(instances_path, 'algorithm_runs.arff.txt')
    labels = load_labels(label_path)
    # split the dataset
    keys = list(labels.keys())
    random.shuffle(keys)
    train_keys, val_keys = keys[:1500], keys[1500:]
    train_labels = { k : labels[k] for k in train_keys}
    val_labels = {k : labels[k] for k in val_keys}


    #train_dataset = InstanceDataset(num_node_feats, instances_path, train_labels)
    #val_dataset = InstanceDataset(num_node_feats, instances_path, val_labels)

    train_dataset = AugmentInstanceDataset(num_node_feats, instances_path, train_labels)
    val_dataset = AugmentInstanceDataset(num_node_feats, instances_path, val_labels)
    if args.verbose:
        print("# training images: {}".format(train_dataset.num))
        print("# validation images: {}".format(val_dataset.num))

    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle= True)
    val_dataloader = DataLoader(val_dataset, batch_size = 32, shuffle= True)

    model = select_model(args)

    optimizer = optim.Adam(model.parameters(),
                           lr=lr, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor= decay_factor)

    _, max_accuracy = cnn_train(args, model, train_dataloader, val_dataloader, optimizer, scheduler)
    return max_accuracy

if __name__ == "__main__":
    parser = ArgumentParser("TSP Selector", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler="resolve")
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
    # Model Settings (ONLY FOR CNN)
    parser.add_argument("--model_type", type=str, default='vgg11_bn')
    # Training settings
    parser.add_argument("--epoches", default=100, type=int)
    parser.add_argument("--batch_size", default= 16, type=int)
    parser.add_argument("--learning_rate", default=0.00001, type=float)
    parser.add_argument('--weight_decay', type=float, default=5e-3,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--decay_factor', type=float, default=0.99,
                        help='decay rate of learning rate (linear).')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    # Other
    parser.add_argument("--instances_path", type=str, default="/home/kfzhao/data/ECJ_instances")
    parser.add_argument("--verbose", default=True, type=bool)
    args = parser.parse_args()
    if args.verbose:
        print(args)
    #main(args)
    max_accuracy = main_cnn(args)
    print("max_accuracy", max_accuracy)