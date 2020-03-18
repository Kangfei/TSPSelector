import os
import pickle
import tsplib95
import csv
import numpy as np
from math import pi, sin, cos
import statistics as stat
import scipy.sparse as sp
from sklearn.decomposition import PCA
from scipy import sparse
from hilbertcurve.hilbertcurve import HilbertCurve
import matplotlib.pyplot as plt

from torch_geometric.utils.convert import from_scipy_sparse_matrix
from scipy.spatial.distance import euclidean
pathes = [
            # our instances
            '/home/kfzhao/data/our_instances/cl',
            '/home/kfzhao/data/our_instances/cluster',
            '/home/kfzhao/data/our_instances/compression',
            '/home/kfzhao/data/our_instances/expansion',
            '/home/kfzhao/data/our_instances/explosion',
            '/home/kfzhao/data/our_instances/grid',
            '/home/kfzhao/data/our_instances/implosion',
            '/home/kfzhao/data/our_instances/linearprojection',
            '/home/kfzhao/data/our_instances/rotation',
            '/home/kfzhao/data/our_instances/RUE',
        # ECJ instances
        #   '/home/kfzhao/data/ECJ_instances/national'
        #  ,'/home/kfzhao/data/ECJ_instances/rue'
        #  '/home/kfzhao/data/ECJ_instances/tsplib'
        # , '/home/kfzhao/data/ECJ_instances/vlsi'
        ]

filename = '/home/kfzhao/data/ECJ_instances/tsplib/att532.tsp'
def load_tsp_instance(filename, norm = False):
    problem = tsplib95.load_problem(filename)
    N = problem.dimension

    print("problem size:", N)
    # skip large graph with >10000 nodes
    #if N > 10000 or problem.edge_weight_type != 'EUC_2D':
    #    return

    """
    G = problem.get_graph()
#    print(G.number_of_nodes())
    #print(G.nodes)
#    print(G.number_of_edges())
    #print(G.edges)
    """

    mat = np.zeros(shape=(N, N), dtype= np.float32)
    x = np.zeros(shape=(N, 2), dtype= np.float32) # node coordinate


    for i in range(N):
        x[i][0], x[i][1] = problem.node_coords[i + 1][0], problem.node_coords[i + 1][1]
    if norm:
        x = coordinate_normalize(x)
    """
    for i in range(N):
        for j in range(i + 1, N):
            #mat[i, j] = mat[j, i] = problem.wfunc(i + 1, j + 1)
            mat[i, j] = mat[j, i] = euclidean(x[i], x[j])

    A = sp.csr_matrix(mat)
    edge_index, edge_attr = from_scipy_sparse_matrix(A)
    edge_index, edge_attr = edge_index.numpy(), edge_attr.numpy()
    
    instance = {'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr,  'adj': A.todense()}
    with open(os.path.splitext(filename)[0] + '_norm.pickle', 'wb') as out_file:
        pickle.dump(obj = instance, file = out_file, protocol= 3)
        print(filename + " saved.")
        out_file.close()
    """
    instance = {'x': x}
    with open(os.path.splitext(filename)[0] + '.coo.pickle', 'wb') as out_file:
        pickle.dump(obj=instance, file=out_file, protocol=3)
        print(filename + " saved.")
        out_file.close()


def coordinate_normalize(x):
    # normalize the tsp coordinate
    x_min, x_max = x[:,0].min(), x[:,0].max()
    y_min, y_max = x[:,1].min(), x[:,1].max()
    x[:,0], x[:,1] = x[:,0] - x_min, x[:,1] - y_min
    scale = max(x_max - x_min, y_max - y_min)
    x = x / scale
    return x


def matrix_reorder(filename):
    in_file = open(os.path.splitext(filename)[0] + '_norm.pickle', 'rb')
    data = pickle.load(in_file)
    x = data['x']
    A = data['adj']
    N = A.shape[0]
    pca = PCA(n_components=1)
    x = pca.fit_transform(x).squeeze()
    temp = x.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(x))
    for i in range(N):
        A[:, i] = A[ranks, i]
    for i in range(N):
        A[i, :] = A[i, ranks]
    print(A.shape)


def hilbert_matrix_reorder(filename):
    in_file = open(os.path.splitext(filename)[0] + '_norm.pickle', 'rb')
    data = pickle.load(in_file)
    x = data['x']
    A = data['adj']
    N = A.shape[0]
    x_norm = x * 1000
    x_norm = x_norm.astype(int)
    hilbert_curve = HilbertCurve(10,  2)
    temp = np.zeros(shape=( N ))
    for i in range( N ):
        temp[i] = hilbert_curve.distance_from_coordinates([x_norm[i][0], x_norm[i][1]])

    temp = temp.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(temp))
    for i in range(N):
        A[:, i] = A[ranks, i]
    for i in range(N):
        A[i, :] = A[i, ranks]
    print(A.shape)

def savetoTSPImage(filename, num_grid = 256):
    in_file = open(os.path.splitext(filename)[0] + '_norm.pickle', 'rb')
    data = pickle.load(in_file)
    x = data['x']
    image = coordinate_to_grid_image(x = x, num_grid= num_grid)
    for direction in range(4):
        image = np.rot90(image)
        instance = {'adj': image}
        out_file_dir = os.path.splitext(filename)[0] + '_{0}_{1}_image.pickle'.format(num_grid, direction)
        with open(out_file_dir, 'wb') as out_file:
            pickle.dump(obj=instance, file=out_file, protocol=3)
            out_file.close()

def tsp_image_rotate_and_flip(filename, num_grid = 256):
    in_file = open(os.path.splitext(filename)[0] + '_norm.pickle', 'rb')
    data = pickle.load(in_file)
    x = data['x']
    image = coordinate_to_grid_image(x = x, num_grid= num_grid)

    v_image = sparse.csr_matrix(np.flip(image, axis= 0))
    instance = {'adj': v_image}
    out_file_dir = os.path.splitext(filename)[0] + '_{0}_flip0_image.pickle'.format(num_grid)
    with open(out_file_dir, 'wb') as out_file:
        pickle.dump(obj=instance, file=out_file, protocol=3)
        out_file.close()

    h_image = sparse.csr_matrix(np.flip(image, axis=1))
    instance = {'adj': h_image}
    out_file_dir = os.path.splitext(filename)[0] + '_{0}_flip1_image.pickle'.format(num_grid)
    with open(out_file_dir, 'wb') as out_file:
        pickle.dump(obj=instance, file=out_file, protocol=3)
        out_file.close()

    '''
    for direction in range(4):
        image = np.rot90(image)
        instance = {'adj': np.flip(image, axis=0)}
        out_file_dir = os.path.splitext(filename)[0] + '_{0}_{1}_flip0_image.pickle'.format(num_grid, direction)
        with open(out_file_dir, 'wb') as out_file:
            pickle.dump(obj=instance, file=out_file, protocol=3)
            out_file.close()

        instance = {'adj': np.flip(image, axis=1)}
        out_file_dir = os.path.splitext(filename)[0] + '_{0}_{1}_flip1_image.pickle'.format(num_grid, direction)
        with open(out_file_dir, 'wb') as out_file:
            pickle.dump(obj=instance, file=out_file, protocol=3)
            out_file.close()
    '''


def coordinate_rotate(x, angle):
    new_x = np.zeros(shape= x.shape)
    angle = float(angle) / 180.0 * pi
    new_x[:,0] = x[:,0] * cos(angle) - x[:,1] * sin(angle)
    new_x[:, 1] = x[:, 0] * sin(angle) + x[:, 1] * cos(angle)
    return new_x

def coordinate_to_grid_image(x, num_grid = 256):
    image = np.zeros((num_grid, num_grid), dtype=np.float32)
    for i in range(x.shape[0]):
        idx_x = int(x[i][0] * num_grid) - 1
        idx_y = int(x[i][1] * num_grid) - 1
        image[idx_x][idx_y] = image[idx_x][idx_y] + 1.0
    return image


def tsp_image_rotate(filename, num_grid = 256):
    in_file = open(os.path.splitext(filename)[0] + '.pickle', 'rb')
    data = pickle.load(in_file)
    x = data['x']

    for angle in [45, 90, 135, 180, 225, 270, 315, 360]:
        new_x = coordinate_rotate(x, angle)
        # rotate on the original coordinate and normalize
        new_x = coordinate_normalize(new_x)
        image = coordinate_to_grid_image(x = new_x, num_grid=num_grid)
        image = sparse.csr_matrix(image)
        instance = {'adj': image}
        out_file_dir = os.path.splitext(filename)[0] + '_{0}_{1}_image.pickle'.format(num_grid, angle)
        with open(out_file_dir, 'wb') as out_file:
            pickle.dump(obj=instance, file=out_file, protocol=3)
            out_file.close()



def test():
    instances_num = 0
    for path in pathes:
        filelist = os.listdir(path)
        for file in filelist:
            # if file is a directory, skip
            print("process file:" + file)
            if os.path.isdir(os.path.join(path, file)):
                continue
            if os.path.splitext(os.path.join(path, file))[1] == '.tsp':
                load_tsp_instance(os.path.join(path, file))
                instances_num = instances_num + 1
    print("# total generated instances: {}".format(instances_num))
    return instances_num


def input_test(path = '/home/kfzhao/data/ECJ_instances/'):
    labels = load_labels()
    file_list = []
    for key in labels.keys():
        dataset = key.strip().split('_')[0]
        instance_id = key.strip().split('_')[1]
        if dataset == 'morphed':
            node_num = instance_id.strip().split('-')[0]
            tmp1, tmp2 = instance_id.strip().split('---')[0], instance_id.strip().split('---')[1]
            instance_id = node_num + "---" + tmp1 + '.tsp---' + tmp2 + '.tsp'
        full_instance_dir = os.path.join(path, dataset, instance_id) + '.pickle'
        file_list.append((key, full_instance_dir))

    for key, full_instance_dir in file_list:
        try:

            with open(full_instance_dir, 'rb') as in_file:
                data = pickle.load(in_file)
                x = data['x']
                out_file_dir = os.path.splitext(full_instance_dir)[0] + '.coo.pickle'
                instance = {'x': x}
                with open(out_file_dir, 'wb') as out_file:
                    pickle.dump(obj=instance, file=out_file, protocol=3)
                    out_file.close()
                in_file.close()

            #load_tsp_instance(full_instance_dir)
            #savetoTSPImage(full_instance_dir)
            #tsp_image_rotate_and_flip(full_instance_dir)
            #print("proceed: " + full_instance_dir)
        except IOError:
            print("cannot open: " + full_instance_dir)


def visual_instance(filename):
    in_file_dir = os.path.splitext(filename)[0] + '_{0}_{1}_image.pickle'.format(256, 360)
    with open(in_file_dir, 'rb') as in_file:
        data = pickle.load(in_file)
        A = data['adj']
        A = A.todense()
        image_visualization(A)


def image_visualization(mat):
    plt.matshow(mat)
    plt.show()


def process_one_instance(data):
    instance_id = data[0][0]
    dataset = instance_id.split('_')[0].strip()
    """
    if dataset == 'rue':
        return None, None, None
    """
    best_runtime = 36000
    best_algorithm = 'all'
    algorithm_to_result = {}
    for i in range(50):
        ins_id, repeat, algorithm, runtime, runstatus = \
        data[i][0], data[i][1], data[i][2], data[i][3], data[i][4]
        if ins_id != instance_id:
            return None, None, None
        if algorithm not in algorithm_to_result.keys():
            algorithm_to_result[algorithm] = list([runtime])
        else:
            algorithm_to_result[algorithm].append(runtime)

    algorithm_to_median = {} # algorithm-> median runtime
    for key, value in algorithm_to_result.items():
        value.sort()
        algorithm_to_median[key] = (value[4] + value[5]) / 2.0
        if value[5] < 3600:
            if (value[4] + value[5]) / 2.0 < best_runtime:
                best_runtime = (value[4] + value[5]) / 2.0 # median of the test performance
                best_algorithm = key
        else:
            algorithm_to_median[key] = 36000.0
    return instance_id, best_algorithm, algorithm_to_median

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
    label_file = open('./tsp_labels.txt', 'w')
    num_instances = 0
    sbs_run_time = 0
    best_run_time = 0
    for i in range(int(len(data) / 50)):
        instance_id, best_algorithm, algorithm_to_median = process_one_instance(data[i * 50: i * 50 + 50])
        #print(instance_id, best_algorithm)
        if instance_id is not None:
            labels[instance_id] = (best_algorithm, algorithm_to_median)
            num_instances += 1
            sbs_run_time += algorithm_to_median['eax.restart']
            best_run_time += algorithm_to_median[best_algorithm]
            res = instance_id + ',' + best_algorithm
            for key, values in algorithm_to_median.items():
                res = res + ',' + key + ',' + str(values)
            res = res + '\n'
            label_file.write(res)
    #print("num of record:", len(data))
    #print("num of labels:", len(labels))
    label_file.close()
    print("average sbs run time={}".format(sbs_run_time / num_instances))
    print("average best run time={}".format(best_run_time / num_instances))
    return labels

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

def our_label_stat(filename = '/home/kfzhao/data/our_instances/algorithm_runs.csv', time_out = 900.0, num_runs = 30):
    data = []
    labels = {}
    with open(filename, 'r') as label_file:
        csvreader = csv.reader(label_file)
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
            print("failed")
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
    for algorithm, cnt in best_algo_cnt.items():
        print("{} best instance number : {}".format(algorithm, cnt))

    for algorithm, cnt in timeout_algo_cnt.items():
        print("{} timeout instance number : {}".format(algorithm, cnt))

    print("average best run time={}".format(best_run_time / num_instances))
    for algorithm, runtime in single_run_time.items():
        print("{} average runtime : {}".format(algorithm, runtime / num_instances))
    print("# virtual best solver timeout {}".format(num_timeout))

    return labels



if __name__ == "__main__":
    print("start")
    #load_tsp_instance(filename)
    #tsp_image_rotate(filename)
    #input_test()
    #visual_instance(filename)
    #matrix_reorder(filename)
    #load_labels()
    #test()
    our_label_stat()
    print("haha")