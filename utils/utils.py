
"""
The MIT License

Copyright (c) 2021 Yeong-Dae Kwon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.



THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import json
import logging
import logging.config
import os
import shutil
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pytz
import torch

process_start_time = datetime.now(pytz.timezone("Asia/Seoul"))
b = os.path.abspath('.')
result_folder = b+'/result/' + process_start_time.strftime("%Y%m%d_%H%M%S") + '{desc}'


def get_result_folder():
    return result_folder


def set_result_folder(folder):
    global result_folder
    result_folder = folder


def create_logger(log_file=None):
    if 'filepath' not in log_file:
        log_file['filepath'] = get_result_folder()

    if 'desc' in log_file:
        log_file['filepath'] = log_file['filepath'].format(desc='_' + log_file['desc'])
    else:
        log_file['filepath'] = log_file['filepath'].format(desc='')

    set_result_folder(log_file['filepath'])

    if 'filename' in log_file:
        filename = log_file['filepath'] + '/' + log_file['filename']
    else:
        filename = log_file['filepath'] + '/' + 'log.txt'

    if not os.path.exists(log_file['filepath']):
        os.makedirs(log_file['filepath'])

    file_mode = 'a' if os.path.isfile(filename)  else 'w'

    root_logger = logging.getLogger()
    root_logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(filename)s(%(lineno)d) : %(message)s", "%Y-%m-%d %H:%M:%S")

    for hdlr in root_logger.handlers[:]:
        root_logger.removeHandler(hdlr)

    # write to file
    fileout = logging.FileHandler(filename, mode=file_mode)
    fileout.setLevel(logging.INFO)
    fileout.setFormatter(formatter)
    root_logger.addHandler(fileout)

    # write to console
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    root_logger.addHandler(console)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += (val * n)
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count else 0


class LogData:
    def __init__(self):
        self.keys = set()
        self.data = {}

    def get_raw_data(self):
        return self.keys, self.data

    def set_raw_data(self, r_data):
        self.keys, self.data = r_data

    def append_all(self, key, *args):
        if len(args) == 1:
            value = [list(range(len(args[0]))), args[0]]
        elif len(args) == 2:
            value = [args[0], args[1]]
        else:
            raise ValueError('Unsupported value type')

        if key in self.keys:
            self.data[key].extend(value)
        else:
            self.data[key] = np.stack(value, axis=1).tolist()
            self.keys.add(key)

    def append(self, key, *args):
        if len(args) == 1:
            args = args[0]

            if isinstance(args, int) or isinstance(args, float):
                if self.has_key(key):
                    value = [len(self.data[key]), args]
                else:
                    value = [0, args]
            elif type(args) == tuple:
                value = list(args)
            elif type(args) == list:
                value = args
            else:
                raise ValueError('Unsupported value type')
        elif len(args) == 2:
            value = [args[0], args[1]]
        else:
            raise ValueError('Unsupported value type')

        if key in self.keys:
            self.data[key].append(value)
        else:
            self.data[key] = [value]
            self.keys.add(key)

    def get_last(self, key):
        if not self.has_key(key):
            return None
        return self.data[key][-1]

    def has_key(self, key):
        return key in self.keys

    def get(self, key):
        split = np.hsplit(np.array(self.data[key]), 2)

        return split[1].squeeze().tolist()

    def getXY(self, key, start_idx=0):
        # print("===========key==========")
        # print(key)
        # print("======self.data[key]======")
        # print(self.data[key])
        split = np.hsplit(np.array(self.data[key]), 2)

        xs = split[0].squeeze().tolist()
        ys = split[1].squeeze().tolist()

        if type(xs) is not list:
            return xs, ys

        if start_idx == 0:
            return xs, ys
        elif start_idx in xs:
            idx = xs.index(start_idx)
            return xs[idx:], ys[idx:]
        else:
            raise KeyError('no start_idx value in X axis data.')

    def get_keys(self):
        return self.keys


class TimeEstimator:
    def __init__(self):
        self.logger = logging.getLogger('TimeEstimator')
        self.start_time = time.time()
        self.count_zero = 0

    def reset(self, count=1):
        self.start_time = time.time()
        self.count_zero = count-1

    def get_est(self, count, total):
        curr_time = time.time()
        elapsed_time = curr_time - self.start_time
        remain = total-count
        remain_time = elapsed_time * remain / (count - self.count_zero)

        elapsed_time /= 3600.0
        remain_time /= 3600.0

        return elapsed_time, remain_time

    def get_est_string(self, count, total):
        elapsed_time, remain_time = self.get_est(count, total)

        elapsed_time_str = "{:.2f}h".format(elapsed_time) if elapsed_time > 1.0 else "{:.2f}m".format(elapsed_time*60)
        remain_time_str = "{:.2f}h".format(remain_time) if remain_time > 1.0 else "{:.2f}m".format(remain_time*60)

        return elapsed_time_str, remain_time_str

    def print_est_time(self, count, total):
        elapsed_time_str, remain_time_str = self.get_est_string(count, total)

        self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
            count, total, elapsed_time_str, remain_time_str))


def util_print_log_array(logger, result_log: LogData):
    assert type(result_log) == LogData, 'use LogData Class for result_log.'

    for key in result_log.get_keys():
        logger.info('{} = {}'.format(key+'_list', result_log.get(key)))


def util_save_log_image_with_label(result_file_prefix,
                                   img_params,
                                   result_log: LogData,
                                   labels=None):
    dirname = os.path.dirname(result_file_prefix)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    _build_log_image_plt(img_params, result_log, labels)

    if labels is None:
        labels = result_log.get_keys()
    file_name = '_'.join(labels)
    fig = plt.gcf()

    # print("========result_file_prefix======")
    # print(result_file_prefix)
    # print("========file_name======")
    # print(file_name)


    fig.savefig('{}-{}.jpg'.format(result_file_prefix, file_name))
    plt.close(fig)
# /public/home/luof/project/BQ-POMO/CVRP/utils/log_image_style/style_tsp_100.json

def _build_log_image_plt(img_params,
                         result_log: LogData,
                         labels=None):
    assert type(result_log) == LogData, 'use LogData Class for result_log.'

    # Read json
    folder_name = img_params['json_foldername']
    file_name = img_params['filename']
    log_image_config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), folder_name, file_name)

    # print("==========log_image_config_file=========")
    # print(log_image_config_file)

    with open(log_image_config_file, 'r') as f:
        config = json.load(f)

    figsize = (config['figsize']['x'], config['figsize']['y'])
    plt.figure(figsize=figsize)

    if labels is None:
        labels = result_log.get_keys()
    for label in labels:
        # print("=====label====")
        # print(label)
        plt.plot(*result_log.getXY(label), label=label)

    plt.draw()

    ylim_min = config['ylim']['min']
    ylim_max = config['ylim']['max']

    # print("=====before_ylim_min=====")
    # print(ylim_min)   
    # print("=====before_ylim_max=====")
    # print(ylim_max)

    if ylim_min is None:
        ylim_min = plt.gca().dataLim.ymin
    if ylim_max is None:
        ylim_max = plt.gca().dataLim.ymax

    # print("=====ylim_min=====")
    # print(ylim_min)   
    # print("=====ylim_max=====")
    # print(ylim_max)

    plt.ylim(ylim_min, ylim_max)
    
    plt.ylim(ylim_min, ylim_max)

    xlim_min = config['xlim']['min']
    xlim_max = config['xlim']['max']
    if xlim_min is None:
        xlim_min = plt.gca().dataLim.xmin
    if xlim_max is None:
        xlim_max = plt.gca().dataLim.xmax
    plt.xlim(xlim_min, xlim_max)

    plt.rc('legend', **{'fontsize': 18})
    plt.legend()
    plt.grid(config["grid"])


def copy_all_src(dst_root):
    # execution dir
    if os.path.basename(sys.argv[0]).startswith('ipykernel_launcher'):
        execution_path = os.getcwd()
    else:
        execution_path = os.path.dirname(sys.argv[0])

    # home dir setting
    tmp_dir1 = os.path.abspath(os.path.join(execution_path, sys.path[0]))
    tmp_dir2 = os.path.abspath(os.path.join(execution_path, sys.path[1]))

    if len(tmp_dir1) > len(tmp_dir2) and os.path.exists(tmp_dir2):
        home_dir = tmp_dir2
    else:
        home_dir = tmp_dir1

    # make target directory
    dst_path = os.path.join(dst_root, 'src')

    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    for item in sys.modules.items():
        key, value = item

        if hasattr(value, '__file__') and value.__file__:
            src_abspath = os.path.abspath(value.__file__)

            if os.path.commonprefix([home_dir, src_abspath]) == home_dir:
                dst_filepath = os.path.join(dst_path, os.path.basename(src_abspath))

                if os.path.exists(dst_filepath):
                    split = list(os.path.splitext(dst_filepath))
                    split.insert(1, '({})')
                    filepath = ''.join(split)
                    post_index = 0

                    while os.path.exists(filepath.format(post_index)):
                        post_index += 1

                    dst_filepath = filepath.format(post_index)

                shutil.copy(src_abspath, dst_filepath)


def tour_nodes_to_tour_len(nodes, W_values):
    """Helper function to calculate tour length from ordered list of tour nodes.
    """
    tour_len = 0
    for idx in range(len(nodes) - 1):
        i = nodes[idx]
        j = nodes[idx + 1]
        tour_len += W_values[i][j]
    # Add final connection of tour in edge target
    tour_len += W_values[j][nodes[0]]
    return tour_len

def is_valid_tour(nodes, num_nodes):
    """Sanity check: tour visits all nodes given.
    """
    return sorted(nodes) == [i for i in range(num_nodes)]


def Scale(X):
    """
    The Scale function takes in a batch of points and scales them to be between 0 and 1.
    It does this by translating the points so that the minimum x-value is at 0, 
    and then dividing all x-values by the maximum value. It does this for both dimensions.
    
    :param X: Store the data and the scale_method parameter is used to determine how to scale it
    :param scale_method: Decide whether to scale the data based on the boundary of all points or just
    :return: The scaled x and the ratio
    """
    B = X.size(0)
    SIZE = X.size(1)
    X = X - torch.reshape(torch.min(X,1).values,(B,1,2)).repeat(1,SIZE,1) # translate
    ratio_x = torch.reshape(torch.max(X[:,:,0], 1).values - torch.min(X[:,:,0], 1).values,(-1,1))
    ratio_y = torch.reshape(torch.max(X[:,:,1], 1).values - torch.min(X[:,:,1], 1).values,(-1,1))
    ratio = torch.max(torch.cat((ratio_x,ratio_y),1),1).values
    ratio[ratio==0] = 1
    X = X / (torch.reshape(ratio,(B,1,1)).repeat(1,SIZE,2))
    return X, ratio

def Scale_for_vrp(X,num):
    """
    The Scale function takes in a batch of points and scales them to be between 0 and 1.
    It does this by translating the points so that the minimum x-value is at 0, 
    and then dividing all x-values by the maximum value. It does this for both dimensions.
    
    :param X: Store the data and the scale_method parameter is used to determine how to scale it
    :param scale_method: Decide whether to scale the data based on the boundary of all points or just
    :return: The scaled x and the ratio
    """
    B = X.size(0)
    SIZE = X.size(1)
    graph = X[:,:num,:]
    min_values = torch.reshape(torch.min(graph,1).values,(B,1,2)).repeat(1,SIZE,1)
    X = X - min_values # translate
    ratio_x = torch.reshape(torch.max(graph[:,:,0], 1).values - torch.min(graph[:,:,0], 1).values,(-1,1))
    ratio_y = torch.reshape(torch.max(graph[:,:,1], 1).values - torch.min(graph[:,:,1], 1).values,(-1,1))
    ratio = torch.max(torch.cat((ratio_x,ratio_y),1),1).values
    ratio[ratio==0] = 1
    X = X / (torch.reshape(ratio,(B,1,1)).repeat(1,SIZE,2))
    X[ratio==0,:,:] = X[ratio==0,:,:]+min_values[ratio==0,:,:]
    return X, ratio

def Rotate_aug(X):
    """
    The Rotate_aug function takes in a batch of points and rotates them by a random angle.
    The function also scales the points to be between 0 and 1.
    
    :param X: Pass the input data to the function
    :return: The rotated point cloud and the ratio of the bounding box
    """
    device = X.device
    B = X.size(0)
    SIZE = X.size(1)
    Theta = torch.rand((B,1),device=device)* 2 * np.pi
    Theta = Theta.repeat(1,SIZE)
    tmp1 = torch.reshape(X[:,:,0]*torch.cos(Theta) - X[:,:,1]*torch.sin(Theta),(B,SIZE,1))
    tmp2 = torch.reshape(X[:,:,0]*torch.sin(Theta) + X[:,:,1]*torch.cos(Theta),(B,SIZE,1))
    X_out = torch.cat((tmp1, tmp2), dim=2)
    X_out += 10
    X_out, ratio = Scale(X_out)
    return X_out, ratio

def Reflect_aug(X):
    """
    The Reflect_aug function takes in a batch of points and performs the following operations:
        1. Rotate each point by a random angle between 0 and 2pi radians
        2. Reflect each point across the x-axis (i.e., multiply y coordinate by -2)
        3. Add 10 to all coordinates so that no points are negative anymore (this is for convenience)
        4. Scale all coordinates down to be between 0 and 1
    
    :param X: Pass the data points to the function
    :return: A reflected point cloud and a scale ratio
    """
    device = X.device
    B = X.size(0)
    SIZE = X.size(1)
    Theta = torch.rand((B,1),device=device)* 2 * np.pi
    Theta = Theta.repeat(1,SIZE)
    tmp1 = torch.reshape(X[:,:,0]*torch.cos(2*Theta) + X[:,:,1]*torch.sin(2*Theta),(B,SIZE,1))
    tmp2 = torch.reshape(X[:,:,0]*torch.sin(2*Theta) - X[:,:,1]*torch.cos(2*Theta),(B,SIZE,1))
    X_out = torch.cat((tmp1, tmp2), dim=2)
    X_out += 10
    X_out, ratio = Scale(X_out)
    return X_out, ratio

def mix_aug(X):
    """
    The mix_aug function takes in a batch of images and returns the same batch with half of them rotated and half reflected.
    The function also returns the ratio between the number of pixels that are black after augmentation to before augmentation.
    
    :param X: Pass in the data
    :return: The augmented images and the ratio of the number of augmented images to original ones
    """
    X_out = X.clone()
    X_out[0::2],ratio = Rotate_aug(X[0::2])
    X_out[1::2],ratio = Reflect_aug(X[1::2])
    return X_out,ratio

def run_aug(aug,x,aug_num=None,aug_all=False):
    """
    The run_aug function takes in an augmentation type, a batch of images, and two optional arguments.
    The first optional argument is the number of images to augment per batch. The second is whether or not to 
    augment all the images in the batch (defaults to False). It then returns a copy of x with some augmented 
    images inserted into it.
    
    :param aug: Select the augmentation to apply
    :param x: Pass in the data
    :param aug_num: Control the number of augmented images in each batch
    :param aug_all: Decide whether to apply the augmentation on all images or only a subset of them
    :return: A tensor with the same size as x, but with some of its values replaced by augmented data
    """
    x_clone = x.clone()
    if aug == 'rotate':
        x_out,_ = Rotate_aug(x)
    elif aug == 'reflect':
        x_out,_ = Reflect_aug(x)
    elif aug == 'mix':
        x_out,_ = mix_aug(x)
    elif aug == 'noise':
        x_out = x+torch.rand(x.size(), device=x.device)*1e-5
    else:
        x_out = x
    if not aug_all:
        if aug_num is not None:
            x_out[0::aug_num]=x_clone[0::aug_num]
        else:
            x_out[0]=x_clone[0]
    return x_out


