# -*- coding: utf-8 -*-
"""
@Project:   metrics_bundle
@File   :   main.py
@Author :   TonyMao@AILab
@Date   :   2019-10-25
@Desc   :   None
"""
import multiprocessing as mp
import os
import sys
from pprint import pprint

import cv2
import yaml

from metrics import metrics_mapping
from utils import AverageMeter
from utils.image_book import image_book
from utils.progbar import progbar


# print = pprint

def dict_meter_value(meter_dict):
    value_dict = dict()
    for k, v in meter_dict.items():
        value_dict[k] = v.avg
    return value_dict


def eval_once(queue, id, batch_size=32):
    if not os.path.exists('results/{}/'.format(sys.argv[1])):
        os.makedirs('results/{}/'.format(sys.argv[1]))
    f = open("results/{}/{}.log".format(sys.argv[1], id), 'w')
    while True:
        try:
            i = queue.get_nowait()
            pid = os.getpid()
            gen_status[pid] = 1 if not pid in gen_status.keys() else gen_status[pid] + 1
            if not imgs[i].endswith('.png'):
                continue

            img_list = []
            for k in range(batch_size):
                array = cv2.imread(os.path.join(root, imgs[i * batch_size + k]))
                img_list.append(array)
            book = image_book(img_list, img_width, img_height, data_keys, data_format='channel_first', batch_size=32)
            f.write("{}@".format(imgs[i]))
            f.flush()
            for data_method in data_keys[data_begin:]:
                for mes in metrics:
                    meter_key = "{}_{}".format(data_method, mes)
                    keys = metric_config[mes]['keys']
                    data = book[(data_method, *keys)]
                    value = metrics_mapping[mes](data)
                    f.write("{}${};".format(meter_key, value))
                    f.flush()
                    meter_dict[meter_key].update(float(value))
                    # print(data_method, mes, value)
            f.write("\n")
            f.flush()
        except Exception as ex:
            print(ex)
            break


# def main():
config_file = 'configs/compare.yaml'
config = yaml.load(open(config_file, 'r'))
pprint(config)
root = "/mnt/cephfs_new_wj/lab_ad_idea/maoyiming/data/CCComparis_vis"
imgs = os.listdir(root)
img_width = config['data_format']['img_width']
img_height = config['data_format']['img_height']
data_keys = config['data_format']['keys']
data_begin = config['data_format']['data_begin_index']
metrics = config[sys.argv[1]]['metrics']
# print(metrics)
metric_config = yaml.load(open("configs/metrics.yaml"))['metrics']
# pprint(metric_config)
print("==> Do", len(metrics), "metrics in", len(data_keys) - data_begin, "methods")

# init avgmeter
meter_dict = mp.Manager().dict()
for data_method in data_keys[data_begin:]:
    for mes in metrics:
        meter_dict["{}_{}".format(data_method, mes)] = AverageMeter()

gen_status = mp.Manager().dict()
n_procs = 1
task_num = 8570
queue = mp.Queue()


# print(meter_dict.items())


def mp_run(batch_size):
    for i in range(task_num // batch_size):
        queue.put(i)

    mp_pools = []
    for k in range(n_procs):
        mp_t = mp.Process(target=eval_once, args=(queue, k,))
        mp_pools.append(mp_t)
        mp_t.start()

    my_bar = progbar(task_num, width=30)
    while True:
        sum_cnt = sum([gen_status[pid] for pid in gen_status.keys()])
        my_bar.update(sum_cnt)
        if sum_cnt == task_num:
            break


def main():
    batch_size = 10
    count = 0
    bar = progbar(len(imgs), width=30)
    for i in range(len(imgs)):
        if not imgs[i].endswith('.png'):
            continue
        # array = cv2.imread(os.path.join(root, imgs[i]))
        img_list = []
        for k in range(batch_size):
            array = cv2.imread(os.path.join(root, imgs[i * batch_size + k]))
            img_list.append(array)
        book = image_book(img_list, img_width, img_height, data_keys, data_format='channel_first', batch_size=32)
        for data_method in data_keys[data_begin:]:
            for mes in metrics:
                meter_key = "{}_{}".format(data_method, mes)
                keys = metric_config[mes]['keys']
                data = book[(data_method, *keys)]
                # print(len(data))
                value = metrics_mapping[mes](data)
                meter_dict[meter_key].update(float(value))
        count += 1
        bar.update(count, values=dict_meter_value(meter_dict).items())
        # print()


if __name__ == '__main__':
    main()
    # experiments_test = sys.argv[1]
    # metrics = config[experiments_test]['metrics']
    # batch_size = 4
    # mp_run(batch_size)
