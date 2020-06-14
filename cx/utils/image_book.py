# -*- coding: utf-8 -*-
"""
@Project:   metrics_bundle
@File   :   img_book
@Author :   TonyMao@AILab
@Date   :   2019-10-25
@Desc   :   None
"""
import numpy as np
import torchvision.transforms as transforms


class image_book:
    def __init__(self, array_list, img_w, img_h, keys, format="horiz", data_format='channel_first', batch_size=32):
        self.image_array = np.stack(array_list, axis=0)
        if self.image_array.ndim == 3:
            self.image_array = self.image_array[:, :, None]
        elif self.image_array.ndim == 2:
            self.image_array = self.image_array[None, :, :, None]
        if data_format == 'channel_first':
            self.image_array = self.image_array.transpose([0, -1, 1, 2])
        self.w = img_w
        self.h = img_h
        self.keys = keys
        self.data_format = data_format
        self.batch_size = batch_size

    def __getitem__(self, keys):
        arr = []
        for k in keys:
            index = self.keys.index(k)
            img = self.image_array[:, :, :, self.w * index:self.w * (1 + index)]
            arr.append([img[j, :, :, :] for j in range(img.shape[0])])
        # for i in arr[0]:
        #     print(i.shape)
        return arr


def main():
    pass


if __name__ == '__main__':
    main()
