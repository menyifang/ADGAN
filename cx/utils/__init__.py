# -*- coding: utf-8 -*-
"""
@Project:   metrics_bundle
@File   :   __init__.py
@Author :   TonyMao@AILab
@Date   :   2019-10-25
@Desc   :   None
"""
__all__ = ['AverageMeter', 'progbar', 'image_book']


def main():
    pass


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def __str__(self):
        return str(self.avg)

    def __repr__(self):
        return str(self.avg)

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
