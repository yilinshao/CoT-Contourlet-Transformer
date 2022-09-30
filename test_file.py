from torch import nn
import torch
from torch.nn.functional import one_hot

class SubNet_1(nn.Module):
    def __init__(self):
        super(SubNet_1, self).__init__()
        self.conv = None


class SubNet_2(nn.Module):
    def __init__(self):
        super(SubNet_2, self).__init__()
        self.conv = None


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
        self.net_1 = SubNet_1()
        self.net_2 = SubNet_2()

        self.net_1.conv = conv
        self.net_2.conv = conv

def main():
    feat = torch.rand(4, 5, 3, 3)
    gt = torch.ones(4, 1, 3, 3)
    one_hot_gt = one_hot(gt, num_classes=3)
    print(one_hot_gt)

if __name__ == '__main__':
    main()


