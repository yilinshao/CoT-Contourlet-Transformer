from torch import nn

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
    net = Net()
    net_1 = net.net_1
    net_2 = net.net_2

    print(id(net_1.conv) == id(net_2.conv))

if __name__ == '__main__':
    main()


