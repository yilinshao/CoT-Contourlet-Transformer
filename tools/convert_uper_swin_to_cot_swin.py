import torch
from mmcv.runner import CheckpointLoader
from collections import OrderedDict

def convert_uper_swin(ckpt):
    new_ckpt = OrderedDict()
    for k, v in ckpt.items():
        print(k)
        if 'decode_head.' in k:
            new_k = k.replace('decode_head.', 'neck.')
            new_v = v
            print('convert {} to {}'.format(k, new_k))
        else:
            new_k = k
            new_v = v

        new_ckpt[new_k] = new_v

    return new_ckpt

def main():
    checkpoint = CheckpointLoader.load_checkpoint('pretrain/upernet_swin_large_patch4_window12_512x512_pretrain_384x384_22K_160k_ade20k.pth',
                                                  map_location='cpu')

    state_dict = checkpoint['state_dict']
    weight = convert_uper_swin(state_dict)
    # torch.save(weight, 'pretrain/converted_upernet_swin_large_patch4_window12_512x512_pretrain_384x384_22K_160k_ade20k.pth')


if __name__ == '__main__':
    main()