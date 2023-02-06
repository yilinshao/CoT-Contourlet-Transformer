import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib.image as mpimg



def vis_hidden_layer(bs_feature):
    import matplotlib.pyplot as plt
    for n, feature in enumerate(bs_feature):
        layer_for_vis = torch.sum(feature, dim=0, keepdim=False)
        # layer_for_vis = feature[0]

        # norm
        layer_for_vis = layer_for_vis.detach().cpu().numpy()
        layer_for_vis = (layer_for_vis - layer_for_vis.min()) / (layer_for_vis.max() - layer_for_vis.min())

        # plt.imshow(numpy.sqrt(layer_for_vis))
        plt.imshow(layer_for_vis)
        plt.show()



def only_show_deficient_points(deficient_points_maps):
    deficient_points_maps = deficient_points_maps.detach().cpu().numpy()
    import matplotlib.pyplot as plt

    bs = deficient_points_maps.shape[0]
    for n in range(bs):
        deficient_points_map = deficient_points_maps[n].transpose(1, 2, 0).squeeze(2)
        decifient_point_coords = np.where(deficient_points_map == 1)

        plt.scatter(decifient_point_coords[1], decifient_point_coords[0], s=1.0, marker='o', c='yellow')
        plt.show()
from PIL import Image
img = Image.open('/data/amazing/CoT-V2-Remastered/old_hidden_layer_result_images/ADE_train_00010147.jpg')
img = np.asarray(img)
print(img)



