"""
    GraphCSPN: Geometry-Aware Depth Completion via Dynamic GCNs

    European Conference on Computer Vision (ECCV) 2022

    The code is based on https://github.com/zzangjinsun/NLSPN_ECCV20
"""


import numpy as np
import torch
import torch.nn as nn 
import torchvision



def conv_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0, bn=True,
                 relu=True):
    assert (kernel % 2) == 1, \
        'only odd kernel is supported but kernel = {}'.format(kernel)

    layers = []
    layers.append(nn.Conv2d(ch_in, ch_out, kernel, stride, padding,
                            bias=not bn))
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))

    layers = nn.Sequential(*layers)

    return layers


def convt_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0, output_padding=0,
                  bn=True, relu=True):
    assert (kernel % 2) == 1, \
        'only odd kernel is supported but kernel = {}'.format(kernel)

    layers = []
    layers.append(nn.ConvTranspose2d(ch_in, ch_out, kernel, stride, padding,
                                     output_padding, bias=not bn))
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))

    layers = nn.Sequential(*layers)

    return layers


def batched_index_select(x, idx):
    r"""fetches neighbors features from a given neighbor idx

    Args:
        x (Tensor): input feature Tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times C \times N \times 1}`.
        idx (Tensor): edge_idx
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times l}`.
    Returns:
        Tensor: output neighbors features
            :math:`\mathbf{X} \in \mathbb{R}^{B \times C \times N \times k}`.
    """
    batch_size, num_dims, num_vertices = x.shape[:3]
    k = idx.shape[-1]
    idx_base = torch.arange(0, batch_size, device=idx.device).view(-1, 1, 1) * num_vertices
    idx = idx + idx_base
    idx = idx.contiguous().view(-1)

    x = x.transpose(2, 1)
    feature = x.contiguous().view(batch_size * num_vertices, -1)[idx, :]
    feature = feature.view(batch_size, num_vertices, k, num_dims).permute(0, 3, 1, 2).contiguous()
    return feature


def pairwise_distance(x):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    x_inner = -2*torch.matmul(x, x.transpose(2, 1))
    x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
    return x_square + x_inner + x_square.transpose(2, 1)


def dense_knn_matrix(x, k=16):
    """Get KNN based on the pairwise distance.
    Args:
        x: (batch_size, num_dims, num_points, 1)
        k: int
    Returns:
        nearest neighbors: (batch_size, num_points ,k) (batch_size, num_points, k)
    """
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        _, nn_idx = torch.topk(-pairwise_distance(x.detach()), k=k)
        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0)


class KnnGraph(nn.Module):
    """
    Find the neighbors' indices based on dilated knn
    """
    def __init__(self, k=16):
        super(KnnGraph, self).__init__()
        self.k = k
        self.knn = dense_knn_matrix

    def forward(self, x):
        edge_index = self.knn(x, self.k)
        return edge_index


class EdgeAttn(nn.Module):
    """
    Edge attention layer
    """
    def __init__(self, in_channels, out_channels):
        super(EdgeAttn, self).__init__()
        self.edge = nn.Conv2d(in_channels*2, out_channels, 1, bias=True)
        self.att = nn.Conv2d(in_channels*2, out_channels, 1, bias=True)

    def forward(self, x, edge_index):
        x_i = batched_index_select(x, edge_index[1])
        x_j = batched_index_select(x, edge_index[0])

        x_edge = self.edge(torch.cat([x_i, x_j - x_i], dim=1))
        attn = self.att(torch.cat([x_i, x_j - x_i], dim=1))

        attn = nn.Softmax(dim=-1)(attn)

        x = x_edge * attn
        
        x = torch.sum(x, dim=-1, keepdim=True)

        return x


class GraphConv2d(nn.Module):
    """
    Static graph convolution layer
    """
    def __init__(self, in_channels, out_channels):
        super(GraphConv2d, self).__init__()
        self.gconv = EdgeAttn(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.gconv(x, edge_index)


class DynConv2d(GraphConv2d):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=16):
        super(DynConv2d, self).__init__(in_channels, out_channels)
        self.knn_graph = KnnGraph(kernel_size)

    def forward(self, x):
        edge_index = self.knn_graph(x)
        return super(DynConv2d, self).forward(x, edge_index)


class DynGCN(nn.Module):
    def __init__(self, in_channels=81, hid_channels=96, out_channels=9, kernel_size=16):
        super(DynGCN, self).__init__()
        self.in_channels = in_channels
        self.knn = KnnGraph(kernel_size)
        self.layer1 = GraphConv2d(in_channels, hid_channels)
        self.layer2 = DynConv2d(hid_channels, hid_channels, kernel_size)
        self.layer3 = DynConv2d(hid_channels, out_channels, kernel_size)


    def forward(self, x):
        edge_index = self.knn(x[:, 0:3])
        x = self.layer1(x[:, 3:self.in_channels+3], edge_index)
        x = self.layer2(x)
        x = self.layer3(x)

        return x


class Graph_Prop(nn.Module):

    def __init__(self):
        super(Graph_Prop, self).__init__()

        self.Graph_Prop_Block = DynGCN()
        self.guide_to_graph = nn.Conv2d(in_channels=81,
                                           out_channels=81,
                                           kernel_size=(9, 9),
                                           stride=3,
                                           groups=81,
                                           padding=(3, 4),
                                           bias=False)
        self.depth_to_graph = nn.Conv2d(in_channels=1,
                                           out_channels=1,
                                           kernel_size=(9, 9),
                                           stride=3,
                                           groups=1,
                                           padding=(3, 4),
                                           bias=False)
        self.graph_to_depth = nn.ConvTranspose2d(9, 9, kernel_size = 3, stride = 3, groups=9, padding = 0, bias=False)
        self.conv_sum = nn.Conv2d(in_channels=9,
                                       out_channels=1,
                                       kernel_size=(1, 1),
                                       stride=1,
                                       padding=0,
                                       bias=False)
        x_3d, y_3d = self.camera()
        self.x_3d = x_3d
        self.y_3d = y_3d

        self._ini_conv()

    def _ini_conv(self):
        weight = torch.zeros(81, 1, 9, 9)
        total = 0
        for i in range(9):
            for j in range(9):
                weight[total, 0, i, j] = 1
                total += 1

        weight_guidance_groups = torch.zeros(81, 1, 9, 9)
        for i in range(81):
            weight_guidance_groups[i, 0] = weight[i, 0]


        self.guide_to_graph.weight = nn.Parameter(weight_guidance_groups)

        for param in self.guide_to_graph.parameters():
            param.requires_grad = False

        weight_blur = torch.zeros(1, 1, 9, 9)
        weight_blur[0, 0] = weight[40, 0]


        self.depth_to_graph.weight = nn.Parameter(weight_blur)

        for param in self.depth_to_graph.parameters():
            param.requires_grad = False

        weight2 = torch.zeros(9, 1, 3, 3)
        total = 0
        for i in range(3):
            for j in range(3):
                weight2[total, 0, i, j] = 1
                total += 1

        up_weight_guidance_groups = torch.zeros(9, 1, 3, 3)
        for i in range(9):
            up_weight_guidance_groups[i, 0] = weight2[i, 0]

        self.graph_to_depth.weight = nn.Parameter(up_weight_guidance_groups)
        for param in self.graph_to_depth.parameters():
            param.requires_grad = False

        weight_sum = torch.ones(1, 9, 1, 1)

        self.conv_sum.weight = nn.Parameter(weight_sum)
        for param in self.conv_sum.parameters():
            param.requires_grad = False

    def camera(self):
        x_axis = np.arange(0, 102, 1)
        y_axis = np.arange(0, 76, 1)
        xx, yy = np.meshgrid(x_axis, y_axis)
        xx = torch.from_numpy(xx)
        yy = torch.from_numpy(yy)
        fx_d = 582.62448167737955/2.0
        fy_d = 582.69103270988637/2.0
        cx_d = 313.04475870804731/2.0
        cy_d = 238.44389626620386/2.0

        x_3d = (xx - cx_d)/fx_d
        y_3d = (yy - cy_d)/fy_d

        return x_3d, y_3d

    def forward(self, guidance, ini_depth, sparse_depth):
        sparse_mask = sparse_depth.sign()
        ini_depth =  (1 - sparse_mask) * ini_depth + sparse_mask * sparse_depth

        ini_depth = self.depth_to_graph(ini_depth)

        x_3d = self.x_3d.to(ini_depth)
        y_3d = self.y_3d.to(ini_depth)
        x_3d = x_3d * ini_depth * 3
        y_3d = y_3d * ini_depth * 3

        loc = torch.cat([x_3d, y_3d, ini_depth], dim=1)

        x = self.guide_to_graph(guidance)
        x = torch.cat([loc, x], dim=1)
        width = x.shape[2]
        height = x.shape[3]

        x = x.view(x.shape[0], x.shape[1], width*height, 1)
        x = self.Graph_Prop_Block(x)
        x = x.view(x.shape[0], x.shape[1], width, height)

        x = self.graph_to_depth(x)
        x = self.conv_sum(x)

        x = x.narrow(3, 1, sparse_depth.shape[-1])

        return x


class GraphCSPN(nn.Module):
    def __init__(self, args):
        super(GraphCSPN, self).__init__()

        self.args = args

        # Encoder
        self.conv1_rgb = conv_bn_relu(3, 48, kernel=3, stride=1, padding=1,
                                      bn=False)
        self.conv1_dep = conv_bn_relu(1, 16, kernel=3, stride=1, padding=1,
                                      bn=False)

        net = torchvision.models.resnet34(pretrained=True)

        # 1/1
        self.conv2 = net.layer1
        # 1/2
        self.conv3 = net.layer2
        # 1/4
        self.conv4 = net.layer3
        # 1/8
        self.conv5 = net.layer4

        del net

        # 1/16
        self.conv6 = conv_bn_relu(512, 512, kernel=3, stride=2, padding=1)

        # Shared Decoder
        # 1/8
        self.dec5 = convt_bn_relu(512, 256, kernel=3, stride=2,
                                  padding=1, output_padding=1)
        # 1/4
        self.dec4 = convt_bn_relu(256+512, 128, kernel=3, stride=2,
                                  padding=1, output_padding=1)
        # 1/2
        self.dec3 = convt_bn_relu(128+256, 64, kernel=3, stride=2,
                                  padding=1, output_padding=1)

        # 1/1
        self.dec2 = convt_bn_relu(64+128, 64, kernel=3, stride=2,
                                  padding=1, output_padding=1)

        # Init Depth Branch
        # 1/1
        self.id_dec1 = conv_bn_relu(64+64, 64, kernel=3, stride=1,
                                    padding=1)
        self.id_dec0 = conv_bn_relu(64+64, 1, kernel=3, stride=1,
                                    padding=1, bn=False, relu=True)

        # Guidance Branch
        # 1/1
        self.gd_dec1 = conv_bn_relu(64+64, 64, kernel=3, stride=1,
                                    padding=1)
        self.gd_dec0 = conv_bn_relu(64+64, 81, kernel=3, stride=1,
                                    padding=1, bn=False, relu=False)

        # Graph propagation layer
        self.graph_prop = Graph_Prop()

    def _concat(self, fd, fe, dim=1):
        # Decoder feature may have additional padding
        _, _, Hd, Wd = fd.shape
        _, _, He, We = fe.shape

        # Remove additional padding
        if Hd > He:
            h = Hd - He
            fd = fd[:, :, :-h, :]

        if Wd > We:
            w = Wd - We
            fd = fd[:, :, :, :-w]

        f = torch.cat((fd, fe), dim=dim)

        return f

    def forward(self, sample):
        rgb = sample['rgb']
        dep = sample['dep']

        # Encoding
        fe1_rgb = self.conv1_rgb(rgb)
        fe1_dep = self.conv1_dep(dep)

        fe1 = torch.cat((fe1_rgb, fe1_dep), dim=1)

        fe2 = self.conv2(fe1)
        fe3 = self.conv3(fe2)
        fe4 = self.conv4(fe3)
        fe5 = self.conv5(fe4)
        fe6 = self.conv6(fe5)

        # Shared Decoding
        fd5 = self.dec5(fe6)
        fd4 = self.dec4(self._concat(fd5, fe5))
        fd3 = self.dec3(self._concat(fd4, fe4))
        fd2 = self.dec2(self._concat(fd3, fe3))

        # Init Depth Decoding
        id_fd1 = self.id_dec1(self._concat(fd2, fe2))
        pred_init = self.id_dec0(self._concat(id_fd1, fe1))

        # Guidance Decoding
        gd_fd1 = self.gd_dec1(self._concat(fd2, fe2))
        guide = self.gd_dec0(self._concat(gd_fd1, fe1))


        # Diffusion
        y = self.graph_prop(guide, pred_init, dep)

        # Remove negative depth
        output = torch.clamp(y, min=0)

        return output
