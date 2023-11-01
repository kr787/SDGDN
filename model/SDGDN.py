import torch
import torch.nn as nn
import torch.nn.functional as F


class DGCN(nn.Module):
    # in: dilation; out: residual; e.g. both 32
    def __init__(self, dim_in, dim_out, k, embed_dim, node_num, input_len, dropout_rate):
        super(DGCN, self).__init__()
        self.k = k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, self.k + 1, dim_in, dim_out))
        self.weights_pool2 = nn.Parameter(torch.FloatTensor(embed_dim, self.k + 1, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        self.bias_pool2 = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        self.fc0 = nn.Linear(input_len, 1)
        self.fc1 = nn.Linear(node_num, embed_dim)
        self.fc2 = nn.Linear(dim_in, embed_dim)

        self.node_num = node_num
        self.dropout_rate = dropout_rate

    def mlp(self, x):  # for DGL
        x = x[0]
        x = self.fc0(x)
        x = x.squeeze(-1)
        x = torch.tanh(x)
        x = self.fc1(x)
        x = torch.tanh(x)
        x = x.transpose(1, 0)
        x = self.fc2(x)
        x = torch.tanh(x)
        return x

    # x shaped[B, C, N, S], node_embeddings shaped [N, D] -> supports shaped [N, N], output shape [B, C, N, S]
    # x: H_l, x_gconv: h_l, support: A_l, node_embeddings: E^O, node_embeddings2: E^I, weights_pool: w_l, weights: W_l
    def forward(self, x, node_embeddings, node_embeddings2, transition=None):
        # DGL
        support = F.softmax(F.relu(torch.mm(torch.mm(node_embeddings, self.mlp(x)), node_embeddings2.transpose(0, 1))),
                            dim=1)
        support_set = [torch.eye(self.node_num).to(support.device)]
        for _ in range(0, self.k):  # default K = 2
            support_set.append(torch.mm(support_set[-1], support))
        supports = torch.stack(support_set, dim=0)

        # DGCN
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  # N, k, dim_in, dim_out
        weights += torch.einsum('nd,dkio->nkio', node_embeddings2, self.weights_pool2)  # N, k, dim_in, dim_out
        x_g = torch.einsum("knm,bims->bknis", supports, x)  # B, k, N, dim_in, S
        x_gconv = torch.einsum('bknis,nkio->bsno', x_g, weights)  # B, S, N, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)  # N, dim_out
        bias += torch.matmul(node_embeddings2, self.bias_pool2)
        x_gconv += bias
        x_gconv = x_gconv.permute(0, 3, 2, 1)  # B, dim_out, N, S
        x_gconv = F.dropout(x_gconv, self.dropout_rate, training=self.training)
        return x_gconv


class SDGDN(nn.Module):
    def __init__(self, args, dropout_rate=0.3, gcn_bool=True, addaptadj=True, in_dim=2, out_dim=12,
                 residual_channels=32, dilation_channels=32, skip_channels=256, end_channels=512,
                 kernel_size=2, blocks=4, layers=2):
        super(SDGDN, self).__init__()

        num_nodes = args.num_nodes
        self.step_per_hour = args.step_per_hour
        self.num_node = num_nodes
        if args.input_dim:
            in_dim = args.input_dim
        if args.output_window:
            out_dim = args.output_window

        if args.residual_channels:
            residual_channels = args.residual_channels

        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1))

        self.receptive_field = 1
        self.supports_len = 0
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)
        self.node_embeddings2 = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels, out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))
                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels, out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))
                # 1x1 convolution for skip connection
                self.skip_convs.append(
                    nn.Conv1d(in_channels=dilation_channels, out_channels=skip_channels, kernel_size=(1, 1)))
                new_dilation *= 2
                self.receptive_field += additional_scope
                additional_scope *= 2

        input_len = self.receptive_field
        for b in range(blocks):
            additional_scope = kernel_size - 1
            for i in range(layers):
                input_len -= additional_scope
                additional_scope *= 2
                # print(input_len)  # 12 10 9 7 6 4 3 1
                self.gconv.append(
                    DGCN(dim_in=dilation_channels, dim_out=residual_channels, k=args.k, embed_dim=args.embed_dim,
                         node_num=self.num_node, input_len=input_len, dropout_rate=dropout_rate))

        self.gconv = self.gconv[:-1]

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels, out_channels=end_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels, out_channels=out_dim, kernel_size=(1, 1), bias=True)

    def onehot(self, x):
        tod = F.one_hot(x[:, :, :, 1].long(), num_classes=24 * self.step_per_hour)
        dow = F.one_hot(x[:, :, :, 2].long(), num_classes=7)
        x = torch.cat((tod, dow), dim=-1) * x[:, :, :, :1]
        return x

    def forward(self, input, targets=None):
        input = self.onehot(input)
        input = input.transpose(1, 3)  # e.g. 64, 12, 307, 3 -> 64, 3, 307, 12
        in_len = input.size(3)  # e.g. 64, 3, 307, 13 : 13
        x = input

        # input layer
        if in_len < self.receptive_field:
            x = nn.functional.pad(input,
                                  (self.receptive_field - in_len, 0, 0, 0))  # e.g. 64, 3, 307, 12 -> 64, 3, 307, 13
        x = self.start_conv(x)  # e.g. 64, 3, 307, 13 -> 64, 32, 307, 13

        skip = 0
        layer_num = self.blocks * self.layers

        for i in range(layer_num):
            residual = x

            # G-TCN
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # skip connection
            s = x
            s = self.skip_convs[i](s)

            if i > 0:
                skip = skip[:, :, :, -s.size(3):]
            skip = s + skip

            # final spatial-temporal layer
            if i + 1 == layer_num:
                break

            # DGL & DGCN
            x = self.gconv[i](x, self.node_embeddings, self.node_embeddings2)

            # residual connection
            x = x + residual[:, :, :, -x.size(3):]

        # output layer
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x
