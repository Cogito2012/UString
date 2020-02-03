from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
from torch.nn.parameter import Parameter
import torch
import torch.nn as nn
from src.utils import glorot, zeros, uniform, reset
from torch_geometric.utils import remove_self_loops, add_self_loops
import torch_scatter
from torch_scatter import scatter_mean, scatter_max, scatter_add
from torch.autograd import Variable
import torch.nn.functional as F


class MessagePassing(torch.nn.Module):
    r"""Base class for creating message passing layers
    .. math::
        \mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left( \mathbf{x}_i,
        \square_{j \in \mathcal{N}(i)} \, \phi_{\mathbf{\Theta}}
        \left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{i,j}\right) \right),
    where :math:`\square` denotes a differentiable, permutation invariant
    function, *e.g.*, sum, mean or max, and :math:`\gamma_{\mathbf{\Theta}}`
    and :math:`\phi_{\mathbf{\Theta}}` denote differentiable functions such as
    MLPs.
    See `here <https://rusty1s.github.io/pytorch_geometric/build/html/notes/
    create_gnn.html>`__ for the accompanying tutorial.
    """

    def __init__(self, aggr='add'):
        super(MessagePassing, self).__init__()

        self.message_args = inspect.getargspec(self.message)[0][1:]
        self.update_args = inspect.getargspec(self.update)[0][2:]

    def propagate(self, aggr, edge_index, **kwargs):
        r"""The initial call to start propagating messages.
        Takes in an aggregation scheme (:obj:`"add"`, :obj:`"mean"` or
        :obj:`"max"`), the edge indices, and all additional data which is
        needed to construct messages and to update node embeddings."""

        assert aggr in ['add', 'mean', 'max']
        kwargs['edge_index'] = edge_index

        size = None
        message_args = []
        for arg in self.message_args:
            if arg[-2:] == '_i':
                tmp = kwargs[arg[:-2]]
                size = tmp.size(0)
                message_args.append(tmp[edge_index[0]])
            elif arg[-2:] == '_j':
                tmp = kwargs[arg[:-2]]
                size = tmp.size(0)
                message_args.append(tmp[edge_index[1]])
            else:
                message_args.append(kwargs[arg])

        update_args = [kwargs[arg] for arg in self.update_args]

        out = self.message(*message_args)
        out = scatter_(aggr, out, edge_index[0], dim_size=size)
        out = self.update(out, *update_args)

        return out

    def message(self, x_j):  # pragma: no cover
        r"""Constructs messages in analogy to :math:`\phi_{\mathbf{\Theta}}`
        for each edge in :math:`(i,j) \in \mathcal{E}`.
        Can take any argument which was initially passed to :meth:`propagate`.
        In addition, features can be lifted to the source node :math:`i` and
        target node :math:`j` by appending :obj:`_i` or :obj:`_j` to the
        variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`."""

        return x_j

    def update(self, aggr_out):  # pragma: no cover
        r"""Updates node embeddings in analogy to
        :math:`\gamma_{\mathbf{\Theta}}` for each node
        :math:`i \in \mathcal{V}`.
        Takes in the output of aggregation as first argument and any argument
        which was initially passed to :meth:`propagate`."""

        return aggr_out



def scatter_(name, src, index, dim_size=None):
    r"""Aggregates all values from the :attr:`src` tensor at the indices
    specified in the :attr:`index` tensor along the first dimension.
    If multiple indices reference the same location, their contributions
    are aggregated according to :attr:`name` (either :obj:`"add"`,
    :obj:`"mean"` or :obj:`"max"`).
    Args:
        name (string): The aggregation to use (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"max"`).
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements to scatter.
        dim_size (int, optional): Automatically create output tensor with size
            :attr:`dim_size` in the first dimension. If set to :attr:`None`, a
            minimal sized output tensor is returned. (default: :obj:`None`)
    :rtype: :class:`Tensor`
    """

    assert name in ['add', 'mean', 'max']

    op = getattr(torch_scatter, 'scatter_{}'.format(name))
    fill_value = -1e38 if name is 'max' else 0

    out = op(src, index, 0, None, dim_size, fill_value)
    if isinstance(out, tuple):
        out = out[0]

    if name is 'max':
        out[out == fill_value] = 0

    return out


# layers

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, act=F.relu, improved=True, bias=False):
        super(GCNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.act = act

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def add_self_loops(self, edge_index, edge_weight=None, fill_value=1, num_nodes=None):
        """

        :param edge_index: 10 x 2 x 171
        :param edge_weight: 10 x 171
        :param fill_value: 1
        :param num_nodes: 20
        :return:
        """
        batch_size = edge_index.size(0)
        num_nodes = edge_index.max().item() + 1 if num_nodes is None else num_nodes
        loop_index = torch.arange(0, num_nodes, dtype=torch.long,
                                  device=edge_index.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        loop_index = loop_index.unsqueeze(0).repeat(batch_size, 1, 1)  # 10 x 2 x 20

        if edge_weight is not None:
            assert edge_weight.size(-1) == edge_index.size(-1)
            loop_weight = edge_weight.new_full((num_nodes,), fill_value)
            loop_weight = loop_weight.unsqueeze(0).repeat(batch_size, 1)
            edge_weight = torch.cat([edge_weight, loop_weight], dim=-1)

        edge_index = torch.cat([edge_index, loop_index], dim=-1)

        return edge_index, edge_weight


    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            edge_weight = torch.ones(
                (edge_index.size(0), edge_index.size(-1), ), dtype=x.dtype, device=x.device)
        # edge_weight = edge_weight.view(-1)
        assert edge_weight.size(-1) == edge_index.size(-1)

        # for pytorch 1.4, there are two outputs
        edge_index, edge_weight = self.add_self_loops(edge_index, edge_weight=edge_weight, num_nodes=x.size(1))

        out_batch = []
        for i in range(edge_index.size(0)):
            row, col = edge_index[i]
            deg = scatter_add(edge_weight[i], row, dim=0, dim_size=x.size(1))
            deg_inv = deg.pow(-0.5)
            deg_inv[deg_inv == float('inf')] = 0

            norm = deg_inv[row] * edge_weight[i] * deg_inv[col]
            
            weight = self.weight.to(x.device)
            x_w = torch.matmul(x[i], weight)
            out = self.propagate('add', edge_index[i], x=x_w, norm=norm)
            out_batch.append(self.act(out))
        out_batch = torch.stack(out_batch)

        return out_batch

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            bias = self.bias.to(aggr_out.device)
            aggr_out = aggr_out + bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class SAGEConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, pool='mean', act=F.relu, normalize=False, bias=False):
        super(SAGEConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))
        self.act = act
        self.pool = pool

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.size(0)
        uniform(size, self.weight)
        uniform(size, self.bias)

    def forward(self, x, edge_index):
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        x = x.unsqueeze(-1) if x.dim() == 1 else x
        row, col = edge_index

        if self.pool == 'mean':
            out = torch.matmul(x, self.weight)
            if self.bias is not None:
                out = out + self.bias
            out = self.act(out)
            out = scatter_mean(out[col], row, dim=0, dim_size=out.size(0))

        elif self.pool == 'max':
            out = torch.matmul(x, self.weight)
            if self.bias is not None:
                out = out + self.bias
            out = self.act(out)
            out, _ = scatter_max(out[col], row, dim=0, dim_size=out.size(0))

        elif self.pool == 'add':
            x = torch.matmul(x, self.weight)
            if self.bias is not None:
                out = out + self.bias
            out = self.act(out)
            out = scatter_add(x[col], row, dim=0, dim_size=x.size(0))
        else:
            print('pooling not defined!')

        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)

        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


class GINConv(torch.nn.Module):
    def __init__(self, nn, eps=0, train_eps=False):
        super(GINConv, self).__init__()
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_index, _ = remove_self_loops(edge_index)
        row, col = edge_index

        out = scatter_add(x[col], row, dim=0, dim_size=x.size(0))
        out = (1 + self.eps) * x + out
        out = self.nn(out)
        return out

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class graph_gru_sage(nn.Module):
    def __init__(self, input_size, hidden_size, n_layer, bias=True):
        super(graph_gru_sage, self).__init__()

        self.hidden_size = hidden_size
        self.n_layer = n_layer

        # gru weights
        self.weight_xz = []
        self.weight_hz = []
        self.weight_xr = []
        self.weight_hr = []
        self.weight_xh = []
        self.weight_hh = []

        for i in range(self.n_layer):
            if i== 0:
                self.weight_xz.append(SAGEConv(input_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hz.append(SAGEConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_xr.append(SAGEConv(input_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hr.append(SAGEConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_xh.append(SAGEConv(input_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hh.append(SAGEConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
            else:
                self.weight_xz.append(SAGEConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hz.append(SAGEConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_xr.append(SAGEConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hr.append(SAGEConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_xh.append(SAGEConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hh.append(SAGEConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))

    def forward(self, inp, edgidx, h):
        h_out = torch.zeros(h.size())
        for i in range(self.n_layer):
            if i == 0:
                z_g = torch.sigmoid(self.weight_xz[i](inp, edgidx) + self.weight_hz[i](h[i], edgidx))
                r_g = torch.sigmoid(self.weight_xr[i](inp, edgidx) + self.weight_hr[i](h[i], edgidx))
                h_tilde_g = torch.tanh(self.weight_xh[i](inp, edgidx) + self.weight_hh[i](r_g * h[i], edgidx))
                h_out[i] = z_g * h[i] + (1 - z_g) * h_tilde_g
            #         out = self.decoder(h_t.view(1,-1))
            else:
                z_g = torch.sigmoid(self.weight_xz[i](h_out[i - 1], edgidx) + self.weight_hz[i](h[i], edgidx))
                r_g = torch.sigmoid(self.weight_xr[i](h_out[i - 1], edgidx) + self.weight_hr[i](h[i], edgidx))
                h_tilde_g = torch.tanh(self.weight_xh[i](h_out[i - 1], edgidx) + self.weight_hh[i](r_g * h[i], edgidx))
                h_out[i] = z_g * h[i] + (1 - z_g) * h_tilde_g
        #         out = self.decoder(h_t.view(1,-1))

        out = h_out
        return out, h_out


class graph_gru_gcn(nn.Module):
    def __init__(self, input_size, hidden_size, n_layer, bias=True):
        super(graph_gru_gcn, self).__init__()

        self.hidden_size = hidden_size
        self.n_layer = n_layer

        # gru weights
        self.weight_xz = []
        self.weight_hz = []
        self.weight_xr = []
        self.weight_hr = []
        self.weight_xh = []
        self.weight_hh = []

        for i in range(self.n_layer):
            if i == 0:
                self.weight_xz.append(GCNConv(input_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hz.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_xr.append(GCNConv(input_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hr.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_xh.append(GCNConv(input_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hh.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
            else:
                self.weight_xz.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hz.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_xr.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hr.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_xh.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hh.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))

    def forward(self, inp, edgidx, h):
        h_out = torch.zeros(h.size())
        h_out = h_out.to(inp.device)
        for i in range(self.n_layer):
            if i == 0:
                z_g = torch.sigmoid(self.weight_xz[i](inp, edgidx) + self.weight_hz[i](h[i], edgidx))
                r_g = torch.sigmoid(self.weight_xr[i](inp, edgidx) + self.weight_hr[i](h[i], edgidx))
                h_tilde_g = torch.tanh(self.weight_xh[i](inp, edgidx) + self.weight_hh[i](r_g * h[i], edgidx))
                h_out[i] = z_g * h[i] + (1 - z_g) * h_tilde_g
            #         out = self.decoder(h_t.view(1,-1))
            else:
                z_g = torch.sigmoid(self.weight_xz[i](h_out[i - 1], edgidx) + self.weight_hz[i](h[i], edgidx))
                r_g = torch.sigmoid(self.weight_xr[i](h_out[i - 1], edgidx) + self.weight_hr[i](h[i], edgidx))
                h_tilde_g = torch.tanh(self.weight_xh[i](h_out[i - 1], edgidx) + self.weight_hh[i](r_g * h[i], edgidx))
                h_out[i] = z_g * h[i] + (1 - z_g) * h_tilde_g
        #         out = self.decoder(h_t.view(1,-1))

        out = h_out
        return out, h_out


class InnerProductDecoder(nn.Module):
    def __init__(self, act=torch.sigmoid, dropout=0.):
        super(InnerProductDecoder, self).__init__()

        self.act = act
        self.dropout = dropout

    def forward(self, inp):
        inp = F.dropout(inp, self.dropout, training=self.training)
        x = torch.transpose(inp, dim0=0, dim1=1)
        x = torch.mm(inp, x)
        return self.act(x)

class AccidentPredictor(nn.Module):
    def __init__(self, input_dim, output_dim=2, act=torch.relu, dropout=0.):
        super(AccidentPredictor, self).__init__()
		
        self.act = act
        self.dropout = dropout
        self.dense1 = torch.nn.Linear(input_dim, 16)
        self.dense2 = torch.nn.Linear(16, output_dim)
        
        self.reset_parameters(stdv=1e-2)
	
    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.act(self.dense1(x))
        x = self.dense2(x)

        return x
    
    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)

# VGRNN model

class VGRNN(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layers, n_obj=20, eps=1e-10, conv='GCN', bias=False, loss_func='exp'):
        super(VGRNN, self).__init__()

        self.x_dim = x_dim
        self.eps = eps
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.loss_func = loss_func
        self.n_obj = n_obj

        if conv == 'GCN':
            self.phi_x = nn.Sequential(nn.Linear(x_dim, h_dim), nn.ReLU())
            self.phi_z = nn.Sequential(nn.Linear(z_dim, h_dim), nn.ReLU())

            self.enc = GCNConv(h_dim + h_dim, h_dim)
            self.enc_mean = GCNConv(h_dim, z_dim, act=lambda x: x)
            self.enc_std = GCNConv(h_dim, z_dim, act=F.softplus)

            self.prior = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU())
            self.prior_mean = nn.Sequential(nn.Linear(h_dim, z_dim))
            self.prior_std = nn.Sequential(nn.Linear(h_dim, z_dim), nn.Softplus())

            self.rnn = graph_gru_gcn(h_dim + h_dim, h_dim, n_layers, bias)

            self.predictor = AccidentPredictor(n_obj * z_dim, 2)
            self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')

        elif conv == 'SAGE':
            self.phi_x = nn.Sequential(nn.Linear(x_dim, h_dim), nn.ReLU())
            self.phi_z = nn.Sequential(nn.Linear(z_dim, h_dim), nn.ReLU())

            self.enc = SAGEConv(h_dim + h_dim, h_dim)
            self.enc_mean = SAGEConv(h_dim, z_dim, act=lambda x: x)
            self.enc_std = SAGEConv(h_dim, z_dim, act=F.softplus)

            self.prior = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU())
            self.prior_mean = nn.Sequential(nn.Linear(h_dim, z_dim))
            self.prior_std = nn.Sequential(nn.Linear(h_dim, z_dim), nn.Softplus())

            self.rnn = graph_gru_sage(h_dim + h_dim, h_dim, n_layers, bias)

        elif conv == 'GIN':
            self.phi_x = nn.Sequential(nn.Linear(x_dim, h_dim), nn.ReLU())
            self.phi_z = nn.Sequential(nn.Linear(z_dim, h_dim), nn.ReLU())

            self.enc = GINConv(nn.Sequential(nn.Linear(h_dim + h_dim, h_dim), nn.ReLU()))
            self.enc_mean = GINConv(nn.Sequential(nn.Linear(h_dim, z_dim)))
            self.enc_std = GINConv(nn.Sequential(nn.Linear(h_dim, z_dim), nn.Softplus()))

            self.prior = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU())
            self.prior_mean = nn.Sequential(nn.Linear(h_dim, z_dim))
            self.prior_std = nn.Sequential(nn.Linear(h_dim, z_dim), nn.Softplus())

            self.rnn = graph_gru_gcn(h_dim + h_dim, h_dim, n_layers, bias)



    def forward(self, x, y, edge_idx_list, hidden_in=None, edge_weights=None):
        """
        :param x, (batchsize, nFrames, nBoxes, Xdim)
        """
        kld_loss = 0
        acc_loss = 0
        all_dec, all_prior_mean = [], []

        # import ipdb; ipdb.set_trace()
        if hidden_in is None:
            h = Variable(torch.zeros(self.n_layers, x.size(0), x.size(2), self.h_dim))  # 1 x 10 x 20 x 32
        else:
            h = Variable(hidden_in)
        h = h.to(x.device)

        for t in range(x.size(1)):
            phi_x_t = self.phi_x(x[:, t])  # 10 x 20 x 32

            # encoder
            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], -1), edge_idx_list[:, t], edge_weight=edge_weights[:, t])
            enc_mean_t = self.enc_mean(enc_t, edge_idx_list[:, t])
            enc_std_t = self.enc_std(enc_t, edge_idx_list[:, t])

            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # sampling and reparameterization
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)  # B x N x dz
            phi_z_t = self.phi_z(z_t)
            
            # decoder
            dec_t = self.predictor(z_t.view(z_t.size(0), -1))  # B x 2

            # recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 2), edge_idx_list[:, t], h)
	
            # computing losses
            kld_loss += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
            if self.loss_func == 'exp':
                acc_loss += self._exp_loss(dec_t, y, t)
            elif self.loss_func == 'bernoulli':
                acc_loss += self._nll_bernoulli(dec_t, y)


            all_dec.append(dec_t)
            all_prior_mean.append(prior_mean_t)

        return kld_loss, acc_loss, all_dec, all_prior_mean, h


    def dec(self, z):
        outputs = []
        for i in range(z.size(0)):
            dec = InnerProductDecoder(act=lambda x: x)(z[i])
            outputs.append(dec)
        outputs = torch.stack(outputs)
        return outputs

    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)

    def _init_weights(self, stdv):
        pass

    def _reparameterized_sample(self, mean, std):
        eps1 = torch.FloatTensor(std.size()).normal_()
        eps1 = Variable(eps1).to(mean.device)
        return eps1.mul(std).add_(mean)

    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        # shape: [10, 20, 16]
        batch_size = mean_1.size()[0]
        num_nodes = mean_1.size()[1]
        kld_element = (2 * torch.log(std_2 + self.eps) - 2 * torch.log(std_1 + self.eps) +
                       (torch.pow(std_1 + self.eps, 2) + torch.pow(mean_1 - mean_2, 2)) /
                       torch.pow(std_2 + self.eps, 2) - 1)
        return torch.mean((0.5 / num_nodes) * torch.mean(torch.sum(kld_element, dim=2), dim=1), dim=0)

    def _kld_gauss_zu(self, mean_in, std_in):
        num_nodes = mean_in.size()[0]
        std_log = torch.log(std_in + self.eps)
        kld_element = torch.mean(torch.sum(1 + 2 * std_log - mean_in.pow(2) -
                                           torch.pow(torch.exp(std_log), 2), 1))
        return (-0.5 / num_nodes) * kld_element

    def _nll_bernoulli(self, logits, target_adj_dense):
        temp_size = target_adj_dense.size()[0]
        temp_sum = target_adj_dense.sum()
        posw = float(temp_size * temp_size - temp_sum) / temp_sum
        norm = temp_size * temp_size / float((temp_size * temp_size - temp_sum) * 2)
        nll_loss_mat = F.binary_cross_entropy_with_logits(input=logits
                                                          , target=target_adj_dense
                                                          , pos_weight=posw
                                                          , reduction='none')
        nll_loss = -1 * norm * torch.mean(nll_loss_mat, dim=[0, 1])
        return - nll_loss

    def _exp_loss(self, pred, target, time, frames=100, fps=20.0):
        '''
        :param pred:
        :param target: onehot codings for binary classification
        :param time:
        :param frames:
        :param fps:
        :return:
        '''
        # positive example (exp_loss)
        target_cls = target[:, 1]
        pos_loss = -torch.mul(torch.exp(-torch.tensor((frames - time - 1) / fps)),
                              -self.ce_loss(pred, target_cls.to(torch.long)))
        # negative example
        neg_loss = self.ce_loss(pred, target_cls.to(torch.long))

        loss = torch.mean(torch.add(torch.mul(pos_loss, target[:, 1]), torch.mul(neg_loss, target[:, 0])))
        return loss


