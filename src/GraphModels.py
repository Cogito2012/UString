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
from src.BayesModels import BayesianLinear

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

    def forward(self, inp, edgidx, h, edge_weight=None):
        h_out = torch.zeros(h.size())
        h_out = h_out.to(inp.device)
        for i in range(self.n_layer):
            if i == 0:
                z_g = torch.sigmoid(self.weight_xz[i](inp, edgidx, edge_weight) + self.weight_hz[i](h[i], edgidx, edge_weight))
                r_g = torch.sigmoid(self.weight_xr[i](inp, edgidx, edge_weight) + self.weight_hr[i](h[i], edgidx, edge_weight))
                h_tilde_g = torch.tanh(self.weight_xh[i](inp, edgidx, edge_weight) + self.weight_hh[i](r_g * h[i], edgidx, edge_weight))
                h_out[i] = z_g * h[i] + (1 - z_g) * h_tilde_g
            else:
                z_g = torch.sigmoid(self.weight_xz[i](h_out[i - 1], edgidx, edge_weight) + self.weight_hz[i](h[i], edgidx, edge_weight))
                r_g = torch.sigmoid(self.weight_xr[i](h_out[i - 1], edgidx, edge_weight) + self.weight_hr[i](h[i], edgidx, edge_weight))
                h_tilde_g = torch.tanh(self.weight_xh[i](h_out[i - 1], edgidx, edge_weight) + self.weight_hh[i](r_g * h[i], edgidx, edge_weight))
                h_out[i] = z_g * h[i] + (1 - z_g) * h_tilde_g
        return h_out


class AccidentPredictor(nn.Module):
    def __init__(self, input_dim, output_dim=2, act=torch.relu, dropout=0.):
        super(AccidentPredictor, self).__init__()
		
        self.act = act
        self.dropout = dropout
        self.dense1 = torch.nn.Linear(input_dim, 32)
        self.dense2 = torch.nn.Linear(32, output_dim)

        self.reset_parameters(stdv=1e-2)

	
    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.act(self.dense1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.dense2(x)

        return x
    
    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)


class BayesianPredictor(nn.Module):
    def __init__(self, input_dim, output_dim=2, act=torch.relu, pi=0.5, sigma_1=None, sigma_2=None):
        super(BayesianPredictor, self).__init__()
        self.act = act
        self.l1 = BayesianLinear(input_dim, 32, pi=pi, sigma_1=sigma_1, sigma_2=sigma_2)
        self.l2 = BayesianLinear(32, output_dim, pi=pi, sigma_1=sigma_1, sigma_2=sigma_2)

    def forward(self, x, sample=False):
        x = self.act(self.l1(x, sample))
        x = self.l2(x, sample)
        return x

    def log_prior(self):
        return self.l1.log_prior + self.l2.log_prior

    def log_variational_posterior(self):
        return self.l1.log_variational_posterior + self.l2.log_variational_posterior

    def sample_elbo(self, input, out_dim=2, npass=2, testing=False):
        npass = npass + 1 if testing else npass
        outputs = torch.zeros(npass, input.size(0), out_dim).to(input.device)
        log_priors = torch.zeros(npass).to(input.device)
        log_variational_posteriors = torch.zeros(npass).to(input.device)
        for i in range(npass):
            outputs[i] = self(input, sample=True)
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()
        if testing:
            outputs[npass] = self(input, sample=False)
        output = outputs.mean(0)
        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posteriors.mean()

        return output, log_prior, log_variational_posterior


# GCRNN model

class GCRNN(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layers, n_obj=19, n_frames=100, eps=1e-10, use_hidden=False):
        super(GCRNN, self).__init__()

        self.x_dim = x_dim
        self.eps = eps
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.use_hidden = use_hidden
        self.n_obj = n_obj
        self.n_frames = n_frames

        self.phi_x = nn.Sequential(nn.Linear(x_dim, h_dim), nn.ReLU())

        self.enc_gcn1 = GCNConv(h_dim + h_dim, h_dim)
        self.enc_gcn2 = GCNConv(h_dim + h_dim, z_dim, act=lambda x: x)

        self.rnn = graph_gru_gcn(h_dim + h_dim + z_dim, h_dim, n_layers, bias=True)

        dim_encode = z_dim + h_dim if use_hidden else z_dim
        self.predictor = AccidentPredictor(n_obj * dim_encode, 2, dropout=0.5)
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')

        self.soft_aggregation = SoftAggregate(self.n_frames)
        self.predictor_aux = AccidentPredictor(h_dim + h_dim, 2, dropout=0.1)

        self.reset_parameters(stdv=1e-2)


    def forward(self, x, y, edge_idx, hidden_in=None, edge_weights=None):
        """
        :param x, (batchsize, nFrames, nBoxes, Xdim) = (10, 100, 20, 4096)
        """
        acc_loss = 0
        all_dec, all_hidden = [], []

        # import ipdb; ipdb.set_trace()
        if hidden_in is None:
            h = Variable(torch.zeros(self.n_layers, x.size(0), self.n_obj, self.h_dim))  # 1 x 10 x 19 x 256
        else:
            h = Variable(hidden_in)
        h = h.to(x.device)

        for t in range(x.size(1)):
            # reduce the dim of node feature (FC layer)
            x_t = self.phi_x(x[:, t])  # 10 x 20 x 256
            img_embed = x_t[:, 0, :].unsqueeze(1).repeat(1, self.n_obj, 1).contiguous()  # 10 x 19 x 256
            x_t = torch.cat([x_t[:, 1:, :], img_embed], dim=-1)  # 10 x 19 x 512

            # GCN encoder
            enc = self.enc_gcn1(x_t, edge_idx[:, t], edge_weight=edge_weights[:, t])  # 10 x 19 x 256 (512-->256)
            z_t = self.enc_gcn2(torch.cat([enc, h[-1]], -1), edge_idx[:, t], edge_weight=edge_weights[:, t])  # 10 x 19 x 256 (512-->256)
            
            # decoder
            if self.use_hidden:
                embed = torch.cat([z_t, h[-1]], -1).view(z_t.size(0), -1)  # 10 x (19 x 512)
            else:
                embed = z_t.view(z_t.size(0), -1)  # 10 x (19 x 256)
            dec_t = self.predictor(embed)  # B x 2

            # recurrence
            h = self.rnn(torch.cat([x_t, z_t], 2), edge_idx[:, t], h, edge_weight=edge_weights[:, t])  # 640-->256

            # computing losses
            acc_loss += self._exp_loss(dec_t, y, t)

            all_dec.append(dec_t)
            all_hidden.append(h[-1])

        # soft attention to aggregate hidden states of all frames
        embed_video = self.soft_aggregation(torch.stack(all_hidden, dim=-1))
        dec = self.predictor_aux(embed_video)
        aux_loss = self.ce_loss(dec, y[:, 1].to(torch.long))

        return acc_loss, aux_loss, all_dec, all_hidden


    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)


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
        target_cls = target_cls.to(torch.long)
        pos_loss = -torch.mul(torch.exp(-torch.tensor((frames - time - 1) / fps)),
                              -self.ce_loss(pred, target_cls))
        # negative example
        neg_loss = self.ce_loss(pred, target_cls)

        loss = torch.mean(torch.add(torch.mul(pos_loss, target[:, 1]), torch.mul(neg_loss, target[:, 0])))
        return loss



class BayesGCRNN(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layers, n_obj=19, eps=1e-10, conv='GCN', bias=False, loss_func='exp', use_hidden=False):
        super(BayesGCRNN, self).__init__()

        self.x_dim = x_dim
        self.eps = eps
        self.h_dim = h_dim  # 512 (-->256)
        self.z_dim = z_dim  # 256 (-->128)
        self.n_layers = n_layers
        self.loss_func = loss_func
        self.use_hidden = use_hidden
        self.n_obj = n_obj

        self.phi_x = nn.Sequential(nn.Linear(x_dim, h_dim), nn.ReLU())

        # GCN encoder
        self.enc_gcn1 = GCNConv(h_dim + h_dim, h_dim)
        self.enc_gcn2 = GCNConv(h_dim + h_dim, z_dim, act=lambda x: x)
        # rnn layer
        self.rnn = graph_gru_gcn(h_dim + h_dim + z_dim, h_dim, n_layers, bias)
        # BNN decoder
        dim_encode = z_dim + h_dim if use_hidden else z_dim
        self.predictor = BayesianPredictor(n_obj * dim_encode, 2)
        
        # loss function
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        # intialize parameters
        self.reset_parameters(stdv=1e-2)


    def forward(self, x, y, graph, hidden_in=None, edge_weights=None, npass=2, nbatch=80, testing=False, loss_w=1.0):
        """
        :param x, (batchsize, nFrames, nBoxes, Xdim) = (10 x 100 x 20 x 4096)
        """
        losses = {'cross_entropy': 0,
                  'log_posterior': 0,
                  'log_prior': 0,
                  'total_loss': 0}
        all_dec, all_hidden = [], []

        # import ipdb; ipdb.set_trace()
        if hidden_in is None:
            h = Variable(torch.zeros(self.n_layers, x.size(0), self.n_obj, self.h_dim))  # 1 x 10 x 19 x 256
        else:
            h = Variable(hidden_in)
        h = h.to(x.device)

        for t in range(x.size(1)):
            # reduce the dim of node feature (FC layer)
            x_t = self.phi_x(x[:, t])  # 10 x 20 x 256
            img_embed = x_t[:, 0, :].unsqueeze(1).repeat(1, self.n_obj, 1).contiguous()  # 10 x 19 x 256
            x_t = torch.cat([x_t[:, 1:, :], img_embed], dim=-1)  # 10 x 19 x 512

            # GCN encoder
            enc = self.enc_gcn1(x_t, graph[:, t], edge_weight=edge_weights[:, t])  # 10 x 19 x 256 (512-->256)
            z_t = self.enc_gcn2(torch.cat([enc, h[-1]], -1), graph[:, t], edge_weight=edge_weights[:, t])  # 10 x 19 x 128 (512-->128)

            # BNN decoder
            if self.use_hidden:
                embed = torch.cat([z_t, h[-1]], -1).view(z_t.size(0), -1)  # 10 x (19 x 384)
            else:
                embed = z_t.view(z_t.size(0), -1)  # 10 x (19 x 128)
            dec_t, log_prior, log_variational_posterior = self.predictor.sample_elbo(embed, npass=npass, testing=testing)  # B x 2

            # recurrence
            h = self.rnn(torch.cat([x_t, z_t], -1), graph[:, t], h, edge_weight=edge_weights[:, t])  # rnn latent (640)-->256

            # computing losses
            L1 = log_variational_posterior / nbatch
            L2 = log_prior / nbatch
            L3 = self._exp_loss(dec_t, y, t)
            L = loss_w * (L1 - L2) + L3
            losses['cross_entropy'] += L3
            losses['log_posterior'] += L1
            losses['log_prior'] += L2
            losses['total_loss'] += L

            all_dec.append(dec_t)
            all_hidden.append(h[-1])

        return losses, all_dec, all_hidden


    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)



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
        target_cls = target_cls.to(torch.long)
        pos_loss = -torch.mul(torch.exp(-torch.tensor((frames - time - 1) / fps)),
                              -self.ce_loss(pred, target_cls))
        # negative example
        neg_loss = self.ce_loss(pred, target_cls)

        loss = torch.mean(torch.add(torch.mul(pos_loss, target[:, 1]), torch.mul(neg_loss, target[:, 0])))
        return loss


class SoftAggregate(torch.nn.Module):
    def __init__(self, agg_dim):
        super(SoftAggregate, self).__init__()
        self.agg_dim = agg_dim
        self.weight = nn.Parameter(torch.Tensor(agg_dim, 1))  # (100, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, hiddens):
        """
        hiddens: (10, 19, 256, 100)
        """
        maxpool = torch.max(hiddens, dim=1)[0]  # (10, 256, 100)
        avgpool = torch.mean(hiddens, dim=1)    # (10, 256, 100)
        agg_spatial = torch.cat((avgpool, maxpool), dim=1)  # (10, 512, 100)

        # soft-attention
        energy = torch.bmm(agg_spatial.permute([0, 2, 1]), agg_spatial)  # (10, 100, 100)
        attention = self.softmax(energy)
        weighted_feat = torch.bmm(attention, agg_spatial.permute([0, 2, 1]))  # (10, 100, 512)
        weight = self.weight.unsqueeze(0).repeat([hiddens.size(0), 1, 1])
        agg_feature = torch.bmm(weighted_feat.permute([0, 2, 1]), weight)  # (10, 512, 1)

        return agg_feature.squeeze(dim=-1)  # (10, 512)

