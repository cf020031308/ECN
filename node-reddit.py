import os
import sys
import time
import datetime
import argparse

import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from dgl import data as dgl_data
from torch_geometric import datasets as pyg_data
from ogb.nodeproppred import NodePropPredDataset
from sklearn.metrics import f1_score


SAMPLE_LIMIT = 2 ** 24
parser = argparse.ArgumentParser()
parser.add_argument('method', type=str, default='MLP', help=(
    'MLP | GCN | GraphSAGE | FastGCN | GraphSAINT | ECN'
))
parser.add_argument('dataset', type=str, default='cora', help=(
    'cora | citeseer | pubmed | flickr | ppi | arxiv | yelp | reddit'
))
parser.add_argument(
    '--hidden', type=int, default=0,
    help='Dimension of hidden representations. Default: auto')
parser.add_argument('--batch-size', type=int, default=0, help='Default: auto')
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--gpu2', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate')
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--weight-decay', type=float, default=0.0)
parser.add_argument(
    '--symmetric', action='store_true',
    help='Symmetrically normalized adjacency matrix')
parser.add_argument(
    '--transductive', action='store_true',
    help='Access graph nodes of the test set in training phase')
parser.add_argument(
    '--semi-supervised', action='store_true',
    help=('Only available for cora, citeseer and pubmed. '
          'Default: full supervised'))
parser.add_argument(
    '--early-stop-iters', type=int, default=100,
    help='Maximum number of iterations for performance decline to stop')
parser.add_argument(
    '--early-stop-epochs', type=int, default=3,
    help='Maximum number of epochs for performance decline to stop')
parser.add_argument(
    '--max-epochs', type=int, default=500,
    help='Maximum number of epochs before early stop')
parser.add_argument(
    '--precompute', type=int, default=-1,
    help=('Times to transform features with normalized adjacency matrix. '
          'Default: auto'))
parser.add_argument(
    '--skip-connection', action='store_true',
    help='Enable skip connections for all layers')
parser.add_argument(
    '--embedding', type=float, default=0.0,
    help='Scale the effect of regularization for layer1 with GAE')
parser.add_argument(
    '--attention', type=int, default=0,
    help='Number of attention heads for layer2')
parser.add_argument(
    '--no-importance-sampling',
    action='store_true',
    help='Disable importance sampling. Sample uniformly')
parser.add_argument(
    '--middle-layers', type=int, default=-1,
    help='Number of layers between layer1 and layer2. Default: precompute - 1')
args = parser.parse_args()

if args.method == 'GCN':
    args.early_stop_epochs = max(
        args.early_stop_epochs, args.early_stop_iters)
elif args.method == 'GraphSAGE':
    args.skip_connection = True
if args.hidden <= 0:
    args.hidden = (
        256 if args.dataset in ('yelp', 'amazon')
        else 128 if args.dataset in ('ppi', 'arxiv')
        else 64 if args.dataset in ('flickr', 'pubmed')
        else 32
    )
if args.precompute == -1:
    if args.method in ('MLP', 'ECN'):
        args.precompute = {
            'cora': 6,
            'citeseer': 1,
            'pubmed': 1,
            'flickr': 2,
            'ppi': 1,
            'arxiv': 3,
            'yelp': 2,
            'reddit': 2,
        }.get(args.dataset, 2)
        if args.method == 'ECN' and not args.skip_connection:
            args.precompute -= 1
    elif args.method == 'GraphSAINT':
        args.precompute = 0
    else:
        args.precompute = 1
if args.middle_layers == -1:
    args.middle_layers = max(0, args.precompute - 1)

if args.batch_size <= 0:
    if args.method == 'MLP':
        # NOTE: SGC = MLP + Precompute
        args.batch_size = {
            'cora': 256,
            'citeseer': 256,
            'pubmed': 1024,
            'flickr': 1024,
            'ppi': 4 * 1024,
            'arxiv': 4 * 1024,
            'yelp': 4 * 1024,
            'reddit': 4 * 1024,
        }.get(args.dataset, 2 * 1024)
    elif args.method == 'GCN':
        args.batch_size = {
        }.get(args.dataset, 4 * 1024)
    elif args.method == 'GraphSAGE':
        args.batch_size = {
            'cora': 128,
            'citeseer': 128,
            'pubmed': 512,
            'flickr': 512,
            'ppi': 512,
            'arxiv': 512,
            'yelp': 512,
            'reddit': 512,
        }.get(args.dataset, 512)
    elif args.method == 'FastGCN':
        args.batch_size = {
            'cora': 256,
            'citeseer': 256,
            'pubmed': 1024,
            'flickr': 1024,
            'ppi': 4 * 1024,
            'arxiv': 4 * 1024,
            'yelp': 4 * 1024,
            'reddit': 4 * 1024,
        }.get(args.dataset, 2 * 1024)
    elif args.method == 'GraphSAINT':
        args.batch_size = {
            'cora': 256,
            'citeseer': 256,
            'pubmed': 1024,
            'flickr': 1024,
            'ppi': 2 * 1024,
            'arxiv': 2 * 1024,
            'yelp': 4 * 1024,
            'reddit': 4 * 1024,
        }.get(args.dataset, 1024)
    else:
        # ECNs
        args.batch_size = {
            'cora': 256,
            'citeseer': 256,
            'pubmed': 1024,
            'flickr': 1024,
            'ppi': 2 * 1024,
            'arxiv': 2 * 1024,
            'yelp': 4 * 1024,
            'reddit': 4 * 1024,
        }.get(args.dataset, 1024)
if not torch.cuda.is_available():
    args.gpu = -1
print(datetime.datetime.now(), args)

g_dev = None
gpu = lambda x: x
if args.gpu >= 0:
    g_dev = torch.device('cuda:%d' % args.gpu)
    gpu = lambda x: x.to(g_dev)
dev2 = g_dev
if args.gpu2 >= 0:
    dev2 = torch.device('cuda:%d' % args.gpu2)
coo = torch.sparse_coo_tensor
get_score = lambda y_true, y_pred: f1_score(
    y_true.cpu(), y_pred.cpu(), average='micro').item()
ce = lambda x, y: -F.logsigmoid((x * y).mean(dim=1)).mean()


class Optim(object):
    def __init__(self, params):
        self.params = params
        self.opt = torch.optim.Adam(
            params, lr=args.lr, weight_decay=args.weight_decay)

    def __repr__(self):
        return 'params: %d' % sum(p.numel() for p in self.params)

    def __enter__(self):
        self.opt.zero_grad()
        self.elapsed = time.time()
        return self.opt

    def __exit__(self, *vs, **kvs):
        self.opt.step()
        self.elapsed = time.time() - self.elapsed


def spmm(adj, x, batch_size=0, out=None, dotm=False):
    src, dst = adj._indices()
    n_edges = src.shape[0]
    if dotm:
        assert n_edges == x.shape[0]
    elif not 0 < batch_size < n_edges:
        h = adj @ x
        if out is not None:
            h = out + h
        return h
    values = adj._values()
    if out is None:
        out = torch.zeros(adj.shape[0], x.shape[1]).to(adj.device)
    erange = torch.arange(n_edges)
    for cur in erange[::batch_size]:
        perm = erange[cur: cur + batch_size]
        out.scatter_add_(
            dim=0,
            index=src[perm].view(-1, 1).repeat(1, x.shape[1]),
            src=values[perm].view(-1, 1) * x[
                perm if dotm else dst[perm]])
    return out


def spfwd(m, *x, batch_size=0, out=None):
    n_nodes = x[0].shape[0]
    if not 0 < batch_size < n_nodes:
        h = m(*x)
        if out is not None:
            h = out + h
        return h
    nrange = torch.arange(n_nodes)
    for cur in nrange[::batch_size]:
        perm = nrange[cur: cur + batch_size]
        h = m(*([y[perm] for y in x]))
        if out is None:
            out = torch.zeros(n_nodes, *h.shape[1:]).to(x[0].device)
        out[perm] = out[perm] + h
    return out


class ResLayer(nn.Module):
    def __init__(self, dim, dropout):
        super(self.__class__, self).__init__()
        self.layers = gpu(nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        ))

    def forward(self, x):
        return F.leaky_relu(x + self.layers(x))


class Net(nn.Module):
    def __init__(
            self, din, hid, dout, dropout=0,
            precompute=0, middle_layers=0,
            skip_connection=False, attention=None, **kwargs):
        super(self.__class__, self).__init__()
        self.A = [None, None]
        self.attention = None
        self.skip_layer = None
        if attention:
            # self.attention = gpu(nn.Linear(2 * dout, attention))
            self.attention = nn.Parameter(gpu(
                torch.rand(1, attention, 2 * dout)))
        if skip_connection:
            self.skip_layer = gpu(nn.Sequential(
                nn.Dropout(dropout),
                nn.LayerNorm(hid),
                nn.Linear(hid, dout),
            ))
        self.layers = nn.ModuleList([
            gpu(nn.Sequential(*([
                nn.Dropout(dropout),
                nn.Linear((1 + precompute * skip_connection) * din, hid),
                nn.LeakyReLU(),
            ] + [
                ResLayer(hid, dropout)
                for _ in range(middle_layers)
            ]))),
            gpu(nn.Sequential(
                nn.Dropout(dropout),
                nn.LayerNorm(hid),
                nn.Linear(hid, dout * (attention or 1)),
            ))
        ])

    def forward(self, x, batch_size=0):
        for adj, layer in zip(self.A, self.layers):
            h = x
            if adj is not None:
                x = spmm(adj, h, batch_size)
                h = torch.cat((h, x), dim=1) if self.skip_layer else x
            x = spfwd(layer, h, batch_size=batch_size)
        return x

    def score(self, x, y):
        return F.leaky_relu(
            (torch.cat((x, y), dim=-1) * self.attention).mean(dim=-1), 0.2)

    def batch_attention(self, x, y, w):
        s = self.score(x, y)
        return x.shape[0] * torch.softmax(s - torch.log(w).view(-1, 1), dim=1)


def count_subgraphs(edges, n):
    subs = 0
    while n:
        subs += 1
        mask = torch.zeros(n, dtype=bool).to(edges.device)
        mask[0] = True
        src, dst = edges
        flag = -1
        while 1:
            m = torch.cat((src[mask[dst]], dst[mask[src]])).unique()
            mask[m] = True
            if flag == m.shape[0]:
                break
            flag = m.shape[0]
        if mask.all():
            return subs
        nodes, edges = edges[:, ~mask[src]].unique(return_inverse=True)
        n = nodes.shape[0]
    return subs


def load_data(name):
    if args.dataset in ('arxiv', 'mag', 'products'):
        ds = NodePropPredDataset(name='ogbn-%s' % args.dataset)
        train_idx, valid_idx, test_idx = map(
            ds.get_idx_split().get, 'train valid test'.split())
        if args.dataset == 'mag':
            train_idx = train_idx['paper']
            valid_idx = valid_idx['paper']
            test_idx = test_idx['paper']
        g, labels = ds[0]
        if args.dataset == 'mag':
            labels = labels['paper']
            g['edge_index'] = g['edge_index_dict'][('paper', 'cites', 'paper')]
            g['node_feat'] = g['node_feat_dict']['paper']
        X = torch.from_numpy(g['node_feat'])
        Y = torch.from_numpy(labels).clone().squeeze(-1)
        E = torch.from_numpy(g['edge_index'])
        n_nodes = X.shape[0]
        train_mask = torch.zeros(n_nodes, dtype=bool)
        valid_mask = torch.zeros(n_nodes, dtype=bool)
        test_mask = torch.zeros(n_nodes, dtype=bool)
        train_mask[train_idx] = True
        valid_mask[valid_idx] = True
        test_mask[test_idx] = True
        is_bidir = False
    elif args.dataset in ('flickr', 'yelp', 'amazon'):
        dn = 'dataset/' + args.dataset
        g = (
            pyg_data.Flickr(dn) if args.dataset == 'flickr'
            else pyg_data.Yelp(dn) if args.dataset == 'yelp'
            else pyg_data.AmazonProducts(dn) if args.dataset == 'amazon'
            else None
        ).data
        X, Y, E, train_mask, valid_mask, test_mask = map(
            g.get, 'x y edge_index train_mask val_mask test_mask'.split())
        # for i in range(E.shape[1]):
        #     src, dst = E[:, i]
        #     if src.item() != dst.item():
        #         print(src, dst)
        #         break
        # is_bidir = ((E[0] == dst) & (E[1] == src)).any().item()
        # print('guess is bidir:', is_bidir)
        is_bidir = True
    elif args.dataset == 'ppi':
        g_train, g_valid, g_test = map(
            dgl_data.PPIDataset, 'train valid test'.split())
        n_nodes = 0
        X, Y, E, mask = [], [], [], []
        for mode in 'train valid test'.split():
            for g in dgl_data.PPIDataset(mode):
                X.append(g.ndata['feat'])
                Y.append(g.ndata['label'])
                E.append(n_nodes + torch.cat(
                    [e.view(1, -1) for e in g.edges()], dim=0))
                n_nodes += X[-1].shape[0]
            mask.append(n_nodes)
        X = torch.cat(X, dim=0)
        Y = torch.cat(Y, dim=0)
        E = torch.cat(E, dim=1)
        train_mask, valid_mask, test_mask = torch.zeros(
            (3, mask[2]), dtype=bool)
        train_mask[:mask[0]] = True
        valid_mask[mask[0]:mask[1]] = True
        test_mask[mask[1]:] = True
        is_bidir = True
    else:
        g = (
            dgl_data.CoraGraphDataset() if args.dataset == 'cora'
            else dgl_data.CiteseerGraphDataset() if args.dataset == 'citeseer'
            else dgl_data.PubmedGraphDataset() if args.dataset == 'pubmed'
            else dgl_data.RedditDataset() if args.dataset == 'reddit'
            else None
        )[0]
        X, Y, train_mask, valid_mask, test_mask = map(
            g.ndata.get, 'feat label train_mask val_mask test_mask'.split())
        E = torch.cat([e.view(1, -1) for e in g.edges()], dim=0)
        if not args.semi_supervised and args.dataset in (
                'cora', 'citeseer', 'pubmed'):
            train_mask[:] = True
            train_mask[valid_mask] = False
            train_mask[test_mask] = False
        is_bidir = True
    # Get Undirectional Edges
    if not is_bidir:
        E = torch.cat((E, E[[1, 0]]), dim=1)
    # Add Self-Loops
    E = torch.cat((
        torch.arange(X.shape[0]).view(1, -1).repeat(2, 1),
        E[:, E[0] != E[1]]), dim=1)
    return X, Y, E, train_mask, valid_mask, test_mask


class Stat(object):
    def __init__(self, as_ecn=False):
        self.ecn = as_ecn

        self.preprocess_time = 0
        self.sampling_times = []
        self.training_times = []
        self.evaluation_times = []

        self.best_test_scores = []
        self.best_times = []
        self.best_training_times = []
        self.best_sampling_times = []

        self.mem = psutil.Process().memory_info().rss / 1024 / 1024
        self.gpu = 0
        if g_dev is not None:
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass
            self.gpu = torch.cuda.memory_allocated(g_dev) / 1024 / 1024

    def start_preprocessing(self):
        self.preprocess_time = time.time()

    def stop_preprocessing(self):
        self.preprocess_time = time.time() - self.preprocess_time

    def start_run(self):
        self.params = None
        self.scores = []
        self.acc_training_times = []
        self.acc_sampling_times = []
        self.acc_times = []
        self.iterations = 0
        self.sampling_times.append(0.)
        self.training_times.append(0.)
        self.evaluation_times.append(0.)

    def start_sampling(self):
        self.sampling_times[-1] = time.time() - self.sampling_times[-1]

    def stop_sampling(self):
        self.sampling_times[-1] = time.time() - self.sampling_times[-1]

    def record_training(self, elapsed):
        self.iterations += 1
        self.training_times[-1] += elapsed

    def record_evaluation(self, elapsed):
        self.evaluation_times[-1] += elapsed

    def evaluate_result(self, valid_y, test_y):
        if is_multiclass:
            self.scores.append([
                get_score(Y[valid_mask], valid_y > 0),
                get_score(Y[test_mask], test_y > 0)])
        else:
            self.scores.append([
                get_score(Y[valid_mask], valid_y.argmax(dim=1)),
                get_score(Y[test_mask], test_y.argmax(dim=1))])
        self.acc_training_times.append(self.training_times[-1])
        self.acc_sampling_times.append(self.sampling_times[-1])
        self.acc_times.append(
            self.preprocess_time
            + self.sampling_times[-1]
            + self.training_times[-1])
        dec_epochs = len(self.scores) - 1 - torch.tensor(
            self.scores).max(dim=0).indices[0]
        if dec_epochs == 0:
            self.iterations = 0
        return (
            dec_epochs >= args.early_stop_epochs
            and self.iterations >= args.early_stop_iters)

    def evaluate_model(self, model, args):
        if args.method == 'MLP':
            return self.evaluate_mlp(model, args.batch_size)
        if self.ecn:
            return self.evaluate_ecn(model, args.batch_size)
        return self.evaluate_gcn(model, args.batch_size)

    def evaluate_mlp(self, model, batch_size):
        with torch.no_grad():
            model.eval()
            t = time.time()
            valid_y = model(P[valid_mask], batch_size=args.batch_size)
            test_y = model(P[test_mask], batch_size=args.batch_size)
            ev.record_evaluation(time.time() - t)
            ret = self.evaluate_result(valid_y, test_y)
        model.train()
        return ret

    def evaluate_gcn(self, model, batch_size):
        model_A = model.A
        with torch.no_grad():
            model.eval()
            layer1, layer2 = model.layers
            t = time.time()
            H = layer1(P).to(dev2)
            if model.attention is not None:
                H2 = layer2.to(dev2)(H)
                layer2.to(g_dev)
            zs = []
            for adj, mask in [(valid_A, valid_mask), (test_A, test_mask)]:
                if model.attention is None:
                    z = layer2.to(dev2)(spmm(adj, H, batch_size))
                    layer2.to(g_dev)
                else:
                    heads = model.attention.shape[1]
                    H2 = H2.view(H2.shape[0], heads, -1)
                    n = adj.shape[0]
                    src, dst = adj._indices()
                    e = src.shape[0]
                    erange = torch.arange(e)
                    h2 = H2[mask]
                    s = torch.zeros(e, heads).to(dev2)
                    z = torch.zeros(n, heads).to(dev2)
                    for cur in erange[::batch_size]:
                        psrc = src[cur: cur + batch_size]
                        zi = h2[psrc]
                        zj = H2[dst[cur: cur + batch_size]]
                        model.attention.data = model.attention.data.to(dev2)
                        sj = torch.exp(model.score(zi, zj))
                        model.attention.data = model.attention.data.to(g_dev)
                        s[cur: cur + batch_size] = sj
                        z.scatter_add_(
                            0, psrc.view(-1, 1).repeat(1, sj.shape[1]), sj)
                    for cur in erange[::batch_size]:
                        s[cur: cur + batch_size] /= z[
                            src[cur: cur + batch_size]]
                    z = torch.zeros(n, zj.shape[-1]).to(dev2)
                    for cur in erange[::batch_size]:
                        psrc = src[cur: cur + batch_size]
                        sj = s[cur: cur + batch_size]
                        zj = H2[dst[cur: cur + batch_size]]
                        z.scatter_add_(
                            0, psrc.view(-1, 1).repeat(1, zj.shape[-1]),
                            (sj.unsqueeze(-1) * zj).mean(dim=1))
                if model.skip_layer is not None:
                    z = z + model.skip_layer.to(dev2)(H[mask])
                    model.skip_layer.to(g_dev)
                zs.append(z)
            self.record_evaluation(time.time() - t)
            ret = self.evaluate_result(*zs)
        model.A = model_A
        model.train()
        return ret

    def evaluate_ecn(self, model, batch_size):
        model_A = model.A
        model.A = [None, None]
        with torch.no_grad():
            model.eval()
            layer1, layer2 = model.layers
            t = time.time()
            H = layer1(P).to(dev2)
            H2 = layer2.to(dev2)(H)
            layer2.to(g_dev)
            ys = []
            for adj, mask in [(valid_A, valid_mask), (test_A, test_mask)]:
                if model.attention is not None:
                    H2 = H2.view(H2.shape[0], model.attention.shape[1], -1)
                    h2 = H2[mask]
                    if model.skip_layer is not None:
                        sk = model.skip_layer.to(dev2)(H[mask])
                        model.skip_layer.to(g_dev)
                    src, dst = adj._indices()
                    erange = torch.arange(src.shape[0])
                    v = torch.zeros(src.shape[0], H2.shape[-1]).to(dev2)
                    w = adj._values()
                    for perm in DataLoader(erange, batch_size, shuffle=True):
                        psrc = src[perm]
                        zi = h2[psrc]
                        zj = H2[dst[perm]]
                        model.attention.data = model.attention.data.to(dev2)
                        s = model.batch_attention(zi, zj, w[perm])
                        model.attention.data = model.attention.data.to(g_dev)
                        if model.skip_layer is not None:
                            zj = zj + sk[psrc].unsqueeze(1)
                        v.scatter_add_(
                            0, psrc.view(-1, 1).repeat(1, zj.shape[-1]),
                            sg((s.unsqueeze(-1) * zj).mean(dim=1)))
                    y = spmm(adj, v, batch_size, dotm=True)
                elif model.skip_layer is not None:
                    sk = model.skip_layer.to(dev2)(H[mask])
                    model.skip_layer.to(g_dev)
                    src, dst = adj._indices()
                    erange = torch.arange(src.shape[0])
                    v = torch.zeros(src.shape[0], H2.shape[-1]).to(dev2)
                    for cur in erange[::batch_size]:
                        psrc = src[cur: cur + batch_size]
                        pdst = dst[cur: cur + batch_size]
                        v.scatter_add_(
                            0, psrc.view(-1, 1).repeat(1, H2.shape[-1]),
                            sg(sk[psrc] + H2[pdst]))
                    y = spmm(adj, v, batch_size, dotm=True)
                else:
                    y = spmm(adj, sg(H2), batch_size)
                ys.append(y - 0.5)
            self.record_evaluation(time.time() - t)
            ret = self.evaluate_result(*ys)
        model.A = model_A
        model.train()
        return ret

    def end_run(self):
        print('evaluate_as_ecn:', self.ecn)
        print('val scores:', [s for s, _ in self.scores])
        print('test scores:', [s for _, s in self.scores])
        print('acc sampling times:', self.acc_sampling_times)
        print('acc training times:', self.acc_training_times)
        self.scores = torch.tensor(self.scores)
        print('max scores:', self.scores.max(dim=0).values)
        idx = self.scores.max(dim=0).indices[0]
        self.best_test_scores.append((idx, self.scores[idx, 1]))
        self.best_training_times.append(self.acc_training_times[idx])
        self.best_sampling_times.append(self.acc_sampling_times[idx])
        self.best_times.append(self.acc_times[idx])
        print('best test score:', self.best_test_scores[-1])

    def end_all(self):
        print('evaluate_as_ecn:', self.ecn)
        conv = 1.0 + torch.tensor([
            idx for idx, _ in self.best_test_scores])
        score = 100 * torch.tensor([
            score for _, score in self.best_test_scores])
        tm = torch.tensor(self.best_times)
        ttm = torch.tensor(self.best_training_times)
        stm = torch.tensor(self.best_sampling_times)
        print('converge time: %.3f±%.3f' % (
            tm.mean().item(), tm.std().item()))
        print('converge training time: %.3f±%.3f' % (
            ttm.mean().item(), ttm.std().item()))
        print('converge sampling time: %.3f±%.3f' % (
            stm.mean().item(), stm.std().item()))
        print('converge epochs: %.3f±%.3f' % (
            conv.mean().item(), conv.std().item()))
        print('score: %.2f±%.2f' % (score.mean().item(), score.std().item()))

        # Output Used Time
        print('preprocessing time: %.3f' % self.preprocess_time)
        for name, times in (
            ('total sampling', self.sampling_times),
            ('total training', self.training_times),
            ('total evaluation', self.evaluation_times),
        ):
            times = torch.tensor(times or [0], dtype=float)
            print('%s time: %.3f±%.3f' % (
                name, times.mean().item(), times.std().item()))
        ratios = torch.tensor([
            s / t for s, t in zip(self.sampling_times, self.training_times)
        ] or [0])
        print('sampling / training: %.2f±%.2f' % (
            ratios.mean().item(), ratios.std().item()))

        # Output Used Space
        mem = psutil.Process().memory_info().rss / 1024 / 1024
        gpu = 0
        if g_dev is not None:
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass
            gpu = torch.cuda.max_memory_allocated(g_dev) / 1024 / 1024
        print('pre_memory: %.2fM + %.2fM = %.2fM' % (
            self.mem, self.gpu, self.mem + self.gpu))
        print('max_memory: %.2fM + %.2fM = %.2fM' % (
            mem, gpu, mem + gpu))
        print('memory_diff: %.2fM + %.2fM = %.2fM' % (
            mem - self.mem,
            gpu - self.gpu,
            mem + gpu - self.mem - self.gpu))


def node_sampling(crow, col, nids, rank):
    neis = []
    for nid in nids:
        ids = col[crow[nid]: crow[nid + 1]]
        neis.append(ids[torch.randint(0, ids.shape[0], (rank, ))].view(1, -1))
    return torch.cat(neis, dim=0)


def layer_sampling(crow, col, values, nids, q, rank):
    src, dst, val = [], [], []
    # NOTE: This loop is the most time-consuming part
    for nid in nids:
        idx_s, idx_e = crow[nid], crow[nid + 1]
        src.append(torch.tensor([nid] * (idx_e - idx_s)))
        dst.append(col[idx_s: idx_e])
        val.append(values[idx_s: idx_e])
    dst = torch.cat(dst)
    neis, inv = dst.unique(sorted=False, return_inverse=True)
    p = q[neis]
    p = p / p.sum()
    sampled = torch.zeros(neis.shape[0], dtype=bool)
    sampled[list(WeightedRandomSampler(p, rank))] = True
    mask = sampled[inv]
    src, sinv = torch.cat(src)[mask].unique(sorted=False, return_inverse=True)
    dst, dinv = dst[mask].unique(sorted=False, return_inverse=True)
    return (src, dst), coo(
        torch.cat((sinv.view(1, -1), dinv.view(1, -1)), dim=0),
        (torch.cat(val) / p[inv].cpu() / sampled.sum())[mask],
        (src.shape[0], dst.shape[0]))


def edge_sampling(edges, q, batch_size, return_weight=False):
    for perm in DataLoader(range(q.shape[0]), SAMPLE_LIMIT, shuffle=True):
        p, e = q[perm], edges[:, perm]
        sampled = torch.tensor(
            list(WeightedRandomSampler(p, p.shape[0])), dtype=int)
        for cur in range(0, p.shape[0], batch_size):
            idx = sampled[cur: cur + batch_size]
            yield (e[:, idx], p[idx]) if return_weight else e[:, idx]


def subg_construction(crow, col, nodes):
    inv = {n: i for i, n in enumerate(nodes.tolist())}
    src, dst, edge_index = [], [], []
    deg = torch.zeros(nodes.shape[0])
    for nid, idx in inv.items():
        idx_s, idx_e = crow[nid], crow[nid + 1]
        for i_n, n in enumerate(col[idx_s: idx_e].tolist()):
            if n in inv:
                deg[idx] += 1
                src.append(idx)
                dst.append(inv[n])
                edge_index.append(idx_s + i_n)
    edges = torch.tensor([src, dst], dtype=int)
    # values = (deg ** -0.5)[edges].prod(dim=0)
    values = (deg ** -1)[edges[0]]
    edge_index = torch.tensor(edge_index, dtype=int)
    return edges, values, edge_index


X, Y, E, train_mask, valid_mask, test_mask = load_data(args.dataset)
n_nodes = X.shape[0]
n_features = X.shape[1]
is_multiclass = len(Y.shape) == 2
n_labels = Y.shape[1] if is_multiclass else int(Y.max().item() + 1)
deg = E.shape[1] / n_nodes
print('nodes: %d' % n_nodes)
print('features: %d' % n_features)
print('classes: %d' % n_labels)
print('is_multiclass:', is_multiclass)
print('edges without self-loops: %d' % ((E.shape[1] - n_nodes) / 2))
print('average degree: %.2f' % deg)
print('split: %d (%.2f%%) / %d (%.2f%%) / %d (%.2f%%)' % (
    train_mask.sum(), 100 * train_mask.sum() / n_nodes,
    valid_mask.sum(), 100 * valid_mask.sum() / n_nodes,
    test_mask.sum(), 100 * test_mask.sum() / n_nodes))
print('intra_rate: %.2f%%' % (100 * (
    Y[E[0]] == Y[E[1]]).sum().float() / E.shape[1] / (
        n_labels if is_multiclass else 1)))
if is_multiclass:
    _cri = nn.BCEWithLogitsLoss(reduction='none')
    criterion = lambda x, y: _cri(x, y).sum(dim=1)
    sg = torch.sigmoid
else:
    criterion = lambda x, y: F.cross_entropy(x, y, reduction='none')
    sg = lambda x: torch.softmax(x, dim=-1)


def norm_adj(edges, n, symmetric=False):
    deg = torch.zeros(n).to(edges.device)
    deg.scatter_add_(
        dim=0, index=edges[0],
        src=torch.ones(edges.shape[1]).to(edges.device))
    # with open('degree_counts/%s_train.txt' % args.dataset, 'w') as file:
    #     for xs in deg.unique(sorted=True, return_counts=True):
    #         file.write(','.join('%d' % x for x in xs))
    #         file.write('\n')
    if symmetric:
        val = (deg ** -0.5)[edges].prod(dim=0)
    else:
        val = (deg ** -1)[edges[0]]
    return coo(edges, val, (n, n))


def slice_adj(adj, mask):
    edges = adj._indices()
    val = adj._values()
    m = mask[edges[0]]
    src, dst = edges[:, m]
    _, src = src.unique(sorted=True, return_inverse=True)
    return coo(
        torch.cat((src.view(1, -1), dst.view(1, -1)), dim=0),
        val[m], (mask.sum(), mask.shape[0]))


X, Y = map(gpu, [X, Y])
# print('subgraphs:', count_subgraphs(E, n_nodes))
# symmetric = args.method not in ('GraphSAGE', 'ECN')
A = norm_adj(E, n_nodes, args.symmetric)
valid_A = slice_adj(A, valid_mask)
test_A = slice_adj(A, test_mask)

# Inductive Setting
if args.transductive:
    induc_E = E
    induc_A = A
else:
    # induc_E = E[:, ~(valid_mask[E] | test_mask[E]).sum(dim=0).bool()]
    induc_E = E[:, ~(test_mask[E].sum(dim=0).bool())]
    induc_A = norm_adj(induc_E, n_nodes,  args.symmetric)
E = None

if args.method == 'ECN':
    evs = [Stat(False), Stat(True)]

    class Stats(object):
        def __getattribute__(self, name):
            return lambda *vs, **kvs: all([
                getattr(e, name)(*vs, **kvs) for e in evs])
    ev = Stats()
else:
    ev = Stat(False)

# Preprocessing
ev.start_preprocessing()

# Precompute for Evaluation
A = A.to(X.device)
P = X
for _ in range(args.precompute):
    if args.skip_connection:
        P = torch.cat((
            P, spmm(A, P[:, -n_features:], args.batch_size)), dim=1)
    else:
        P = spmm(A, P, args.batch_size)
A = None

# Precompute for Training
# DO NOT combine this section with the previous one in case memory issue
induc_A = induc_A.to(X.device)
induc_P = X
if args.skip_connection:
    X = None
    for _ in range(args.precompute):
        induc_P = torch.cat((
            induc_P, spmm(induc_A, induc_P[:, -n_features:], args.batch_size)
        ), dim=1)
elif args.method != 'GraphSAINT':
    X = None
    for _ in range(args.precompute):
        induc_P = spmm(induc_A, induc_P, args.batch_size)

valid_A = valid_A.to(dev2)
test_A = test_A.to(dev2)

# Set Distribution for Node Samplers
if args.method == 'FastGCN':
    nq = torch.zeros(n_nodes).to(induc_A.device)
    nq.scatter_add_(0, induc_E[0].to(induc_A.device), induc_A._values() ** 2)
    nq /= nq.sum()
# Set Distribution for Edge Samplers
if args.method == 'GraphSAINT':
    eq = (1 / induc_A._values()[induc_E]).sum(dim=0)
    eq *= eq.shape[0] / eq.sum()

# Train with Mini Batches of Nodes
if args.method in ('MLP', 'GraphSAGE', 'FastGCN'):
    train_nidx = torch.arange(n_nodes)[train_mask]
# Train with Mini Batches of Edges
if args.method == 'ECN':
    train_emask = train_mask[induc_E[0]]
    train_E = induc_E[:, train_emask]
    train_C = induc_A._values()[train_emask]

ev.stop_preprocessing()

if args.method != 'GraphSAINT':
    induc_E = None

# CSR Matrix for Neighbour Indexing
if args.method in ('GraphSAGE', 'FastGCN', 'GraphSAINT'):
    # NOTE: Unable to Move CSR Matrices Into CUDA
    fn = 'dataset/csr/%s' % args.dataset
    if os.path.exists(fn):
        print('CSR exists, load from cache')
        crow, col, values = map(torch.load(fn).get, ['crow', 'col', 'values'])
    else:
        induc_A = induc_A.cpu()._to_sparse_csr()
        crow = induc_A.crow_indices()
        col = induc_A.col_indices()
        values = induc_A.values()
        induc_A = None
        torch.save({'crow': crow, 'col': col, 'values': values}, fn)
    storage = sys.getsizeof(crow.storage())
    print('Crow Storage: %.3fM' % (storage / 1024 / 1024))
    storage += sys.getsizeof(col.storage())
    if args.method in ('FastGCN', 'GraphSAINT'):
        storage += sys.getsizeof(values.storage())
    print('CSR Storage: %.3fM' % (storage / 1024 / 1024))
elif args.method != 'GCN':
    induc_A = None


for run in range(args.runs):
    torch.manual_seed(run)
    ev.start_run()

    if args.method == 'MLP':
        net = Net(n_features, args.hidden, n_labels, **args.__dict__)
        opt = Optim([*net.parameters()])
        for epoch in range(1, 1 + args.max_epochs):
            ev.start_sampling()
            for perm in DataLoader(train_nidx, args.batch_size, shuffle=True):
                ev.stop_sampling()
                with opt:
                    criterion(net(induc_P[perm]), Y[perm]).mean().backward()
                ev.record_training(opt.elapsed)
                ev.start_sampling()
            ev.stop_sampling()
            if ev.evaluate_model(net, args):
                break
    elif args.method == 'GCN':
        net = Net(n_features, args.hidden, n_labels, **args.__dict__)
        net.A = [None, induc_A] if args.precompute else [induc_A, induc_A]
        opt = Optim([*net.parameters()])
        for epoch in range(1, 1 + args.max_epochs):
            with opt:
                criterion(
                    net(induc_P, batch_size=args.batch_size)[train_mask],
                    Y[train_mask]).mean().backward()
            ev.record_training(opt.elapsed)
            if ev.evaluate_model(net, args):
                break
    elif args.method == 'GraphSAGE':
        net = Net(n_features, args.hidden, n_labels, **args.__dict__)
        opt = Optim([*net.parameters()])
        layer1, layer2 = net.layers
        rank1 = 1 + int(deg / 2)
        rank2 = deg if args.precompute else rank1
        for epoch in range(1, 1 + args.max_epochs):
            for iteration in range(1 + int(deg * deg / rank1 / rank2)):
                ev.start_sampling()
                for perm in DataLoader(train_nidx, args.batch_size, True):
                    n1, i1 = torch.cat((
                        perm.view(-1, 1), node_sampling(crow, col, perm, rank1)
                    ), dim=1).unique(sorted=False, return_inverse=True)
                    p = induc_P[n1]
                    # TODO: here's a bug if precompute == 0
                    if not args.precompute:
                        n2 = node_sampling(crow, col, n1, rank2)
                        p = torch.cat((p, induc_P[n2].mean(dim=1)), dim=1)
                    ev.stop_sampling()
                    with opt:
                        h = layer1(p)
                        z = (net.skip_layer(h[i1[:, 0]])
                             + layer2(h[i1[:, 1:]].mean(dim=1)))
                        criterion(z, Y[perm]).mean().backward()
                    ev.record_training(opt.elapsed)
                    ev.start_sampling()
                ev.stop_sampling()
            if ev.evaluate_model(net, args):
                break
    elif args.method == 'FastGCN':
        net = Net(n_features, args.hidden, n_labels, **args.__dict__)
        opt = Optim([*net.parameters()])
        rank = 2 * args.batch_size
        for epoch in range(1, 1 + args.max_epochs):
            for iteration in range(1 + int(deg * args.batch_size / rank)):
                ev.start_sampling()
                for perm in DataLoader(train_nidx, args.batch_size, True):
                    (src, dst), adj = layer_sampling(
                        crow, col, values, perm, nq, rank)
                    ev.stop_sampling()
                    with opt:
                        net.A = [None, adj.to(induc_P.device)]
                        criterion(
                            net(induc_P[dst]), Y[src]).mean().backward()
                    ev.record_training(opt.elapsed)
                    ev.start_sampling()
                ev.stop_sampling()
            if ev.evaluate_model(net, args):
                break
    elif args.method == 'GraphSAINT':
        net = Net(n_features, args.hidden, n_labels, **args.__dict__)
        opt = Optim([*net.parameters()])
        N = 1
        cv = values.clone().to(X.device)
        cuv = torch.ones(induc_E.shape[1]).to(X.device)
        for epoch in range(1, 1 + args.max_epochs):
            ev.start_sampling()
            for edges in edge_sampling(induc_E, eq, args.batch_size):
                subg_nodes = edges.unique(sorted=False)
                n_subg_nodes = subg_nodes.shape[0]
                subg_edges, val, edge_index = subg_construction(
                    crow, col, subg_nodes)
                cv[subg_nodes] += 1
                cuv[edge_index] += 1
                N += 1
                adj = coo(
                    subg_edges.to(X.device),
                    val.to(X.device) * cv[subg_edges[0]] / cuv[edge_index],
                    (n_subg_nodes, n_subg_nodes))
                ev.stop_sampling()
                net.A = [adj, adj]
                mask = train_mask[subg_nodes]
                ids = subg_nodes[mask]
                with opt:
                    (N / cv[ids] * criterion(
                        net(X[subg_nodes])[mask], Y[ids])).mean().backward()
                ev.record_training(opt.elapsed)
                ev.start_sampling()
            ev.stop_sampling()
            if ev.evaluate_model(net, args):
                break
    elif args.method == 'ECN' and (
            args.skip_connection or args.embedding or args.attention):
        net = Net(n_features, args.hidden, n_labels, **args.__dict__)
        opt = Optim([*net.parameters()])
        layer1, layer2 = net.layers
        if args.no_importance_sampling:
            def sampler(e, c, b):
                for p in DataLoader(
                        range(c.shape[0]), b, True):
                    yield e[:, p], c[p]
        else:
            sampler = lambda e, c, b: edge_sampling(
                e, c, b, return_weight=True)
        for epoch in range(1, 1 + args.max_epochs):
            ev.start_sampling()
            for edges, w in sampler(train_E, train_C, args.batch_size):
                ev.stop_sampling()
                with opt:
                    ids, inv = edges.unique(
                        sorted=False, return_inverse=True)
                    hi, hj = layer1(induc_P[ids])[inv]
                    z = layer2(hj)
                    if args.attention:
                        z = z.view(-1, args.attention, n_labels)
                        zi = layer2(hi).view(*z.shape)
                        a = net.batch_attention(zi, z, w)
                        if args.skip_connection:
                            z = z + net.skip_layer(hi).unsqueeze(1)
                        z = (z * a.unsqueeze(-1)).mean(dim=1)
                    elif args.skip_connection:
                        z = z + net.skip_layer(hi)
                    loss = criterion(z, Y[edges[0]])
                    if args.no_importance_sampling:
                        loss = loss * w
                    if args.embedding:
                        loss = loss.mean() + args.embedding * (
                            ce(hi, hj)
                            + ce(hj, -hj[torch.randperm(hj.shape[0])]))
                    loss.mean().backward()
                ev.record_training(opt.elapsed)
                ev.start_sampling()
            ev.stop_sampling()
            if ev.evaluate_model(net, args):
                break
    elif args.method == 'ECN' and args.no_importance_sampling:
        net = Net(n_features, args.hidden, n_labels, **args.__dict__)
        opt = Optim([*net.parameters()])
        for epoch in range(1, 1 + args.max_epochs):
            ev.start_sampling()
            for perm in DataLoader(
                    range(train_C.shape[0]), args.batch_size, True):
                ev.stop_sampling()
                with opt:
                    src, dst = train_E[:, perm]
                    (criterion(net(induc_P[dst]), Y[src])
                     * train_C[perm]).mean().backward()
                ev.record_training(opt.elapsed)
                ev.start_sampling()
            ev.stop_sampling()
            if ev.evaluate_model(net, args):
                break
    elif args.method == 'ECN':
        net = Net(n_features, args.hidden, n_labels, **args.__dict__)
        opt = Optim([*net.parameters()])
        for epoch in range(1, 1 + args.max_epochs):
            ev.start_sampling()
            for src, dst in edge_sampling(train_E, train_C, args.batch_size):
                ev.stop_sampling()
                with opt:
                    criterion(net(induc_P[dst]), Y[src]).mean().backward()
                ev.record_training(opt.elapsed)
                ev.start_sampling()
            ev.stop_sampling()
            if ev.evaluate_model(net, args):
                break
    else:
        assert False, 'Unknown Method'
        # Label Propagation
        # alpha = 0.4
        # Z = gpu(torch.zeros(n_nodes, n_labels))
        # train_probs = gpu(F.one_hot(Y, n_labels)[train_mask].float())
        # for epoch in range(1, 51):
        #     Z[train_mask] = train_probs
        #     t = time.time()
        #     Z = (1 - alpha) * Z + alpha * spmm(A, Z, args.batch_size)
        #     ev.record_evaluation(time.time() - t)
        #     if ev.evaluate_result(Z):
        #         break
    ev.end_run()
ev.end_all()
