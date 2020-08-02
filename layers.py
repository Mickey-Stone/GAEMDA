import math
import mxnet as mx
from mxnet import ndarray as nd
from mxnet.gluon import nn
import dgl
import dgl.function as FN
import numpy as np


class GraphSageLayer(nn.Block):
    def __init__(self, feature_size, G, disease_nodes, mirna_nodes, dropout, slope, ctx):
        super(GraphSageLayer, self).__init__()

        self.feature_size = feature_size
        self.G = G
        self.disease_nodes = disease_nodes
        self.mirna_nodes = mirna_nodes
        self.ctx = ctx

        self.disease_update = NodeUpdate(feature_size, dropout, slope)
        self.miran_update = NodeUpdate(feature_size, dropout, slope)

        all_nodes = mx.nd.arange(G.number_of_nodes(), dtype=np.int64)
        self.deg = G.in_degrees(all_nodes).astype(np.float32).copyto(ctx)

    def forward(self, G):
        assert G.number_of_nodes() == self.G.number_of_nodes()
        G.ndata['deg'] = self.deg

        G.update_all(FN.copy_src('h', 'h'), FN.sum('h', 'h_agg'))  # mean, max, sum

        G.apply_nodes(self.disease_update, self.disease_nodes)
        G.apply_nodes(self.miran_update, self.mirna_nodes)


class NodeUpdate(nn.Block):
    def __init__(self, feature_size, dropout, slope):
        super(NodeUpdate, self).__init__()

        self.feature_size = feature_size
        self.leakyrelu = nn.LeakyReLU(slope)
        self.W = nn.Dense(feature_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes):
        h = nodes.data['h']
        h_agg = nodes.data['h_agg']
        deg = nodes.data['deg'].expand_dims(1)

        h_concat = nd.concat(h, h_agg / nd.maximum(deg, 1e-6), dim=1)
        # h_concat = nd.concat(h, h_agg, dim=1)
        h_new = self.dropout(self.leakyrelu(self.W(h_concat)))

        return {'h': h_new}
