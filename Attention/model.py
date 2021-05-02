import torch
from torch import nn
import torch.nn.init as init
import math

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def distance(i, j, metric='l2'):
    return (i - j) ** 2 if metric == 'l2' else abs(i - j)


class WSDModel(nn.Module):

    def __init__(self, V, Y, D=300, dropout_prob=0.2, use_padding=False, use_positional=False, use_causal=False):
        super(WSDModel, self).__init__()
        self.use_padding = use_padding
        self.use_positional = use_positional
        self.use_causal = use_causal

        self.D = D
        self.pad_id = 0
        self.E_v = nn.Embedding(V, D, padding_idx=self.pad_id)
        self.E_y = nn.Embedding(Y, D, padding_idx=self.pad_id)
        init.kaiming_uniform_(self.E_v.weight[1:], a=math.sqrt(5))
        init.kaiming_uniform_(self.E_y.weight[1:], a=math.sqrt(5))

        self.W_A = nn.Parameter(torch.Tensor(D, D))
        self.W_O = nn.Parameter(torch.Tensor(D, D))
        init.kaiming_uniform_(self.W_A, a=math.sqrt(5))
        init.kaiming_uniform_(self.W_O, a=math.sqrt(5))

        self.dropout_layer = nn.Dropout(p=dropout_prob)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.layer_norm = nn.LayerNorm([self.D])

    def attention(self, X, Q, mask):
        """
        Computes the contextualized representation of query Q, given context X, using the attention model.

        :param X:
            Context matrix of shape [B, N, D]
        :param Q:
            Query matrix of shape [B, k, D], where k equals 1 (in single word attention) or N (self attention)
        :param mask:
            Boolean mask of size [B, N] indicating padding indices in the context X.

        :return:
            Contextualized query and attention matrix / vector
        """

        softmax_arg = torch.bmm(torch.matmul(Q, self.W_A), torch.transpose(X, 1, 2))
        B, k, N = softmax_arg.size()
        if self.use_positional:
            max_dist = distance(0, N)
            positions = torch.tensor([
                [
                    -max_dist if self.use_causal and j > i else -distance(i, j)
                    for j in range(N)
                ]
                for i in range(k)
            ], device=device)
            positions = positions.unsqueeze(-1).expand(-1, -1, B).permute(2, 0, 1)
            softmax_arg = softmax_arg + positions
        if self.use_padding:
            mask = mask.unsqueeze(-1)
            if k > 1:
                mask = mask.expand([B, k, N])
            mask = mask.permute(0, 2, 1)
            softmax_arg[~mask] = float('-inf')

        A = torch.softmax(softmax_arg, dim=2)
        Q_c = torch.matmul(torch.bmm(A, X), self.W_O)

        return Q_c, A.squeeze()

    def forward(self, M_s, v_q=None):
        """
        :param M_s:
            [B, N] dimensional matrix containing token integer ids
        :param v_q:
            [B] dimensional vector containing query word indices within the sentences represented by M_s.
            This argument is only passed in single word attention mode.

        :return: logits and attention tensors.
        """

        X = self.dropout_layer(self.E_v(M_s))
        B, N, D = X.size()
        if v_q is not None:
            v_q = v_q.unsqueeze(-1).expand(-1, N).unsqueeze_(-1).expand(-1, -1, D)
            Q = X.gather(1, v_q).mean(1).unsqueeze(1)
        else:
            Q = X

        mask = M_s.ne(self.pad_id)
        Q_c, A = self.attention(X, Q, mask)
        H = self.layer_norm(Q_c + Q)

        E_y = self.dropout_layer(self.E_y.weight)
        y_logits = (H @ E_y.T).squeeze()
        return y_logits, A
