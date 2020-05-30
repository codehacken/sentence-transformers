import torch
import torch.nn.functional  as F
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict
from ..SentenceTransformer import SentenceTransformer

class CosineSimilarityLoss(nn.Module):
    def __init__(self, model: SentenceTransformer):
        super(CosineSimilarityLoss, self).__init__()
        self.model = model


    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        rep_a, rep_b = reps

        output = torch.cosine_similarity(rep_a, rep_b)
        loss_fct = nn.MSELoss()

        if labels is not None:
            loss = loss_fct(output, labels.view(-1))
            return loss
        else:
            return reps, output

class CosineMSEAlignSimilarityLoss(nn.Module):
    def __init__(self, model: SentenceTransformer):
        super(CosineMSEAlignSimilarityLoss, self).__init__()

        # Additional layers.
        self.model = model
        self._MAX_MARGIN = 0.1
        self._h_size = 768
        self._ph1 = nn.Linear(self._h_size, 1024)
        self._pr = nn.Linear(1024, self._h_size)
        self._par1 = nn.Linear(self._h_size, self._h_size)
        self._par2 = nn.Linear(self._h_size, self._h_size)
        self._margin = torch.FloatTensor([self._MAX_MARGIN]).to("cuda")
        self._zero = torch.FloatTensor([0.0]).to("cuda")

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor, scale: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        rep_a, rep_b = reps

        if labels is not None:
            # ph_a = F.leaky_relu(self._ph1(rep_a))
            # rep_a = self._pr(ph_a) # Get the projection of a->b.

            # Additional Single layers.
            fin_a = self._par1(rep_a)
            fin_b = self._par2(rep_b)
            output = torch.cosine_similarity(fin_a, fin_b)
            loss_fct = nn.MSELoss()

            # Contrastive loss.
            # distance = torch.norm(rep_a - rep_b, dim=1)
            # t_loss = scale * distance + (1.0 - scale) * torch.max(self._zero.expand_as(distance),
            #                                                       self._margin.expand_as(distance) - distance)
            # t_loss = torch.max(scale * distance, self._margin.expand_as(distance))
            # c_loss = torch.mean(t_loss)
            loss = loss_fct(output, labels.view(-1))
            # print(loss, c_loss)
            # loss = loss + 0.005 * c_loss
            return loss
        else:
            return reps, output

# class CosineMSEAlignSimilarityLoss(nn.Module):
#     def __init__(self, model: SentenceTransformer):
#         super(CosineMSEAlignSimilarityLoss, self).__init__()
#
#         # Additional layers.
#         self.model = model
#         self._MAX_MARGIN = 0.1
#         self._h_size = 768
#         self._ph1 = nn.Linear(self._h_size, 1024)
#         self._pr = nn.Linear(1024, self._h_size)
#         self._par1 = nn.Linear(self._h_size, self._h_size)
#         self._par2 = nn.Linear(self._h_size, self._h_size)
#         self._margin = torch.FloatTensor([self._MAX_MARGIN]).to("cuda")
#         self._zero = torch.FloatTensor([0.0]).to("cuda")
#
#     def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor, scale: Tensor):
#         reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
#         rep_a, rep_b = reps
#
#         if labels is not None:
#             # ph_a = F.leaky_relu(self._ph1(rep_a))
#             # rep_a = self._pr(ph_a) # Get the projection of a->b.
#
#             # Additional Single layers.
#             fin_a = self._par1(rep_a)
#             fin_b = self._par2(rep_b)
#             output = torch.cosine_similarity(fin_a, fin_b)
#             loss_fct = nn.MSELoss()
#
#             # Contrastive loss.
#             # distance = torch.norm(rep_a - rep_b, dim=1)
#             # t_loss = scale * distance + (1.0 - scale) * torch.max(self._zero.expand_as(distance),
#             #                                                       self._margin.expand_as(distance) - distance)
#             # t_loss = torch.max(scale * distance, self._margin.expand_as(distance))
#             # c_loss = torch.mean(t_loss)
#             loss = loss_fct(output, labels.view(-1))
#             # print(loss, c_loss)
#             # loss = loss + 0.005 * c_loss
#             return loss
#         else:
#             return reps, output
