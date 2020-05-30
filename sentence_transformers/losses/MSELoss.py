import torch
import torch.nn.functional  as F
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict


class MSELoss(nn.Module):
    """
    Computes the MSE loss between the computed sentence embedding and a target sentence embedding
    """
    def __init__(self, model):
        super(MSELoss, self).__init__()
        self.model = model

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        rep = self.model(sentence_features[0])['sentence_embedding']
        loss_fct = nn.MSELoss()
        loss = loss_fct(rep, labels)
        return loss

class MSEPairLoss(nn.Module):
    """
    Computes the MSE loss between the computed sentence embedding and a target sentence embedding
    """
    def __init__(self, model):
        super(MSEPairLoss, self).__init__()
        self.model = model

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        rep_a, rep_b = reps
        rep_a = rep_a / torch.norm(rep_a, p=2, dim=1).unsqueeze(1).detach()
        rep_b = rep_b / torch.norm(rep_a, p=2, dim=1).unsqueeze(1).detach()

        distance = ((rep_a - rep_b) ** 2).mean(dim=1)
        loss_fct = nn.MSELoss()
        loss = loss_fct(distance, labels)
        return loss

class MSESTSLoss(nn.Module):
    """
    Computes the MSE loss between the computed sentence embedding and a target sentence embedding
    """
    def __init__(self, model):
        super(MSESTSLoss, self).__init__()
        self.model = model
        self._MAX_MARGIN = 0.5
        self._margin = torch.FloatTensor([self._MAX_MARGIN]).to("cuda")
        self._zero = torch.FloatTensor([0.0]).to("cuda")

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor, scale: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        rep_a, rep_b = reps
        rep_a = rep_a / torch.norm(rep_a, p=2, dim=1).unsqueeze(1).detach()
        rep_b = rep_b / torch.norm(rep_a, p=2, dim=1).unsqueeze(1).detach()

        cos = torch.cosine_similarity(rep_a, rep_b)
        loss_fct = nn.MSELoss()
        loss = loss_fct(cos, labels)

        distance = ((rep_a - rep_b) ** 2).mean(dim=1)
        t_loss = scale * distance + (1.0 - scale) * torch.max(self._zero.expand_as(distance),
                                                              self._margin.expand_as(distance) - distance)
        t_loss = torch.max(scale * distance, self._margin.expand_as(distance))
        loss += torch.mean(t_loss)
        return loss

# class MSEAlignSTSLoss(nn.Module):
#     """
#     Computes the MSE loss between the computed sentence embedding and a target sentence embedding
#     """
#     def __init__(self, model):
#         super(MSEAlignSTSLoss, self).__init__()
#         self.model = model
#         self._h_size = 4096
#         self._pr = nn.Linear(768, self._h_size)
#         self.output = nn.Linear(self._h_size, 1)
#
#     def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
#         reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
#         rep_a, rep_b = reps
#         rep = F.leaky_relu(self._pr(rep_a - rep_b))
#         rep = self.output(rep)
#         loss_fct = nn.MSELoss()
#         loss = loss_fct(rep, labels)
#         return loss

class MSEAlignSTSLoss(nn.Module):
    """
    Computes the MSE loss between the computed sentence embedding and a target sentence embedding
    """
    def __init__(self, model):
        super(MSEAlignSTSLoss, self).__init__()
        self.model = model
        self._h_size = 4096
        self._pr = nn.Linear(768, self._h_size)
        self.output = nn.Linear(self._h_size, 1)

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        rep_a, rep_b = reps
        rep = F.leaky_relu(self._pr(rep_a - rep_b))
        rep = self.output(rep)
        loss_fct = nn.MSELoss()
        loss = loss_fct(rep, labels)
        return loss
