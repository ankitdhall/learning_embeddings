import torch
from torch import nn
from data.db import ETHECLabelMap


class MultiLevelCELoss(torch.nn.Module):
    def __init__(self, labelmap, level_weights=None, weight=None):
        torch.nn.Module.__init__(self)
        self.labelmap = labelmap
        self.level_weights = [1.0] * len(self.labelmap.levels) if level_weights is None else level_weights
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = []
        if weight is None:
            for level_len in self.labelmap.levels:
                self.criterion.append(nn.CrossEntropyLoss(weight=None, reduction='none'))
        else:
            level_stop, level_start = [], []
            for level_id, level_len in enumerate(self.labelmap.levels):
                if level_id == 0:
                    level_start.append(0)
                    level_stop.append(level_len)
                else:
                    level_start.append(level_stop[level_id - 1])
                    level_stop.append(level_stop[level_id - 1] + level_len)
                self.criterion.append(nn.CrossEntropyLoss(weight=weight[level_start[level_id]:level_stop[level_id]].to(self.device),
                                                          reduction='none'))

        print('==Using the following weights config for multi level cross entropy loss: {}'.format(self.level_weights))

    def forward(self, outputs, labels, level_labels):
        loss = 0.0
        for level_id, level in enumerate(self.labelmap.levels):
            if level_id == 0:
                loss += self.level_weights[level_id] * self.criterion[level_id](outputs[:, 0:level], level_labels[:, level_id])
            else:
                start = sum([self.labelmap.levels[l_id] for l_id in range(level_id)])
                loss += self.level_weights[level_id] * self.criterion[level_id](outputs[:, start:start + level],
                                                                          level_labels[:, level_id])
        return torch.mean(loss)


class MultiLabelSMLoss(torch.nn.MultiLabelSoftMarginLoss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean'):
        print(weight)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if weight is not None:
            weight = weight.to(self.device)
        torch.nn.MultiLabelSoftMarginLoss.__init__(self, weight, size_average, reduce, reduction)

    def forward(self, outputs, labels, level_labels):
        return super().forward(outputs, labels)


if __name__ == '__main__':
    lmap = ETHECLabelMap()
    criterion = MultiLevelCELoss(labelmap=lmap, level_weights=[1, 1, 1, 1])
    output, level_labels = torch.zeros((1, lmap.n_classes)), torch.tensor([[0,
                                                                            7-lmap.levels[0],
                                                                            90-(lmap.levels[0]+lmap.levels[1]),
                                                                            400-(lmap.levels[0]+lmap.levels[1]+lmap.levels[2])]])
    labels = torch.zeros((1, lmap.n_classes))
    labels[0, torch.tensor([0, 7, 90, 400])] = 1
    output[:, 0] = 100
    output[:, 7] = 100
    output[:, 90] = 10000
    output[:, 400] = 10000
    print(output)
    print(labels)
    print(level_labels)
    print('MLCELoss: {}'.format(criterion(output, labels, level_labels)))

    criterion_multi_label = torch.nn.MultiLabelSoftMarginLoss()
    custom_criterion_multi_label = MultiLabelSMLoss()
    print('MLSMLoss: {}'.format(criterion_multi_label(output, labels)))
    print('MLSMLoss: {}'.format(custom_criterion_multi_label(output, labels, level_labels)))


