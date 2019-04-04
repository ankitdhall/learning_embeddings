import torch
from torch import nn
from data.db import ETHECLabelMap


class MultiLevelCELoss(torch.nn.Module):
    def __init__(self, labelmap, weights=None):
        torch.nn.Module.__init__(self)
        self.labelmap = labelmap
        self.weights = [1.0]*len(self.labelmap.levels) if weights is None else weights
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        print('==Using the following weights config for multi level cross entropy loss: {}'.format(self.weights))

    def forward(self, outputs, labels, level_labels):
        # print('Outputs: {}'.format(outputs))
        # print('Level labels: {}'.format(level_labels))
        # print('Levels: {}'.format(self.labelmap.levels))
        loss = 0.0
        for level_id, level in enumerate(self.labelmap.levels):
            if level_id == 0:
                loss += self.weights[level_id] * self.criterion(outputs[:, 0:level], level_labels[:, level_id])
                # print(self.weights[level_id] * self.criterion(outputs[:, 0:level], level_labels[:, level_id]))
            else:
                start = sum([self.labelmap.levels[l_id] for l_id in range(level_id)])
                # print([self.labelmap.levels[l_id] for l_id in range(level_id)], level)
                # print(outputs[:, start:start+level])
                # print(self.weights[level_id] * self.criterion(outputs[:, start:start+level],
                #                                                 level_labels[:, level_id]))
                loss += self.weights[level_id] * self.criterion(outputs[:, start:start+level],
                                                                level_labels[:, level_id])
        # print('Loss per sample: {}'.format(loss))
        # print('Avg loss: {}'.format(torch.mean(loss)))
        return torch.mean(loss)


class MultiLabelSMLoss(torch.nn.MultiLabelSoftMarginLoss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean'):
        torch.nn.MultiLabelSoftMarginLoss.__init__(self, weight, size_average, reduce, reduction)

    def forward(self, outputs, labels, level_labels):
        return super().forward(outputs, labels)


if __name__ == '__main__':
    lmap = ETHECLabelMap()
    criterion = MultiLevelCELoss(labelmap=lmap, weights=[1, 1, 1, 1])
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


