import pdb
from Configuration import *

from torch.utils.data import Dataset
from shared.helpers import *
import numpy as np
class CDataset(Dataset):
    def __init__(self, args, transform=None, target_transform=None):
        data = ''
        if args.routine == 'train' or args.routine == 'adTrain' or args.routine == 'bayesianTrain':
            data = np.load(args.dataPath + '/' + args.dataset + '/' + args.trainFile)
        elif args.routine == 'attack':
            data = np.load(args.retPath + '/' + args.dataset + '/' + args.classifier + '/' + args.trainFile)
        elif args.routine == 'test' or args.routine == 'gatherCorrectPrediction' or args.routine == 'bayesianTest':
            data = np.load(args.dataPath + '/' + args.dataset + '/' + args.testFile)
        else:
            print('Unknown routine, cannot create the dataset')

        self.data = torch.from_numpy(data['clips'])
        self.rlabels = data['classes']
        self.labels = torch.from_numpy(self.rlabels).type(torch.int64)
        self.transform = transform
        self.target_transform = target_transform
        self.classNum = args.classNum
        self.args = args

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data[idx]
        ##data format conversion to be consistent with ntu
        if self.args.dataset == 'hdm05':
            x = data.reshape((data.shape[0], -1, 3, 1))
            data = x.permute(2, 0, 1, 3)

        label = self.labels[idx]
        return data.to(device), label.to(device)

    # this is the topology of the skeleton, assuming the joints are stored in an array and the indices below
    # indicate their parent node indices. E.g. the parent of the first node is 10 and node[10] is the root node
    # of the skeleton
    parents = np.array([10, 0, 1, 2, 3,
                        10, 5, 6, 7, 8,
                        10, 10, 11, 12, 13,
                        13, 15, 16, 17, 18,
                        13, 20, 21, 22, 23])


class NTUDataset(CDataset):
    def __init__(self, args, transform=None, target_transform=None):
        super().__init__(args, transform, target_transform)
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        return data.to(device), label.to(device)

    # this is the topology of the skeleton, assuming the joints are stored in an array and the indices below
    # indicate their parent node indices.
    parents = np.array([1, 1, 21, 3, 21,
                        5, 6, 7, 21, 9,
                        10, 11, 1, 13, 14,
                        15, 1, 17, 18, 19,
                        2, 8, 8, 12, 12]) - 1
