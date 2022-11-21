import pdb

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from datasets.CDataset import *
import numpy as np


def createDataLoader(args):
    trainloader = ''
    testloader = ''

    if args.dataset == 'hdm05' or args.dataset == 'ntu60' or args.dataset == 'ntu120':
        if args.routine == 'train' or args.routine == 'adTrain' or args.routine == 'bayesianTrain':
            if args.dataset == 'ntu60' or args.dataset == 'ntu120':
                traindataset = NTUDataset(args)
            else:
                traindataset = CDataset(args)

            trainloader = DataLoader(traindataset, batch_size=args.batchSize, shuffle=True, drop_last=False)

            if len(args.testFile):
                routine = args.routine
                args.routine = 'test'
                if args.dataset == 'ntu60' or args.dataset == 'ntu120':
                    testdataset = NTUDataset(args)
                else:
                    testdataset = CDataset(args)
                args.routine = routine
                testloader = DataLoader(testdataset, batch_size=args.batchSize, shuffle=False, drop_last=False)

        elif args.routine == 'test' or args.routine == 'gatherCorrectPrediction' or args.routine == 'bayesianTest':
            if len(args.testFile):
                if args.dataset == 'ntu60' or args.dataset == 'ntu120':
                    testdataset = NTUDataset(args)
                else:
                    testdataset = CDataset(args)

                testloader = DataLoader(testdataset, batch_size=args.batchSize, shuffle=False, drop_last=False)


        elif args.routine == 'attack':
            if args.dataset == 'ntu60' or args.dataset == 'ntu120':
                traindataset = NTUDataset(args)
            else:
                traindataset = CDataset(args)

            trainloader = DataLoader(traindataset, batch_size=args.batchSize, shuffle=True, drop_last=True)

    else:
        print ('No dataset is loaded')

    return trainloader, testloader

