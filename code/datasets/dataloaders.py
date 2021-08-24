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
            trainloader = DataLoader(traindataset, batch_size=args.batchSize, shuffle=False, drop_last=True)
    else:
        print ('No dataset is loaded in ThreeLayerMLPArgs')

    return trainloader, testloader

<<<<<<< HEAD
def createDataLoader_sgn(args):
    trainloader = ''
    validationloader = ''
    path_train = ''
    path_val = ''
    testloader = ''
    if args.dataset == 'hdm05' or args.dataset == 'ntu60':
        if args.routine == 'train' or args.routine == 'adTrain' or args.routine == 'bayesianTrain':
            path_train = args.dataPath + '/' + args.dataset + '/' + args.trainFile
            path_val = args.dataPath + '/' + args.dataset + '/' + args.testFile
            ntu_loaders = NTUDataLoaders(args=args, pt=path_train, pv=path_val, dataset=args.dataset,aug=0)  # 0:CS 1:CV
            trainloader = ntu_loaders.get_train_loader(args.batchSize,0)
            if len(args.testFile):
                testloader = ntu_loaders.get_val_loader(args.batchSize,0)
        elif args.routine == 'attack':
            path_val = args.retPath + '/' + args.dataset + '/' + args.classifier + '/' + args.trainFile
            ntu_loaders = NTUDataLoaders(args=args, pt=path_train, pv=path_val, dataset=args.dataset)  # 0:CS 1:CV
            trainloader = ntu_loaders.get_val_loader(args.batchSize,0)
        elif args.routine == 'test' or args.routine == 'gatherCorrectPrediction' or args.routine == 'bayesianTest':
            path_val = args.dataPath + '/' + args.dataset + '/' + args.testFile
            ntu_loaders = NTUDataLoaders(args=args, pt=path_train, pv=path_val, dataset=args.dataset)  # 0:CS 1:CV
            if len(args.testFile):
                testloader = ntu_loaders.get_val_loader(args.batchSize,0)
    else:
        print ('No dataset is loaded')
=======
>>>>>>> parent of 41f8526 (added SGN)

