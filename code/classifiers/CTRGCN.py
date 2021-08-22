import pdb
from classifiers.ActionClassifier import ActionClassifier
from classifiers.ctrgcn.ctrgcn import TCN_GCN_unit, bn_init
from classifiers.ctrgcn.ntu_rgb_d import Graph
import math
import numpy as np
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time
from datasets.dataloaders import *
from Configuration import *

class CTRGCN(ActionClassifier):
    def __init__(self, args):
        super().__init__(args)
        self.trainloader, self.testloader = createDataLoader(args)
        self.createModel()
        self.steps = [35, 55]
        self.warmUpEpoch = 5
    def createModel(self):
        class Classifier(nn.Module):
            def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                         drop_out=0, adaptive=True):
                super(Classifier, self).__init__()

                if graph is None:
                    raise ValueError()
                else:
                    #Graph = import_class(graph)
                    self.graph = Graph(labeling_mode='spatial')

                A = self.graph.A  # 3,25,25

                self.num_class = num_class
                self.num_point = num_point
                self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

                base_channel = 64
                self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
                self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
                self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
                self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
                self.l5 = TCN_GCN_unit(base_channel, base_channel * 2, A, stride=2, adaptive=adaptive)
                self.l6 = TCN_GCN_unit(base_channel * 2, base_channel * 2, A, adaptive=adaptive)
                self.l7 = TCN_GCN_unit(base_channel * 2, base_channel * 2, A, adaptive=adaptive)
                self.l8 = TCN_GCN_unit(base_channel * 2, base_channel * 4, A, stride=2, adaptive=adaptive)
                self.l9 = TCN_GCN_unit(base_channel * 4, base_channel * 4, A, adaptive=adaptive)
                self.l10 = TCN_GCN_unit(base_channel * 4, base_channel * 4, A, adaptive=adaptive)

                self.fc = nn.Linear(base_channel * 4, num_class)
                nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
                bn_init(self.data_bn, 1)
                if drop_out:
                    self.drop_out = nn.Dropout(drop_out)
                else:
                    self.drop_out = lambda x: x

            def forward(self, x):
                if len(x.shape) == 3:
                    N, T, VC = x.shape
                    x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
                N, C, T, V, M = x.size()

                x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
                x = self.data_bn(x)
                x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
                x = self.l1(x)
                x = self.l2(x)
                x = self.l3(x)
                x = self.l4(x)
                x = self.l5(x)
                x = self.l6(x)
                x = self.l7(x)
                x = self.l8(x)
                x = self.l9(x)
                x = self.l10(x)

                # N*M,C,T,V
                c_new = x.size(1)
                x = x.view(N, M, c_new, -1)
                x = x.mean(3).mean(1)
                x = self.drop_out(x)

                return self.fc(x)

        num_class = self.args.classNum
        num_point = 25
        if self.args.dataset == 'hdm05':
            num_person = 1
        elif self.args.dataset == 'ntu60' or self.args.dataset == 'ntu120':
            num_person = 2
        graph = 'classifiers.ctrgcn.ntu_rgb_d.Graph'
        graph_args = dict([('labeling_mode:', 'spatial')])
        in_channels = 3
        drop_out = 0
        adaptive = True

        # create the train data loader, if the routine is 'attack', then the data will be attacked in an Attacker
        if self.args.routine == 'train' or self.args.routine == 'attack' \
                or self.args.routine == 'adTrain' or self.args.routine == 'bayesianTrain':
            self.model = Classifier(num_class, num_point, num_person, graph, graph_args, in_channels, drop_out, adaptive)
        elif self.args.routine == 'test' or self.args.routine == 'gatherCorrectPrediction' \
                or self.args.routine == 'bayesianTest':
            self.model = Classifier(num_class, num_point, num_person, graph, graph_args, in_channels, drop_out, adaptive)
            self.model.eval()
        else:
            print("no model is created")

        self.retFolder = self.args.retPath + self.args.dataset + '/' + self.args.classifier + '/'

        if len(self.args.trainedModelFile) > 0:
            if len(self.args.adTrainer) == 0:
                self.model.load_state_dict(torch.load(self.retFolder + self.args.trainedModelFile))
            else:
                if self.args.bayesianTraining:
                    self.model.load_state_dict(torch.load(self.args.retPath + self.args.dataset + '/' +
                                        self.args.baseClassifier + '/' + self.args.trainedModelFile))
                else:
                    if len(self.args.initWeightFile) > 0:
                        self.model.load_state_dict(
                            torch.load(self.retFolder + self.args.initWeightFile))
                    else:
                        self.model.load_state_dict(
                        torch.load(self.retFolder + self.args.adTrainer + '/' + self.args.trainedModelFile))

        self.configureOptimiser()
        self.classificationLoss()
        self.model.to(device)
    def configureOptimiser(self):

        if self.args.optimiser == 'Adam':
            self.optimiser = torch.optim.Adam(self.model.parameters(), lr=self.args.learningRate, weight_decay=0.0001)
        elif self.args.optimiser == 'SGD':
            self.optimiser = torch.optim.SGD(
                    self.model.parameters(),
                    lr=self.args.learningRate,
                    momentum=0.9,
                    nesterov=True,
                    weight_decay=0.0004)

    def adjustLearningRate(self, epoch):

        if epoch < self.warmUpEpoch:
            lr = self.args.learningRate * (epoch + 1) / self.warmUpEpoch
        else:
            lr = self.args.learningRate * (
                    0.1 ** np.sum(epoch >= np.array(self.steps)))
        for param_group in self.optimiser.param_groups:
            param_group['lr'] = lr
        return lr


    def classificationLoss(self):

        self.classLoss = torch.nn.CrossEntropyLoss()

    def setTrain(self):
        self.model.train()
    def setEval(self):
        self.model.eval()

    def modelEval(self, X, modelNo = -1):
        return self.model(X)


    #this function is to train the classifier from scratch
    def train(self):
        size = len(self.trainloader.dataset)

        bestLoss = np.infty
        bestValAcc = 0

        logger = SummaryWriter()
        startTime = time.time()
        valTime = 0

        for ep in range(self.args.epochs):
            epLoss = 0
            batchNum = 0
            self.adjustLearningRate(ep)
            # print(f"epoch: {ep} GPU memory allocated: {torch.cuda.memory_allocated(1)}")
            for batch, (X, y) in enumerate(self.trainloader):
                batchNum += 1
                # Compute prediction and loss
                pred = self.model(X)
                loss = self.classLoss(pred, y)

                # Backpropagation
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

                epLoss += loss.detach().item()
                if batch % 100 == 0:
                    loss, current = loss.item(), batch * len(X)
                    print(f"epoch: {ep}  loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

            # print(f"epoch: {ep} GPU memory allocated after one epoch: {torch.cuda.memory_allocated(1)}")
            # save a model if the best training loss so far has been achieved.
            epLoss /= batchNum
            logger.add_scalar('Loss/train', epLoss, ep)
            if epLoss < bestLoss:
                if not os.path.exists(self.retFolder):
                    os.makedirs(self.retFolder)
                print(f"epoch: {ep} per epoch average training loss improves from: {bestLoss} to {epLoss}")
                torch.save(self.model.state_dict(), self.retFolder + 'minLossModel.pth')
                bestLoss = epLoss
            print(f"epoch: {ep} time elapsed: {(time.time() - startTime) / 3600 - valTime} hours")

            # run validation and save a model if the best validation loss so far has been achieved.
            valStartTime = time.time()
            misclassified = 0
            self.model.eval()
            for v, (tx, ty) in enumerate(self.testloader):
                pred = torch.argmax(self.model(tx), dim=1)
                diff = (pred - ty) != 0
                misclassified += torch.sum(diff)
            acc = 1 - misclassified / len(self.testloader.dataset)
            logger.add_scalar('Loss/testing accuracy', acc, ep)
            self.model.train()
            if acc > bestValAcc:
                print(f"epoch: {ep} per epoch average validation accuracy improves from: {bestValAcc} to {acc}")
                torch.save(self.model.state_dict(), self.retFolder + 'minValLossModel.pth')
                bestValAcc = acc
            valEndTime = time.time()
            valTime += (valEndTime - valStartTime) / 3600
            # print(f"epoch: {ep} GPU memory allocated after one epoch validation: {torch.cuda.memory_allocated(1)}")


    def test(self):
        self.model.eval()
        misclassified = 0
        results = np.empty(len(self.testloader.dataset.rlabels))
        for v, (tx, ty) in enumerate(self.testloader):
            pred = torch.argmax(self.model(tx), dim=1)
            results[v * self.args.batchSize:(v + 1) * self.args.batchSize] = pred.cpu()
            diff = (pred - ty) != 0
            misclassified += torch.sum(diff)

        acc = 1 - misclassified / len(self.testloader.dataset)

        print(f"accuracy: {acc:>4f}")
        np.savetxt(self.retFolder + 'testRets.txt', results)
        np.savetxt(self.retFolder + 'testGroundTruth.txt', self.testloader.dataset.rlabels)

        return acc

    # this function is to collected all the testing samples that can be correctly collected
    # by the pre-trained classifier, to make a dataset for adversarial attack
    def collectCorrectPredictions(self):
        self.model.eval()

        # collect data from the test data
        misclassified = 0
        results = np.empty(len(self.testloader.dataset.rlabels))
        for v, (tx, ty) in enumerate(self.testloader):
            pred = torch.argmax(self.model(tx), dim=1)
            diff = (pred - ty) == 0
            results[v * self.args.batchSize:(v + 1) * self.args.batchSize] = diff.cpu()

        adData = self.testloader.dataset.data[results.astype(bool)]
        adLabels = self.testloader.dataset.rlabels[results.astype(bool)]

        print(f"{len(adLabels)} out of {len(results)} motions are collected")

        if not os.path.exists(self.retFolder):
            os.mkdir(self.retFolder)
        np.savez_compressed(self.retFolder+self.args.adTrainFile, clips=adData, classes=adLabels)

        return len(adLabels)