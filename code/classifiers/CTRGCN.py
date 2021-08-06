import pdb
from classifiers.ActionClassifier import ActionClassifier
from classifiers.ctrgcn.ctrgcn import Model
import numpy as np
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time
from datasets.dataloaders import *
from Configuration import *

class STGCN(ActionClassifier):
    def __init__(self, args):
        super().__init__(args)
        self.trainloader, self.testloader = createDataLoader(args)
        self.createModel()
        self.steps = [10, 50]
    def createModel(self):
        class Model(nn.Module):
            def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                         drop_out=0, adaptive=True):

        num_class = self.args.args.classNum
        num_point = 25
        num_person = 2
        graph = 'graph.ntu_rgb_d.Graph'
        graph_args = dict()
        in_channels = 3
        drop_out = 0
        adaptive = True

        # create the train data loader, if the routine is 'attack', then the data will be attacked in an Attacker
        if self.args.routine == 'train' or self.args.routine == 'attack' \
                or self.args.routine == 'adTrain' or self.args.routine == 'bayesianTrain':
            self.model = Classifier(self.args, self.trainloader)
        elif self.args.routine == 'test' or self.args.routine == 'gatherCorrectPrediction' \
                or self.args.routine == 'bayesianTest':
            self.model = Classifier(self.args, self.testloader)
            self.model.eval()
        else:
            print("no model is created")

        self.retFolder = self.args.retFolder + self.args.dataset + '/' + self.args.classifier + '/'

        if len(self.args.trainedModelFile) > 0:
            if len(self.args.args.adTrainer) == 0:
                self.model.load_state_dict(torch.load(self.retFolder + self.args.trainedModelFile))
            else:
                if self.args.args.bayesianTraining:
                    self.model.load_state_dict(torch.load(self.args.retFolder + self.args.dataset + '/' +
                                        self.args.args.baseClassifier + '/' + self.args.trainedModelFile))
                else:
                    self.model.load_state_dict(
                        torch.load(self.retFolder + self.args.args.adTrainer + '/' + self.args.trainedModelFile))

        self.configureOptimiser()
        self.classificationLoss()
        self.model.to(device)
    def configureOptimiser(self):

        #self.optimiser = torch.optim.Adam(self.model.parameters(), lr=self.args.args.learningRate, weight_decay=0.0001)
        self.optimiser = torch.optim.SGD(
                self.model.parameters(),
                lr=self.args.args.learningRate,
                momentum=0.9,
                nesterov=True,
                weight_decay=0.0001)

    def adjustLearningRate(self, epoch):
        if self.steps:
            lr = self.args.args.learningRate * (
                0.1**np.sum(epoch >= np.array(self.steps)))
            for param_group in self.optimiser.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.args.args.learningRate
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
        results = np.empty(len(self.testloader.dataset.rlabels))
        for ep in range(self.args.epochs):
            epLoss = 0
            batchNum = 0
            self.adjustLearningRate(ep)
            #print(f"epoch: {ep} GPU memory allocated: {torch.cuda.memory_allocated(1)}")
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


            #print(f"epoch: {ep} GPU memory allocated after one epoch: {torch.cuda.memory_allocated(1)}")
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
            if ep % 1 == 0:
                # run validation and save a model if the best validation loss so far has been achieved.
                valStartTime = time.time()
                misclassified = 0
                self.model.eval()
                for v, (tx, ty) in enumerate(self.testloader):
                    pred = torch.argmax(self.model(tx), dim=1)
                    results[v * self.args.batchSize:(v + 1) * self.args.batchSize] = pred.cpu()
                    diff = (pred - ty) != 0
                    misclassified += torch.sum(diff)

                    #print(f"epoch: {ep} GPU memory allocated after one batch validation: {torch.cuda.memory_allocated(1)}")
                acc = 1 - misclassified / len(self.testloader.dataset)
                logger.add_scalar('Loss/testing accuracy', acc, ep)
                self.model.train()
                if acc > bestValAcc:
                    print(f"epoch: {ep} per epoch average validation accuracy improves from: {bestValAcc} to {acc}")
                    torch.save(self.model.state_dict(), self.retFolder + 'minValLossModel.pth')
                    bestValAcc = acc
                valEndTime = time.time()
                valTime += (valEndTime - valStartTime)/3600
                #print(f"epoch: {ep} GPU memory allocated after one epoch validation: {torch.cuda.memory_allocated(1)}")

    #this function tests the trained classifier and also save correctly classified samples in 'adClassTrain.npz' for
    #further adversarial attack
    def test(self):
        self.model.eval()
        if len(self.args.trainedModelFile) == 0 or self.testloader == '':
            print('no pre-trained model to load')
            return

        misclassified = 0
        results = np.empty(len(self.testloader.dataset.rlabels))
        for v, (tx, ty) in enumerate(self.testloader):
            pred = torch.argmax(self.model(tx), dim=1)
            results[v*self.args.batchSize:(v+1)*self.args.batchSize] = pred.cpu()
            diff = (pred - ty) != 0
            misclassified += torch.sum(diff)

        error = misclassified / len(self.testloader.dataset)
        print(f"accuracy: {1-error:>4f}")
        np.savetxt(self.retFolder + 'testRets.txt', results)
        np.savetxt(self.retFolder + 'testGroundTruth.txt', self.testloader.dataset.rlabels)

        return 1 - error

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