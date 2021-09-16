import pdb

from classifiers.ActionClassifier import ActionClassifier
import numpy as np
import os
from Configuration import *
from torch import nn
from optimisers.optimisers import SGAdaHMC
from shared.helpers import *
from classifiers.ThreeLayerMLP import ThreeLayerMLP
from classifiers.STGCN import STGCN
from classifiers.CTRGCN import CTRGCN
from datasets.dataloaders import *


class ExtendedBayesianClassifier(ActionClassifier):
    def __init__(self, args):
        super().__init__(args)
        args.bayesianTraining = True

        self.trainloader, self.testloader = createDataLoader(args)
        if args.baseClassifier == '3layerMLP':
            self.classifier = ThreeLayerMLP(args)
        elif args.baseClassifier == 'STGCN':
            self.classifier = STGCN(args)
        elif args.baseClassifier == 'CTRGCN':
            self.classifier = CTRGCN(args)

        self.classifier.setEval()
        self.retFolder = self.args.retPath + '/' + self.args.dataset + '/' + self.args.classifier + '/' + self.args.baseClassifier + '/'
        self.createModel()

    def createModel(self):
        class AppendedModel(nn.Module):
            def __init__(self, classifier):
                super().__init__()
                self.classifier = classifier
                self.model = torch.nn.Sequential(
                    #torch.nn.Linear(self.classifier.model.featureSize, self.classifier.args.classNum),
                    torch.nn.Linear(self.classifier.args.classNum, self.classifier.args.classNum),
                    torch.nn.Linear(self.classifier.args.classNum, self.classifier.args.classNum),
                    torch.nn.ReLU()
                )
            def forward(self, x):
                x = torch.nn.ReLU()(self.classifier.model(x))
                #logits = self.model(torch.flatten(self.classifier.model.features, start_dim=1))
                logits = self.model(x)
                logits = x + logits
                return logits


        self.modelList = [AppendedModel(self.classifier) for i in range(self.args.bayesianModelNum)]
        for model in self.modelList:
            model.to(device)
            model.model.train()

        if len(self.args.trainedAppendedModelFile) > 0:
            for i in range(self.args.bayesianModelNum):
                self.modelList[i].model.load_state_dict(
                    torch.load(self.retFolder + '/' + str(i) + '_' + self.args.trainedAppendedModelFile))

        self.configureOptimiser()
        self.classificationLoss()
    def configureOptimiser(self):

        self.optimiserList = [SGAdaHMC(self.modelList[i].model.parameters(), config=dict(lr=self.args.learningRate, alpha=0, gamma=0.01, L=30, T=1e-5, tao=2, C=1))
                              for i in range(self.args.bayesianModelNum)]

    def classificationLoss(self):

        self.classLoss = torch.nn.CrossEntropyLoss()

    def modelEval(self, X, modelNo = -1):
        if modelNo == -1:
            pred = torch.zeros((len(X), self.classifier.args.classNum), dtype=torch.float).to(device)
            for model in self.modelList:
                pred += model(X)
        else:
            pred = self.modelList[modelNo](X)
        return pred

    def classificationLoss(self):

        self.classLoss = torch.nn.CrossEntropyLoss()

    def setTrain(self, modelNo = -1):
        if modelNo == -1:
            for model in self.modelList:
                model.model.train()
        else:
            self.modelList[modelNo].train()

    def setEval(self, modelNo = -1):
        if modelNo == -1:
            for model in self.modelList:
                model.model.eval()
        else:
            self.modelList[modelNo].eval()
    # train does nothing here as this classifier can only be trained in the EBMATrainer
    def train(self):
        return
    def test(self):

        misclassified = 0
        results = np.empty(len(self.classifier.testloader.dataset.rlabels))
        for v, (tx, ty) in enumerate(self.classifier.testloader):
            predY = self.modelEval(tx)
            predY = torch.argmax(predY, dim=1)
            results[v * self.args.batchSize:(v + 1) * self.args.batchSize] = predY.cpu()
            diff = (predY - ty) != 0
            misclassified += torch.sum(diff)

        error = misclassified / len(self.classifier.testloader.dataset)
        print(f"accuracy: {1-error:>4f}")
        np.savetxt(self.retFolder + 'testRets.txt', results)
        np.savetxt(self.retFolder + 'testGroundTruth.txt', self.classifier.testloader.dataset.rlabels)

        return
    # this function is to collected all the testing samples that can be correctly collected
    # by the pre-trained classifier, to make a dataset for adversarial attack
    def collectCorrectPredictions(self):

        # collect data from the training data
        misclassified = 0
        results = np.empty(len(self.classifier.testloader.dataset.rlabels))
        for v, (tx, ty) in enumerate(self.classifier.testloader):
            pred = torch.argmax(self.modelEval(tx), dim=1)
            diff = (pred - ty) == 0
            results[v * self.args.batchSize:(v + 1) * self.args.batchSize] = diff.cpu()

        adData = self.classifier.testloader.dataset.data[results.astype(bool)]
        adLabels = self.classifier.testloader.dataset.rlabels[results.astype(bool)]

        print(f"{len(adLabels)} out of {len(results)} motions are collected")
        path = self.retFolder
        if not os.path.exists(path):
            os.makedirs(path)
        np.savez_compressed(path + self.args.bayesianAdTrainFile, clips=adData, classes=adLabels)

        return len(adLabels)