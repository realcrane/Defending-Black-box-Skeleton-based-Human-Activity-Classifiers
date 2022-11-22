

from classifiers.ActionClassifier import ActionClassifier
import numpy as np
import os
from Configuration import *
from torch import nn

from shared.helpers import *
from torch.utils.tensorboard import SummaryWriter

from datasets.dataloaders import *


class ThreeLayerMLP(ActionClassifier):
    def __init__(self, args):
        super().__init__(args)
        self.trainloader, self.testloader = createDataLoader(args)
        self.createModel()

    def createModel(self):
        class Classifier(nn.Module):
            def __init__(self, args, dataloader):
                super().__init__()

                self.dataShape = dataloader.dataset.data.shape
                self.flatten = nn.Flatten()
                self.mlpstack = nn.Sequential(
                    nn.Linear(self.dataShape[1] * self.dataShape[2], 512),
                    nn.ReLU(),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, dataloader.dataset.classNum),
                    nn.ReLU()
                )
            def forward(self, x):
                x = self.flatten(x)
                logits = self.mlpstack(x)
                return logits

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

        self.retFolder = self.args.retPath + self.args.dataset + '/' + self.args.classifier + '/'

        if len(self.args.trainedModelFile) > 0:
            if len(self.args.adTrainer) == 0:
                self.model.load_state_dict(torch.load(self.retFolder + self.args.trainedModelFile))
            else:
                if self.args.bayesianTraining:
                    self.model.load_state_dict(torch.load(self.args.retPath + self.args.dataset + '/' +
                                        self.args.baseClassifier + '/' + self.args.trainedModelFile))
                else:
                    self.model.load_state_dict(
                        torch.load(self.retFolder + self.args.adTrainer + '/' + self.args.trainedModelFile))

        self.configureOptimiser()
        self.classificationLoss()
        self.model.to(device)

    def configureOptimiser(self):

        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=self.args.learningRate, betas=(0.9, 0.999),
                                          eps=1e-08, weight_decay=0.0001, amsgrad=False)

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
        bestValLoss = np.infty

        logger = SummaryWriter()

        for ep in range(self.args.epochs):
            epLoss = 0
            batchNum = 0
            for batch, (X, y) in enumerate(self.trainloader):
                batchNum += 1
                # Compute prediction and loss
                pred = self.model(X)
                loss = self.classLoss(pred, y)

                epLoss += loss.detach().item()
                # Backpropagation
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

                if batch % 50 == 0:
                    loss, current = loss.item(), batch * len(X)
                    print(f"epoch: {ep}  loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

            # save a model if the best training loss so far has been achieved.
            epLoss /= batchNum
            logger.add_scalar('Loss/train', epLoss, ep)
            if epLoss < bestLoss:
                if not os.path.exists(self.retFolder):
                    os.makedirs(self.retFolder)
                print(f"epoch: {ep} per epoch average training loss improves from: {bestLoss} to {epLoss}")
                torch.save(self.model.state_dict(), self.retFolder + '/'+'minLossModel.pth')
                bestLoss = epLoss

            # run validation and save a model if the best validation loss so far has been achieved.
            valLoss = 0
            vbatch = 0
            self.model.eval()
            for v, (tx, ty) in enumerate(self.testloader):
                pred = self.model(tx)
                valLoss += self.classLoss(pred, ty)
                valLoss += loss.detach().item()
                vbatch += 1

            valLoss /= vbatch
            logger.add_scalar('Loss/validation', valLoss, ep)
            self.model.train()
            if valLoss < bestValLoss:
                print(f"epoch: {ep} per epoch average validation loss improves from: {bestValLoss} to {valLoss}")
                torch.save(self.model.state_dict(), self.retFolder + 'minValLossModel.pth')
                bestValLoss = valLoss

    #this function tests the trained classifier and also save correctly classified samples in 'adClassTrain.npz' for
    #further adversarial attack
    def test(self):
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
        np.savetxt(self.args.retPath + '/' + self.args.dataset + '/'+ self.args.classifier + '/' + 'testRets.txt', results)
        np.savetxt(self.args.retPath +'/'+ self.args.dataset + '/' + self.args.classifier + '/' + 'testGroundTruth.txt', self.testloader.dataset.rlabels)

        return 1 - error

    # this function is to collected all the testing samples that can be correctly collected
    # by the pre-trained classifier, to make a dataset for adversarial attack
    def collectCorrectPredictions(self):
        if len(self.args.trainedModelFile) == 0 or self.testloader == '':
            print('no pre-trained model to load')
            return


        # collect data from the training data
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




