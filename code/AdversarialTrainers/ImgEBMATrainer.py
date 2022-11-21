from Configuration import *
from AdversarialTrainers.AdversarialTrainer import AdversarialTrainer
from classifiers.loadClassifiers import loadClassifier
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
import time

class ImgEBMATrainer(AdversarialTrainer):
    def __init__(self, args):
        super().__init__(args)

        self.classifier = loadClassifier(args)

        self.retFolder = self.args.retPath + '/' + self.args.dataset + '/' + self.args.classifier + '/' + self.args.baseClassifier + '/'
        if args.bayesianTraining:
            self.replayBufferList = [[] for i in range(args.bayesianModelNum)]
        else:
            if len(args.bufferSamples) > 0:
                self.replayBuffer = torch.FloatTensor(np.load(args.bufferSamples)['clips'])
            else:
                self.replayBuffer = []

        if not os.path.exists(self.retFolder):
            os.makedirs(self.retFolder)

        self.configureOptimiser()

    def configureOptimiser(self):

        if not self.args.bayesianTraining:
            self.optimiser = torch.optim.Adam(self.classifier.model.parameters(), lr=self.args.learningRate,
                                              weight_decay=0.0001)

    def modelEval(self, X, modelNo=-1):

        if self.args.bayesianTraining:
            pred = self.classifier.modelEval(X, modelNo)
        else:
            pred = torch.nn.ReLU()(self.classifier.model(X))
        return pred

    def initRandom(self, X):
        return torch.FloatTensor(np.random.uniform(-1, 1, X.shape).astype('float32'))

    def sampleP0(self, X, rb=-1, y=None):
        # currently, we do not do conditioned sampling, y is not used
        if rb == -1:
            replayBuffer = self.replayBuffer
        else:
            replayBuffer = self.replayBufferList[rb]
        if len(replayBuffer) == 0:
            return self.initRandom(X), []
        self.args.bufferSize = len(replayBuffer) if y is None else len(replayBuffer)
        inds = torch.randint(0, self.args.bufferSize, (self.args.batchSize,))
        # if cond, convert inds to class conditional inds
        # if y is not None:
        #     inds = y.cpu() * self.args.bufferSize + inds
        bufferSamples = replayBuffer[inds]
        randomSamples = self.initRandom(X)
        choose_random = (torch.rand(self.args.batchSize) < self.args.reinitFreq).float()[:, None, None]
        samples = choose_random * randomSamples + (1 - choose_random) * bufferSamples
        return samples, inds

    def sampleX(self, X, rb=-1, y=None):
        if rb == -1:
            replayBuffer = self.replayBuffer
        else:
            replayBuffer = self.replayBufferList[rb]
        if not self.args.bayesianTraining:
            self.classifier.model.eval()
        # get batch size
        bs = self.args.batchSize if y is None else y.size(0)
        # generate initial samples and buffer inds of those samples (if buffer is used)
        initSample, bufferInds = self.sampleP0(X, rb)
        x_k = torch.autograd.Variable(initSample, requires_grad=True).to(device)
        # sgld
        for k in range(self.args.samplingStep):
            f_prime = torch.autograd.grad(torch.logsumexp(self.modelEval(x_k, rb), 0).sum(), [x_k], retain_graph=True)[0]
            x_k.data += self.args.sgldLr * f_prime + np.sqrt(
                self.args.sgldLr * 2) * self.args.sgldStd * torch.randn_like(x_k)
        if not self.args.bayesianTraining:
            self.classifier.model.train()

        final_samples = x_k.detach()
        # update replay buffer
        if len(replayBuffer) > 0:
            replayBuffer[bufferInds] = final_samples.cpu()
        return final_samples


    def xXTildeLoss(self, x, xTilde):
        loss = torch.nn.functional.mse_loss(x, xTilde)

        return loss

    def log_xTilde_x(self, x, xTilde, y, beta, modelNo = -1):

        loss_dis = self.xXTildeLoss(x, xTilde)

        #use cross-entropy
        loss = torch.nn.CrossEntropyLoss()(self.modelEval(xTilde, modelNo), y)

        return -(loss + beta * loss_dis)

    def sampleXTilde(self, x, y, modelNo=-1):
        if not self.args.bayesianTraining:
            self.classifier.model.eval()
        # based on data x, generate perturbed data x_tilde
        # import time
        # start_time = time.time()
        if self.args.perturbThreshold > 0:
            xTilde = x.cpu() + np.random.uniform(-self.args.perturbThreshold, self.args.perturbThreshold, x.shape).astype('float32')
        else:
            xTilde = np.copy(x)

        x_tilde_k = torch.autograd.Variable(xTilde, requires_grad=True).to(device)
        for k in range(self.args.samplingStep):
            f_prime = torch.autograd.grad(self.log_xTilde_x(x, x_tilde_k, y, self.args.drvWeight, modelNo = modelNo), [x_tilde_k], retain_graph=True)[0]
            x_tilde_k.data += self.args.sgldLr * f_prime + np.sqrt(self.args.sgldLr*2) * self.args.sgldStd * torch.randn_like(x_tilde_k)

        if not self.args.bayesianTraining:
            self.classifier.model.train()

        return x_tilde_k.detach()

    def bayesianAdversarialTrain(self):

        size = len(self.classifier.trainloader.dataset)

        bestLoss = [np.infty for i in range(self.args.bayesianModelNum)]
        bestValLoss = [np.infty for i in range(self.args.bayesianModelNum)]
        bestClfLoss = [np.infty for i in range(self.args.bayesianModelNum)]
        bestValClfLoss = [np.infty for i in range(self.args.bayesianModelNum)]
        bestValClfAcc = [0 for i in range(self.args.bayesianModelNum)]
        logger = SummaryWriter()

        #burn-in for classifier
        print(f"burn-in training {self.args.burnIn} epochs")
        for i in range(self.args.burnIn):
            for m in range(self.args.bayesianModelNum):
                for batch, (X, y) in enumerate(self.classifier.trainloader):
                    pred = self.modelEval(X, m)
                    loss = self.classifier.classLoss(pred, y)
                    # Backpropagation
                    self.classifier.optimiserList[m].zero_grad()
                    loss.backward()
                    self.classifier.optimiserList[m].step()

        startTime = time.time()
        valTime = 0
        for ep in range(self.args.epochs):
            epLoss = np.zeros(self.args.bayesianModelNum)
            epClfLoss = np.zeros(self.args.bayesianModelNum)

            for i in range(self.args.bayesianModelNum):
                batchNum = 0
                for batch, (X, y) in enumerate(self.classifier.trainloader):
                    batchNum += 1
                    refBoneLengths = self.boneLengths(X)

                    # sample x and compute logP(X)
                    #ySamples = torch.randint(0, self.args.classNum, (self.args.batchSize,))

                    XSamples = self.sampleX(X, i)

                    logPX = self.modelEval(X, i).mean()

                    logPXSamples = self.modelEval(XSamples, i).mean()

                    lossLogPX = -(logPX - logPXSamples)

                    # compute logP(x_tilde|X, y)

                    XTilde = self.sampleXTilde(X, y, refBoneLengths, i)

                    lossPYXTilde = torch.nn.CrossEntropyLoss()(self.modelEval(XTilde, i), y)
                    lossXXTilde = self.xXTildeLoss(X, XTilde, refBoneLengths)

                    lossPXTildeXY = lossPYXTilde + self.args.drvWeight * lossXXTilde

                    lossPYX = torch.nn.CrossEntropyLoss()(self.modelEval(X, i), y)

                    loss = self.args.xWeight*lossLogPX + self.args.xTildeWeight*lossPXTildeXY + self.args.clfWeight * lossPYX

                    epLoss[i] += loss.detach().item()
                    epClfLoss[i] += lossPYX.detach().item()

                    # Backpropagation
                    self.classifier.optimiserList[i].zero_grad()
                    loss.backward()
                    self.classifier.optimiserList[i].step()

                    if (batchNum - 1) % 20 == 0:
                        loss, current = loss.detach().item(), batch * len(X)
                        print(f"epoch: {ep}/{self.args.epochs} model: {i} loss: {loss:>7f}  lossLogPX: {lossLogPX:>6f}, lossPXTildeXY: {lossPXTildeXY:>6f}, lossPYX: {lossPYX:>6f} [{current:>5d}/{size:>5d}]")
                        print(f"epoch: {ep}/{self.args.epochs} model: {i} (logPX: {logPX:>6f} logPXSamples: {logPXSamples:>6f} lossPYXTilde: {lossPYXTilde:>7f} lossXXTilde: {lossXXTilde:>6f}) [{current:>5d}/{size:>5d}]")

                # save a model if the best training loss so far has been achieved.
                epLoss[i] /= batchNum
                epClfLoss[i] /= batchNum
                logger.add_scalar(('Loss/train/model%d' % i), epLoss[i], ep)
                logger.add_scalar(('Loss/train/clf/model%d' % i), epClfLoss[i], ep)
                if epLoss[i] < bestLoss[i]:
                    print(f"epoch: {ep} model: {i} per epoch average training loss improves from: {bestLoss[i]} to {epLoss[i]}")
                    # modelFile = self.retFolder + '/' + str(i) + '_minLossAppendedModel_adtrained_' + str(epLoss[i]) + '.pth'
                    # torch.save(self.classifier.modelList[i].model.state_dict(), modelFile)
                    bestLoss[i] = epLoss[i]

                if epClfLoss[i] < bestClfLoss[i]:
                    print(f"epoch: {ep} model: {i} per epoch average training clf loss improves from: {bestClfLoss[i]} to {epClfLoss[i]}")
                    #
                    # modelFile = self.retFolder + '/' + str(i) + '_minLossAppendedModel_adtrained.pth'
                    # torch.save(self.classifier.modelList[i].model.state_dict(), modelFile)
                    bestClfLoss[i] = epClfLoss[i]

                valStartTime = time.time()
                # run validation and save a model if the best validation loss so far has been achieved.

                self.classifier.setEval(modelNo = i)

                misclassified = 0
                for v, (tx, ty) in enumerate(self.classifier.testloader):
                    pred = torch.argmax(self.modelEval(tx, i), dim=1)
                    diff = (pred - ty) != 0
                    misclassified += torch.sum(diff)

                valClfAcc = 1 - misclassified / len(self.classifier.testloader.dataset)

                if valClfAcc > bestValClfAcc[i]:
                    print(f"epoch: {ep} model: {i} per epoch average clf validation acc improves from: {bestValClfAcc[i]} to {valClfAcc}")
                    modelFile = self.retFolder + '/' + str(i) + '_minValLossAppendedModel_adtrained.pth'
                    torch.save(self.classifier.modelList[i].model.state_dict(), modelFile)
                    bestValClfAcc[i] = valClfAcc

                valEndTime = time.time()
                valTime += (valEndTime - valStartTime) / 3600
                self.classifier.setTrain(modelNo = i)
            print(f"epoch: {ep} time elapsed: {(time.time() - startTime) / 3600 - valTime} hours")

        return