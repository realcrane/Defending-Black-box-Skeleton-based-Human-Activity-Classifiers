import pdb
from Configuration import *
from AdversarialTrainers.AdversarialTrainer import AdversarialTrainer
from classifiers.loadClassifiers import loadClassifier
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
from optimisers.optimisers import SGAdaHMC
import copy
class EBMATrainer(AdversarialTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.classifier = loadClassifier(args)
        self.retFolder = self.args.retPath + '/' + self.args.dataset + '/' + self.args.classifier + '/' + self.args.adTrainer + '/'
        if args.bayesianTraining:
            self.replayBufferList = [self.initRandom() for i in range(args.bayesianModelNum)]
        else:
            if len(args.bufferSamples) > 0:
                self.replayBuffer = torch.FloatTensor(np.load(args.bufferSamples)['clips'])
            else:
                self.replayBuffer = self.initRandom()


        if not os.path.exists(self.retFolder):
            os.makedirs(self.retFolder)

    def modelEval(self, X, modelNo = -1):
        if self.args.bayesianTraining:
            pred = self.classifier.modelEval(X, modelNo)
        else:
            pred = torch.nn.ReLU()(self.classifier.model(X))
        return pred

    def initRandom(self):
        return torch.FloatTensor(self.args.batchSize, self.classifier.trainloader.dataset.data.shape[1], self.classifier.trainloader.dataset.data.shape[2]).uniform_(-1, 1)

    def sampleP0(self, rb = -1, y=None):
        #currently, we do not do conditioned sampling, y is not used
        if rb == -1:
            replayBuffer = self.replayBuffer
        else:
            replayBuffer = self.replayBufferList[rb]
        if len(replayBuffer) == 0:
            return self.initRandom(), []
        self.args.bufferSize = len(replayBuffer) if y is None else len(replayBuffer)
        inds = torch.randint(0, self.args.bufferSize, (self.args.batchSize,))
        # if cond, convert inds to class conditional inds
        # if y is not None:
        #     inds = y.cpu() * self.args.bufferSize + inds
        bufferSamples = replayBuffer[inds]
        randomSamples = self.initRandom()
        choose_random = (torch.rand(self.args.batchSize) < self.args.reinitFreq).float()[:, None, None]
        samples = choose_random * randomSamples + (1 - choose_random) * bufferSamples
        return samples, inds

    def sampleX(self, rb = -1, y=None):
        if rb == -1:
            replayBuffer = self.replayBuffer
        else:
            replayBuffer = self.replayBufferList[rb]
        if not self.args.bayesianTraining:
            self.classifier.model.eval()
        # get batch size
        bs = self.args.batchSize if y is None else y.size(0)
        # generate initial samples and buffer inds of those samples (if buffer is used)
        initSample, bufferInds = self.sampleP0(rb)
        x_k = torch.autograd.Variable(initSample, requires_grad=True).to(device)
        # sgld
        for k in range(self.args.samplingStep):
            f_prime = torch.autograd.grad(torch.logsumexp(self.modelEval(x_k), 0).sum(), [x_k], retain_graph=True)[0]
            x_k.data += self.args.sgldLr * f_prime + np.sqrt(self.args.sgldLr*2) * self.args.sgldStd * torch.randn_like(x_k)
        if not self.args.bayesianTraining:
            self.classifier.model.train()

        final_samples = x_k.detach()
        # update replay buffer
        if len(replayBuffer) > 0:
            replayBuffer[bufferInds] = final_samples.cpu()
        return final_samples

    def boneLengths(self, x):
        jpositions = torch.reshape(x, (x.shape[0], x.shape[1], -1, 3))

        boneVecs = jpositions - jpositions[:, :, self.classifier.trainloader.dataset.parents, :]

        boneLengths = torch.sqrt(torch.sum(torch.square(boneVecs), axis=-1) + 1.e-10)

        return boneLengths

    def xXTildeLoss(self, x, xTilde, refBoneLengths, drv_order=3):
        loss_drv = 0
        x_k = x.clone()
        x_tilde_k = xTilde.clone()

        #bone length loss
        boneLengths = self.boneLengths(xTilde)

        boneLengthsLoss = torch.mean(
            torch.sum(torch.sum(torch.square(boneLengths - refBoneLengths), axis=-1), axis=-1))


        for k in range(drv_order):
            x_k = x_k[:, 1:, :] - x_k[:, :-1, :]
            x_tilde_k = x_tilde_k[:, 1:, :] - x_tilde_k[:, :-1, :]

            loss_drv += torch.norm(x_k - x_tilde_k)

        return loss_drv + boneLengthsLoss

    def log_xTilde_x(self, x, xTilde, y, beta, refBoneLengths, drv_order=3, modelNo = -1):
        loss = torch.nn.CrossEntropyLoss()(self.modelEval(xTilde, modelNo), y)
        loss_motion = self.xXTildeLoss(x, xTilde, refBoneLengths, drv_order)
        return loss + beta * loss_motion

    def sampleXTilde(self, x, y, refBoneLengths):
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
            f_prime = torch.autograd.grad(self.log_xTilde_x(x, x_tilde_k, y, self.args.drvWeight, refBoneLengths), [x_tilde_k], retain_graph=True)[0]
            x_tilde_k.data += self.args.sgldLr * f_prime + np.sqrt(self.args.sgldLr*2) * self.args.sgldStd * torch.randn_like(x_tilde_k)

        if not self.args.bayesianTraining:
            self.classifier.model.train()

        return x_tilde_k.detach()

    def adversarialTrain(self):

        size = len(self.classifier.trainloader.dataset)

        bestLoss = np.infty
        bestValLoss = np.infty
        bestClfLoss = np.infty
        bestValClfLoss = np.infty
        logger = SummaryWriter()

        #burn-in for classifier
        print(f"burn-in training {self.args.burnIn} epochs")
        for i in range(self.args.burnIn):
            for batch, (X, y) in enumerate(self.classifier.trainloader):
                pred = self.modelEval(X)
                loss = self.classifier.classLoss(pred, y)
                # Backpropagation
                self.classifier.optimiser.zero_grad()
                loss.backward()
                self.classifier.optimiser.step()

        for ep in range(self.args.epochs):
            epLoss = 0
            epClfLoss = 0
            batchNum = 0

            for batch, (X, y) in enumerate(self.classifier.trainloader):
                batchNum += 1
                refBoneLengths = self.boneLengths(X)

                # sample x and compute logP(X)
                # ySamples = torch.randint(0, self.args.classNum, (self.args.batchSize,))

                XSamples = self.sampleX()

                logPX = self.modelEval(X).mean()

                logPXSamples = self.modelEval(XSamples).mean()

                lossLogPX = -(logPX - logPXSamples)

                # compute logP(x_tilde|X, y)

                XTilde = self.sampleXTilde(X, y, refBoneLengths)

                lossPYXTilde = torch.nn.CrossEntropyLoss()(self.modelEval(XTilde), y)
                lossXXTilde = self.xXTildeLoss(X, XTilde, refBoneLengths)

                lossPXTildeXY = lossPYXTilde + self.args.drvWeight * lossXXTilde

                lossPYX = torch.nn.CrossEntropyLoss()(self.modelEval(X), y)

                loss = self.args.xWeight*lossLogPX + self.args.xTildeWeight*lossPXTildeXY + self.args.clfWeight * lossPYX

                epLoss += loss.detach().item()
                epClfLoss += lossPYX.detach().item()

                # Backpropagation
                self.classifier.optimiser.zero_grad()
                loss.backward()
                self.classifier.optimiser.step()

                loss, current = loss.detach().item(), batch * len(X)
                print(f"epoch: {ep}/{self.args.epochs}  loss: {loss:>7f}  lossLogPX: {lossLogPX:>6f}, lossPXTildeXY: {lossPXTildeXY:>6f}, lossPYX: {lossPYX:>6f}"
                      f"(logPX: {logPX:>6f} logPXSamples: {logPXSamples:>6f} lossPYXTilde: {lossPYXTilde:>7f} lossXXTilde: {lossXXTilde:>6f}) [{current:>5d}/{size:>5d}]")

            # save a model if the best training loss so far has been achieved.
            epLoss /= batchNum
            epClfLoss /= batchNum
            logger.add_scalar('Loss/train', epLoss, ep)
            logger.add_scalar('Loss/train/clf', epClfLoss, ep)
            if epLoss < bestLoss:
                print(f"epoch: {ep} per epoch average training loss improves from: {bestLoss} to {epLoss}")
                #modelFile = self.retFolder + '/minLossModel_adtrained_' + str(epLoss) + '.pth'
                # modelFile = self.retFolder + '/minLossModel_adtrained.pth'
                # torch.save(self.classifier.model.state_dict(), modelFile)
                bestLoss = epLoss

            if epClfLoss < bestClfLoss:
                print(f"epoch: {ep} per epoch average training clf loss improves from: {bestClfLoss} to {epClfLoss}")
                # modelFile = self.retFolder + '/minLossModel_adtrained.pth'
                # torch.save(self.classifier.model.state_dict(), modelFile)
                bestClfLoss = epClfLoss

            # run validation and save a model if the best validation loss so far has been achieved.
            valLoss = 0
            valClfLoss = 0
            vbatch = 0
            self.classifier.model.eval()
            for v, (tx, ty) in enumerate(self.classifier.testloader):
                refBoneLengths = self.boneLengths(tx)

                # sample x and compute logP(X)
                ySamples = torch.randint(0, self.args.classNum, (self.args.batchSize,))
                XSamples = self.sampleX(ySamples)

                logPX = self.modelEval(tx).mean()
                logPXSamples = self.modelEval(XSamples).mean()

                lossLogPX = -(logPX-logPXSamples)

                # compute logP(x_tilde|X, y)

                XTilde = self.sampleXTilde(tx, ty, refBoneLengths)

                lossPYXTilde = torch.nn.CrossEntropyLoss()(self.modelEval(XTilde), ty)
                lossXXTilde = self.xXTildeLoss(tx, XTilde, refBoneLengths)

                lossPXTildeXY = lossPYXTilde + self.args.drvWeight * lossXXTilde

                lossPXY = torch.nn.CrossEntropyLoss()(self.modelEval(tx), ty)

                loss = self.args.xWeight * lossLogPX + self.args.xTildeWeight * lossPXTildeXY + self.args.clfWeight * lossPXY

                valLoss += loss.detach().item()
                valClfLoss += lossPYX.detach().item()
                vbatch += 1

            valLoss /= vbatch
            valClfLoss /= vbatch
            logger.add_scalar('Loss/validation', valLoss, ep)
            logger.add_scalar('Loss/validation/clf', valClfLoss, ep)
            self.classifier.model.train()
            if valLoss < bestValLoss:
                print(f"epoch: {ep} per epoch average validation loss improves from: {bestValLoss} to {valLoss}")
                #modelFile = self.retFolder + '/minValLossModel_adtrained_' + str(valLoss) + '.pth'
                #torch.save(self.classifier.model.state_dict(), modelFile)
                bestValLoss = valLoss
            if valClfLoss < bestValClfLoss:
                print(f"epoch: {ep} per epoch average clf validation loss improves from: {bestValClfLoss} to {valClfLoss}")
                modelFile = self.retFolder + '/minValLossModel_adtrained.pth'
                torch.save(self.classifier.model.state_dict(), modelFile)
                bestValClfLoss = valClfLoss

        return
    def bayesianAdversarialTrain(self):

        size = len(self.classifier.trainloader.dataset)

        bestLoss = [np.infty for i in range(self.args.bayesianModelNum)]
        bestValLoss = [np.infty for i in range(self.args.bayesianModelNum)]
        bestClfLoss = [np.infty for i in range(self.args.bayesianModelNum)]
        bestValClfLoss = [np.infty for i in range(self.args.bayesianModelNum)]
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

                    XSamples = self.sampleX(i)

                    logPX = self.modelEval(X, i).mean()

                    logPXSamples = self.modelEval(XSamples, i).mean()

                    lossLogPX = -(logPX - logPXSamples)

                    # compute logP(x_tilde|X, y)

                    XTilde = self.sampleXTilde(X, y, refBoneLengths)

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

                    loss, current = loss.detach().item(), batch * len(X)
                    print(f"epoch: {ep}/{self.args.epochs} model: {i} loss: {loss:>7f}  lossLogPX: {lossLogPX:>6f}, lossPXTildeXY: {lossPXTildeXY:>6f}, lossPYX: {lossPYX:>6f}"
                          f"(logPX: {logPX:>6f} logPXSamples: {logPXSamples:>6f} lossPYXTilde: {lossPYXTilde:>7f} lossXXTilde: {lossXXTilde:>6f}) [{current:>5d}/{size:>5d}]")

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

            # run validation and save a model if the best validation loss so far has been achieved.
            valLoss = np.zeros(self.args.bayesianModelNum)
            valClfLoss = np.zeros(self.args.bayesianModelNum)
            self.classifier.setEval()
            for i in range(self.args.bayesianModelNum):
                vbatch = 0
                for v, (tx, ty) in enumerate(self.classifier.testloader):
                    refBoneLengths = self.boneLengths(tx)

                    # sample x and compute logP(X)
                    #ySamples = torch.randint(0, self.args.classNum, (self.args.batchSize,))
                    XSamples = self.sampleX(i)

                    logPX = self.modelEval(tx, i).mean()
                    logPXSamples = self.modelEval(XSamples, i).mean()

                    lossLogPX = -(logPX-logPXSamples)

                    # compute logP(x_tilde|X, y)

                    XTilde = self.sampleXTilde(tx, ty, refBoneLengths)

                    lossPYXTilde = torch.nn.CrossEntropyLoss()(self.modelEval(XTilde, i), ty)
                    lossXXTilde = self.xXTildeLoss(tx, XTilde, refBoneLengths)

                    lossPXTildeXY = lossPYXTilde + self.args.drvWeight * lossXXTilde

                    lossPXY = torch.nn.CrossEntropyLoss()(self.modelEval(tx, i), ty)

                    loss = self.args.xWeight * lossLogPX + self.args.xTildeWeight * lossPXTildeXY + self.args.clfWeight * lossPXY

                    valLoss[i] += loss.detach().item()
                    valClfLoss[i] += lossPYX.detach().item()
                    vbatch += 1

                valLoss[i] /= vbatch
                valClfLoss[i] /= vbatch
                logger.add_scalar(('Loss/validation/model%d' % i), valLoss[i], ep)
                logger.add_scalar(('Loss/validation/clf/model%d' % i), valClfLoss[i], ep)
                if valLoss[i] < bestValLoss[i]:
                    print(f"epoch: {ep} model: {i} per epoch average validation loss improves from: {bestValLoss[i]} to {valLoss[i]}")
                    #modelFile = self.retFolder + '/' + str(i) + '_minValLossAppendedModel_adtrained_' + str(valLoss[i]) + '.pth'
                    #torch.save(self.classifier.modelList[i].model.state_dict(), modelFile)

                    bestValLoss[i] = valLoss[i]
                if valClfLoss[i] < bestValClfLoss[i]:
                    print(f"epoch: {ep} model: {i} per epoch average clf validation loss improves from: {bestValClfLoss[i]} to {valClfLoss[i]}")
                    modelFile = self.retFolder + '/' + str(i) + '_minValLossAppendedModel_adtrained.pth'
                    torch.save(self.classifier.modelList[i].model.state_dict(), modelFile)

                    bestValClfLoss[i] = valClfLoss[i]
            self.classifier.setTrain()
        return


