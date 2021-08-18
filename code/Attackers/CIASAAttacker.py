import pdb
import os
import torch
from Attackers.Attacker import ActionAttacker
from classifiers.loadClassifiers import loadClassifier
import torch as K
import numpy as np
from torch import nn
from Configuration import *

class CIASAAttacker(ActionAttacker):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.name = 'CIASA'
        #parameters for SMART attack
        self.perpLossType = args.perpLoss
        self.classWeight = args.classWeight
        self.reconWeight = args.reconWeight
        self.boneLenWeight = args.boneLenWeight
        self.attackType = args.attackType
        self.epochs = args.epochs
        self.updateRule = args.updateRule
        self.updateClip = args.clippingThreshold
        self.deltaT = 1 / 30
        self.topN = 3
        self.refBoneLengths = []
        self.optimizer = ''
        self.classifier = loadClassifier(args)
        self.retFolder = self.args.retPath + self.args.dataset + '/' + self.args.classifier + '/' + self.name + '/'
        if not os.path.exists(self.retFolder):
            os.mkdir(self.retFolder)

        self.discriminator = ''
        self.createDiscriminator()
        self.discriminator.to(device)

        if self.args.dataset == 'hdm05':
            self.frameFeature = torch.zeros([self.args.batchSize * self.classifier.trainloader.dataset.data.shape[1],
                                              len(self.classifier.trainloader.dataset.parents),
                                              len(self.classifier.trainloader.dataset.parents)])
        elif self.args.dataset == 'ntu60' or self.args.dataset == 'ntu120':
            self.frameFeature = torch.zeros([self.args.batchSize * self.classifier.trainloader.dataset.data.shape[2],
                                              len(self.classifier.trainloader.dataset.parents),
                                              len(self.classifier.trainloader.dataset.parents)])

        # the joint weights are decided per joint, the spinal joints have higher weights.
        self.jointWeights = torch.Tensor([[[0.02, 0.02, 0.02, 0.02, 0.02,
                                       0.02, 0.02, 0.02, 0.02, 0.02,
                                       0.04, 0.04, 0.04, 0.04, 0.04,
                                       0.02, 0.02, 0.02, 0.02, 0.02,
                                       0.02, 0.02, 0.02, 0.02, 0.02]]]).to(device)
    def createDiscriminator(self):
        class Discriminator(nn.Module):
            def __init__(self, datashape, dataset='hdm05'):
                super().__init__()

                self.dataShape = datashape
                #currently the bone number is 25 in all the dataset we use
                if dataset == 'hdm05':
                    self.convStack = nn.Sequential(
                        nn.Conv2d(1, 32, kernel_size = 3),
                        nn.Conv2d(32, 32, kernel_size = 3),
                        nn.Flatten(),
                        nn.Linear(21 * 21 *32, 1),
                        nn.ReLU()
                    )
                elif dataset == 'ntu60' or dataset == 'ntu120':
                    self.convStack = nn.Sequential(
                        nn.Conv2d(1, 32, kernel_size=3),
                        nn.Conv2d(32, 32, kernel_size=3),
                        nn.Flatten(),
                        nn.Linear(21 * 21 *32, 1),
                        nn.ReLU()
                    )
                else:
                    self.convStack = ''

            def forward(self, x):
                x = x[:, None, :, :]
                logits = self.convStack(x)
                return logits
            def loss(self, x, labels):
                return torch.square(self(x) - labels).sum()

        self.discriminator = Discriminator(self.classifier.trainloader.dataset.data.shape, self.args.dataset)
        self.discriminator.optimiser = torch.optim.Adam(self.discriminator.parameters(), lr=0.001, betas=(0.9, 0.999),
                                          eps=1e-08, weight_decay=0, amsgrad=False)

    def perFrameFeature(self, frame, boneLengths):

        featureMap = torch.zeros((len(frame), len(frame)))
        transposed = torch.transpose(frame, 0, 1)

        for i in range(len(frame)):
            featureMap[i:i+1] = torch.matmul(frame[i:i+1], transposed) / boneLengths / boneLengths[i]

        return featureMap

    def updateFeatureMap(self, data):

        buffer = self.frameFeature.detach()
        jpositions = K.reshape(data, (data.shape[0], data.shape[1], -1, 3))


        boneVecs = jpositions - jpositions[:, :, self.classifier.trainloader.dataset.parents, :]

        boneLengths = torch.sqrt(torch.sum(torch.square(boneVecs + 1e-8), axis=-1))

        boneVecs = K.reshape(boneVecs, (-1, boneVecs.shape[2], boneVecs.shape[3]))
        boneLengths = K.reshape(boneLengths, (-1, boneLengths.shape[2]))

        for i in range(self.frameFeature.shape[0]):
                buffer[i, :, :] = self.perFrameFeature(boneVecs[i], boneLengths[i])

    def boneLengths(self, data):

        jpositions = K.reshape(data, (data.shape[0], data.shape[1], -1, 3))

        boneVecs = jpositions - jpositions[:, :, self.classifier.trainloader.dataset.parents, :] + 1e-8

        boneLengths = torch.sqrt(torch.sum(torch.square(boneVecs), axis=-1))

        return boneLengths

    def boneLengthLoss (self, parentIds, adData, refBoneLengths):

        # convert the data into shape (batchid, frameNo, jointNo, jointCoordinates)
        jpositions = K.reshape(adData, (adData.shape[0], adData.shape[1], -1, 3))


        boneVecs = jpositions - jpositions[:, :, self.classifier.trainloader.dataset.parents, :] + 1e-8

        boneLengths = torch.sqrt(torch.sum(torch.square(boneVecs), axis=-1))

        boneLengthsLoss = K.mean(
            K.sum(K.sum(K.square(boneLengths - refBoneLengths), axis=-1), axis=-1))
        return boneLengthsLoss

    def smoothnessLoss (self, adData):
        adAcc = (adData[:, 2:, :] - 2 * adData[:, 1:-1, :] + adData[:, :-2, :]) / self.deltaT / self.deltaT


        return K.mean(K.sum(K.sum(K.square(adAcc), axis=-1), axis=-1), axis=-1)


    def perceptualLoss(self, refData, adData, refBoneLengths):


        elements = self.perpLossType.split('_')

        if elements[0] == 'l2' or elements[0] == 'l2Clip':

            diffmx = K.square(refData - adData),
            squaredLoss = K.sum(K.reshape(K.square(refData - adData), (refData.shape[0], refData.shape[1], 25, -1)),
                                axis=-1)

            weightedSquaredLoss = squaredLoss * self.jointWeights

            squareCost = K.sum(K.sum(weightedSquaredLoss, axis=-1), axis=-1)

            oloss = K.mean(squareCost, axis=-1)



        elif elements[0] == 'lInf':
            squaredLoss = K.sum(K.reshape(K.square(refData - adData), (refData.shape[0], refData.shape[1], 25, -1)),
                                axis=-1)

            weightedSquaredLoss = squaredLoss * self.jointWeights

            squareCost = K.sum(weightedSquaredLoss, axis=-1)

            oloss = K.mean(K.norm(squareCost, ord=np.inf, axis=0))

        else:
            print('warning: no reconstruction loss')
            return

        if len(elements) == 1:
            return oloss

        elif elements[1] == 'acc-bone':

            jointAcc = self.smoothnessLoss(adData)

            boneLengthsLoss = self.boneLengthLoss(self.classifier.trainloader.dataset.parents, adData, refBoneLengths)

            return boneLengthsLoss * (1 - self.reconWeight) * self.boneLenWeight + jointAcc * (1 - self.reconWeight) * (
                        1 - self.boneLenWeight) + oloss * self.reconWeight

    def unspecificAttack(self, labels):

        flabels = np.ones((len(labels), self.classifier.args.classNum))
        flabels = flabels * 1 / self.classifier.args.classNum

        return torch.LongTensor(flabels)

    def specifiedAttack(self, labels, targettedClasses=[]):
        if len(targettedClasses) <= 0:
            flabels = torch.LongTensor(np.random.randint(0, self.classifier.args.classNum, len(labels)))
        else:
            flabels = targettedClasses
        pdb.set_trace()
        return flabels

    def abAttack(self, labels):

        flabels = labels

        return flabels

    def foolRateCal(self, rlabels, flabels, logits = None):

        hitIndices = []

        if self.attackType == 'ab':
            for i in range(0, len(flabels)):
                if flabels[i] != rlabels[i]:
                    hitIndices.append(i)
        elif self.attackType == 'abn':
            for i in range(len(flabels)):
                sorted,indices = torch.sort(logits[i], descending=True)
                ret = (indices[:self.topN] == rlabels[i]).nonzero(as_tuple=True)
                if len(ret) == 0:
                    hitIndices.append(i)

        elif self.attackType == 'sa':
            for i in range(0, len(flabels)):
                if flabels[i] == rlabels[i]:
                    hitIndices.append(i)

        return len(hitIndices) / len(flabels) * 100

    def getUpdate(self, grads, input):

        if self.updateRule == 'gd':
            self.learningRate = 0.01

            return input - grads * self.learningRate

        elif self.updateRule == 'Adam':
            if not hasattr(self, 'Adam'):
                self.Adam = MyAdam()
            return self.Adam.get_updates(grads, input)

    def reshapeData(self, x, toNative=True):

        #ntu format is N, C, T, V, M (batch_no, channel, frame, node, person)
        if toNative:
            x = x.permute(0, 2, 3, 1, 4)
            x = x.reshape((x.shape[0], x.shape[1], -1, x.shape[4]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], -1, 3, x.shape[4]))
            x = x.permute(0, 3, 1, 2, 4)
        return x

    def attack(self):

        self.classifier.setEval()

        #set up the attack labels based on the attack type
        labels = self.classifier.trainloader.dataset.labels
        if self.attackType == 'abn':
            flabels = self.unspecificAttack(labels).to(device)
        elif self.attackType == 'sa':
            flabels = self.specifiedAttack(labels).to(device)
            oflabels = np.argmax(flabels, axis=-1)
        elif self.attackType == 'ab':
            flabels = self.abAttack(labels).to(device)
        else:
            print('specified targetted attack, no implemented')
            return

        overallFoolRate = 0
        batchTotalNum = 0

        if self.args.dataset == 'hdm05':
            zeros = torch.zeros(self.classifier.args.batchSize*self.classifier.trainloader.dataset.data.shape[1], dtype = torch.int64).to(device)
            ones = torch.ones(self.classifier.args.batchSize*self.classifier.trainloader.dataset.data.shape[1], dtype = torch.int64).to(device)
        elif self.args.dataset == 'ntu60' or self.args.dataset == 'ntu120':
            zeros = torch.zeros(self.classifier.args.batchSize * self.classifier.trainloader.dataset.data.shape[2],
                                dtype=torch.int64).to(device)
            ones = torch.ones(self.classifier.args.batchSize * self.classifier.trainloader.dataset.data.shape[2],
                              dtype=torch.int64).to(device)
        else:
            zeros = ''
            ones = ''
            print ('unknown dataset for creating adversarial labels')
            return
        for batchNo, (tx, ty) in enumerate(self.classifier.trainloader):
            adData = tx.clone()
            adData.requires_grad = True
            minCl = np.PINF
            maxFoolRate = np.NINF
            batchTotalNum += 1
            for ep in range(self.classifier.args.epochs):


                # compute the classification loss and gradient
                pred = self.classifier.modelEval(adData)
                predictedLabels = torch.argmax(pred, axis=1)

                # computer the classfication loss gradient

                if self.attackType == 'ab':
                    classLoss = -torch.nn.CrossEntropyLoss()(pred, flabels[batchNo*self.classifier.args.batchSize:(batchNo+1)*self.classifier.args.batchSize])
                elif self.attackType == 'abn':
                    classLoss = torch.mean((pred - flabels[batchNo*self.classifier.args.batchSize:(batchNo+1)*self.classifier.args.batchSize])**2)
                else:
                    classLoss = torch.nn.CrossEntropyLoss()(pred, flabels[batchNo*self.classifier.args.batchSize:(batchNo+1)*self.classifier.args.batchSize])

                adData.grad = None
                classLoss.backward(retain_graph=True)
                cgs = adData.grad

                #computer the perceptual loss and gradient

                # the standard format is [batch_no, frames, DoFs]. If the tx.shape > 3, then we need reformat it
                # if the data contains more than one person, then the loss is the summed losses of all people

                if len(tx.shape) > 3:
                    convertedData = self.reshapeData(tx)
                    convertedAdData = self.reshapeData(adData)
                    if len(convertedData.shape) > 3:
                        percepLoss = 0
                        GANLoss = 0
                        ##we have more than one person, assuming the last index indicates the person
                        for i in range(convertedData.shape[-1]):
                            boneLengths = self.boneLengths(convertedData[:, :, :, i])
                            percepLoss += self.perceptualLoss(convertedData[:, :, :, i], convertedAdData[:, :, :, i], boneLengths)
                            self.updateFeatureMap(convertedAdData[:, :, :, i])
                            GANLoss += self.discriminator.loss(self.frameFeature.to(device), zeros) \
                                      + self.discriminator.loss(self.frameFeature.to(device), ones)
                            self.updateFeatureMap(convertedData[:, :, :, i])
                            GANLoss += self.discriminator.loss(self.frameFeature.to(device), ones)
                    else:
                        boneLengths = self.boneLengths(convertedData)
                        percepLoss = self.perceptualLoss(convertedData, convertedAdData, boneLengths)
                        self.updateFeatureMap(convertedAdData)
                        GANLoss = self.discriminator.loss(self.frameFeature, zeros) \
                                  + self.discriminator.loss(self.frameFeature, ones)
                        self.updateFeatureMap(convertedData)
                        GANLoss += self.discriminator.loss(self.frameFeature, ones)
                else:
                    boneLengths = self.boneLengths(tx)
                    percepLoss = self.perceptualLoss(tx, adData, boneLengths)
                    self.updateFeatureMap(adData)
                    GANLoss = self.discriminator.loss(self.frameFeature, zeros) \
                              + self.discriminator.loss(self.frameFeature, ones)
                    self.updateFeatureMap(tx)
                    GANLoss += self.discriminator.loss(self.frameFeature, ones)


                percepLoss = percepLoss + GANLoss


                self.discriminator.optimiser.zero_grad()
                adData.grad = None
                percepLoss.backward(retain_graph=True)
                pgs = adData.grad

                self.discriminator.optimiser.step()

                if ep % 50 == 0:
                    print(f"Iteration {ep}/{self.classifier.args.epochs}, batchNo {batchNo}: Class Loss {classLoss:>9f}, Perceptual Loss: {percepLoss:>9f}")

                if self.attackType == 'ab':
                    foolRate = self.foolRateCal(ty, predictedLabels)
                elif self.attackType == 'abn':
                    foolRate = self.foolRateCal(ty, predictedLabels, pred)
                elif self.attackType == 'sa':
                    cFlabels = flabels[batchNo * self.classifier.args.batchSize:(batchNo + 1) * self.classifier.args.batchSize]
                    foolRate = self.foolRateCal(cFlabels, predictedLabels)
                else:
                    print('specified targetted attack, no implemented')
                    return

                if maxFoolRate < foolRate:
                    print('foolRate Improved! Iteration %d/%d, batchNo %d: Class Loss %.9f, Perceptual Loss: %.9f, Fool rate:%.2f' % (
                        ep, self.classifier.args.epochs, batchNo, classLoss, percepLoss, foolRate))
                    maxFoolRate = foolRate
                    folder = '/batch%d_%s_clw_%.2f_pl_%s_plw_%.2f/' % (
                        batchNo, self.attackType, self.classWeight, self.perpLossType, self.reconWeight)

                    path = self.retFolder + folder
                    if not os.path.exists(path):
                        os.mkdir(path)

                    if self.attackType == 'ab' or self.attackType == 'abn':
                        np.savez_compressed(
                            self.retFolder + folder + 'AdExamples_maxFoolRate_batch%d_AttackType_%s_clw_%.2f_pl_%s_reCon_%.2f_fr_%.2f.npz' % (
                                batchNo, self.attackType, self.classWeight, self.perpLossType, self.reconWeight, foolRate),
                            clips=adData.cpu().detach().numpy(), classes=predictedLabels.cpu().detach().numpy(),
                            oriClips=tx.cpu().detach().numpy(), tclasses=ty.cpu().detach().numpy(), classLos=classLoss.cpu().detach().numpy(),percepLoss=percepLoss.cpu().detach().numpy())
                    elif self.attackType == 'sa':
                        np.savez_compressed(
                            self.retFolder + folder + 'AdExamples_maxFoolRate_batch%d_AttackType_%s_clw_%.2f_pl_%s_reCon_%.2f_fr_%.2f.npz' % (
                                batchNo, self.attackType, self.classWeight, self.perpLossType, self.reconWeight, foolRate),
                            clips=adData.cpu().detach().numpy(), classes=predictedLabels.cpu().detach().numpy(), fclasses=oflabels,
                            oriClips=tx.cpu().detach().numpy(),
                            tclasses=ty.cpu().detach().numpy(), classLos=classLoss.cpu().detach().numpy(),percepLoss=percepLoss.cpu().detach().numpy())

                if maxFoolRate == 100:
                    break;

                cgsView = cgs.view(cgs.shape[0], -1)
                pgsView = pgs.view(pgs.shape[0], -1)

                cgsnorms = torch.norm(cgsView, dim=1) + 1e-18
                pgsnorms = torch.norm(pgsView, dim=1) + 1e-18

                cgsView /= cgsnorms[:, np.newaxis]
                pgsView /= pgsnorms[:, np.newaxis]

                temp = self.getUpdate(cgs * self.classWeight + pgs * (1 - self.classWeight), adData)

                missedIndices = []

                if self.attackType == 'ab':
                    for i in range(len(ty)):
                        if ty[i] == predictedLabels[i]:
                            missedIndices.append(i)
                elif self.attackType == 'abn':
                    for i in range(len(ty)):
                        sorted, indices = torch.sort(pred[i], descending=True)
                        ret = (indices[:self.topN] == ty[i]).nonzero(as_tuple=True)
                        if len(ret) > 0:
                            missedIndices.append(i)
                elif self.attackType == 'sa':
                    for i in range(len(ty)):
                        if cFlabels[i] != predictedLabels[i]:
                            missedIndices.append(i)

                tempCopy = adData.detach()


                if self.updateClip > 0:

                    updates = temp[missedIndices] - adData[missedIndices]
                    for ci in range(updates.shape[0]):

                        updateNorm = torch.norm(updates[ci])
                        if updateNorm > self.updateClip:
                            updates[ci] = updates[ci] * self.updateClip / updateNorm

                    tempCopy[missedIndices] += updates
                else:
                    tempCopy[missedIndices] = temp[missedIndices]
            overallFoolRate += maxFoolRate
            print(f"Current fool rate is {overallFoolRate/batchTotalNum}")

        print(f"Overall fool rate is {overallFoolRate/batchTotalNum}")
        return overallFoolRate/batchTotalNum
