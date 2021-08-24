import pdb
from classifiers.ActionClassifier import ActionClassifier
from classifiers.sgn.model import *
import numpy as np
import os
import torch
from torch import nn
import torch.nn.functional as F
from shared.helpers import *
from torch.utils.tensorboard import SummaryWriter
import time
from datasets.dataloaders import *
from Configuration import *
class SGN(ActionClassifier):
    def __init__(self, args):
        super().__init__(args)
        self.trainloader, self.testloader = createDataLoader_sgn(args)
        self.createModel()
        self.steps = [60, 90, 110]
    def createModel(self):
        #this is a wrapper of the original STGCN code, with minor modification on the input format
        class Classifier(nn.Module):
            def __init__(self, args, dataloader,seg=20,bias=True):
                super().__init__()
                self.dataShape = dataloader.dataset.data.shape
                num_classes = args.classNum

                self.dim1 = 256
                self.dataset = args.dataset
                self.seg = seg
                num_joint = 25
                bs = args.batchSize
                if args.routine == 'test' or args.routine == 'gatherCorrectPrediction' or 'bayesianTest':
                    self.spa = self.one_hot(bs , num_joint, self.seg)
                    self.spa = self.spa.permute(0, 3, 2, 1).cuda()
                    self.tem = self.one_hot(bs , self.seg, num_joint)
                    self.tem = self.tem.permute(0, 3, 1, 2).cuda()
                else:
                    self.spa = self.one_hot(bs, num_joint, self.seg)
                    self.spa = self.spa.permute(0, 3, 2, 1).cuda()
                    self.tem = self.one_hot(bs, self.seg, num_joint)
                    self.tem = self.tem.permute(0, 3, 1, 2).cuda()

                self.tem_embed = embed(self.seg, 64*4, norm=False, bias=bias)
                self.spa_embed = embed(num_joint, 64, norm=False, bias=bias)
                self.joint_embed = embed(3, 64, norm=True, bias=bias)
                self.dif_embed = embed(3, 64, norm=True, bias=bias)
                self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
                self.cnn = local(self.dim1, self.dim1 * 2, bias=bias)
                self.compute_g1 = compute_g_spa(self.dim1 // 2, self.dim1, bias=bias)
                self.gcn1 = gcn_spa(self.dim1 // 2, self.dim1 // 2, bias=bias)
                self.gcn2 = gcn_spa(self.dim1 // 2, self.dim1, bias=bias)
                self.gcn3 = gcn_spa(self.dim1, self.dim1, bias=bias)
                self.fc = nn.Linear(self.dim1 * 2, num_classes)

                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                        m.weight.data.normal_(0, math.sqrt(2. / n))

                nn.init.constant_(self.gcn1.w.cnn.weight, 0)
                nn.init.constant_(self.gcn2.w.cnn.weight, 0)
                nn.init.constant_(self.gcn3.w.cnn.weight, 0)


            def forward(self, input):

                # Dynamic Representation
                bs, step, dim = input.size()
                num_joints = dim //3
                input = input.view((bs, step, num_joints, 3))
                input = input.permute(0, 3, 2, 1).contiguous()
                dif = input[:, :, :, 1:] - input[:, :, :, 0:-1]
                dif = torch.cat([dif.new(bs, dif.size(1), num_joints, 1).zero_(), dif], dim=-1)
                pos = self.joint_embed(input)
                tem1 = self.tem_embed(self.tem)
                spa1 = self.spa_embed(self.spa)
                dif = self.dif_embed(dif)
                dy = pos + dif
                # Joint-level Module
                input= torch.cat([dy, spa1], 1)
                g = self.compute_g1(input)
                input = self.gcn1(input, g)
                input = self.gcn2(input, g)
                input = self.gcn3(input, g)
                # Frame-level Module
                input = input + tem1
                input = self.cnn(input)
                # Classification
                output = self.maxpool(input)
                output = torch.flatten(output, 1)
                output = self.fc(output)

                return output

            def one_hot(self, bs, spa, tem):

                y = torch.arange(spa).unsqueeze(-1)
                y_onehot = torch.FloatTensor(spa, spa)

                y_onehot.zero_()
                y_onehot.scatter_(1, y, 1)

                y_onehot = y_onehot.unsqueeze(0).unsqueeze(0)
                y_onehot = y_onehot.repeat(bs, tem, 1, 1)

                return y_onehot

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
                    # self.model.load_state_dict(
                    #     torch.load(self.retFolder + self.args.adTrainer + '/' + self.args.trainedModelFile))
                    self.model.load_state_dict(
                        torch.load(self.retFolder + self.args.adTrainer + '/' + self.args.trainedModelFile,map_location='cuda:0'))
        self.configureOptimiser()
        self.classificationLoss()
        self.model.to(device)
    def configureOptimiser(self):

        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=self.args.learningRate, weight_decay=0.0001)

    def adjustLearningRate(self, epoch):
        if self.steps:
            lr = self.args.learningRate * (
                0.1**np.sum(epoch >= np.array(self.steps)))
            for param_group in self.optimiser.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.args.learningRate
    def classificationLoss(self):

        self.classLoss = torch.nn.CrossEntropyLoss()

    def setTrain(self):
        self.model.train()
    def setEval(self):
        self.model.eval()

    def modelEval(self, X, modelNo = -1):
        return self.model(X)


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
        label_output = list()
        pred_output = list()
        misclassified = 0
        results = np.empty(len(self.testloader.dataset.rlabels))
        with torch.no_grad():
            for v, (tx, ty) in enumerate(self.testloader):
                output = self.model(tx)
                output = output.view((-1, tx.size(0)//ty.size(0), output.size(1)))
                output = output.mean(1)
                pred = torch.argmax(output, dim=1)
                results[v * self.args.batchSize:(v + 1) * self.args.batchSize] = pred.cpu()
                diff = (pred - ty) != 0
                misclassified += torch.sum(diff)

        error = misclassified / len(self.testloader.dataset)
        print(f"accuracy: {1 - error:>4f}")
        np.savetxt(self.retFolder + 'testRets.txt', results)
        np.savetxt(self.retFolder + 'testGroundTruth.txt', self.testloader.dataset.rlabels)

        return 1 - error
    def collectCorrectPredictions(self):
        self.model.eval()

        # collect data from the test data
        misclassified = 0
        results = np.empty(len(self.testloader.dataset.rlabels))
        for v, (tx, ty) in enumerate(self.testloader):
            output = self.model(tx)
            output = output.view((-1, tx.size(0) // ty.size(0), output.size(1)))
            output = output.mean(1)
            pred = torch.argmax(output, dim=1)
            diff = (pred - ty) == 0
            results[v * self.args.batchSize:(v + 1) * self.args.batchSize] = diff.cpu()

        adData = self.testloader.dataset.data[results.astype(bool)]
        adLabels = self.testloader.dataset.rlabels[results.astype(bool)]

        print(f"{len(adLabels)} out of {len(results)} motions are collected")

        if not os.path.exists(self.retFolder):
            os.mkdir(self.retFolder)
        np.savez_compressed(self.retFolder+self.args.adTrainFile, clips=adData, classes=adLabels)

        return len(adLabels)