

class ClassifierArgs:
    def __init__(self, args):
        self.args = args
        self.dataFolder = args.dataPath
        self.retFolder = args.retPath
        self.trainFile = args.trainFile
        self.testFile = args.testFile
        self.trainedModelFile = args.trainedModelFile
        self.classNum = args.classNum
        self.batchSize = args.batchSize
        self.epochs = args.epochs
        self.dataset = args.dataset
        self.routine = args.routine
        self.classifier = args.classifier
        self.adTrainFile = args.adTrainFile


class ActionClassifier:
    def __init__(self, args):
        self.args = args
        self.loss = ''
        self.model = '';


    def train(self):
        return

    def test(self):
        self.model.eval()
        misclassified = 0
        results = np.empty(len(self.testloader.dataset.rlabels))
        for v, (tx, ty) in enumerate(self.testloader):
            pred = torch.argmax(self.model(tx), dim=1)
            results[v*self.args.batchSize:(v+1)*self.args.batchSize] = pred.cpu()
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



