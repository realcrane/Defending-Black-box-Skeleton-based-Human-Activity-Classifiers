import pdb

from classifiers.ActionClassifier import ClassifierArgs
from classifiers.ThreeLayerMLP import ThreeLayerMLP
from classifiers.STGCN import STGCN
from classifiers.BayesianClassifier import ExtendedBayesianClassifier

def loadClassifier(args):
    cArgs = ClassifierArgs(args)
    classifier = ''
    if cArgs.classifier == '3layerMLP':
        classifier = ThreeLayerMLP(cArgs)
    elif cArgs.classifier == 'STGCN':
        classifier = STGCN(cArgs)
    elif cArgs.classifier == 'ExtendedBayesian':
        classifier = ExtendedBayesianClassifier(cArgs)
    else:
        print('No classifier created')

    return classifier