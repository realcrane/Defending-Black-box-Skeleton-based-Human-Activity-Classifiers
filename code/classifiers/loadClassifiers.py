import pdb

from classifiers.ActionClassifier import ClassifierArgs
from classifiers.ThreeLayerMLP import ThreeLayerMLP
from classifiers.STGCN import STGCN
from classifiers.CTRGCN import CTRGCN
from classifiers.BayesianClassifier import ExtendedBayesianClassifier

def loadClassifier(args):
    classifier = ''
    if args.classifier == '3layerMLP':
        classifier = ThreeLayerMLP(args)
    elif args.classifier == 'STGCN':
        classifier = STGCN(args)
    elif args.classifier == 'CTRGCN':
        classifier = CTRGCN(args)
    elif args.classifier == 'ExtendedBayesian':
        classifier = ExtendedBayesianClassifier(args)
    else:
        print('No classifier created')

    return classifier