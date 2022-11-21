from classifiers.ThreeLayerMLP import ThreeLayerMLP

from classifiers.BayesianClassifier import ExtendedBayesianClassifier

def loadClassifier(args):
    classifier = ''
    if args.classifier == '3layerMLP':
        classifier = ThreeLayerMLP(args)
    elif args.classifier == 'ExtendedBayesian':
        classifier = ExtendedBayesianClassifier(args)
    else:
        print('No classifier created')

    return classifier