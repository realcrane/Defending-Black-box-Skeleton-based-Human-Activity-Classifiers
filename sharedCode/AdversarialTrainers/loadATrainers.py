from AdversarialTrainers.EBMATrainer import EBMATrainer

def loadATrainer(args):
    adTrainer = ''
    if args.adTrainer == 'EBMATrainer':
        adTrainer = EBMATrainer(args)
    else:
        print('No adTrainer created')

    return adTrainer