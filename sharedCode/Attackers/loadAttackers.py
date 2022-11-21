from Attackers.SMARTAttacker import SmartAttacker
from Attackers.CIASAAttacker import CIASAAttacker
def loadAttacker(args):
    attacker = ''
    if args.attacker == 'SMART':
        attacker = SmartAttacker(args)
    elif args.attacker == 'CIASA':
        attacker = CIASAAttacker(args)
    else:
        print('No classifier created')

    return attacker