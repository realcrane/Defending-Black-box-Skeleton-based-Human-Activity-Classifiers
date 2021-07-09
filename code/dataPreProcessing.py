import os
import numpy as np
import scipy.ndimage.filters as filters
import re

from motion import BVH
from motion import Animation
from motion.Quaternions import Quaternions
from motion.Pivots import Pivots
import ntpath

import pdb
import scipy.io

""" hdm05 """
oclass_map = {
    'cartwheelLHandStart1Reps': 'cartwheel',
    'cartwheelLHandStart2Reps': 'cartwheel',
    'cartwheelRHandStart1Reps': 'cartwheel',
    'clap1Reps': 'clap',
    'clap5Reps': 'clap',
    'clapAboveHead1Reps': 'clap',
    'clapAboveHead5Reps': 'clap',
    # 'depositFloorR': 'deposit',
    # 'depositHighR': 'deposit',
    # 'depositLowR': 'deposit',
    # 'depositMiddleR': 'deposit',
    'depositFloorR': 'grab',
    'depositHighR': 'grab',
    'depositLowR': 'grab',
    'depositMiddleR': 'grab',
    'elbowToKnee1RepsLelbowStart': 'elbow_to_knee',
    'elbowToKnee1RepsRelbowStart': 'elbow_to_knee',
    'elbowToKnee3RepsLelbowStart': 'elbow_to_knee',
    'elbowToKnee3RepsRelbowStart': 'elbow_to_knee',
    'grabFloorR': 'grab',
    'grabHighR': 'grab',
    'grabLowR': 'grab',
    'grabMiddleR': 'grab',
    #'hitRHandHead': 'hit',
    #'hitRHandHead': 'grab',
    'hopBothLegs1hops': 'hop',
    'hopBothLegs2hops': 'hop',
    'hopBothLegs3hops': 'hop',
    'hopLLeg1hops': 'hop',
    'hopLLeg2hops': 'hop',
    'hopLLeg3hops': 'hop',
    'hopRLeg1hops': 'hop',
    'hopRLeg2hops': 'hop',
    'hopRLeg3hops': 'hop',
    'jogLeftCircle4StepsRstart': 'jog',
    'jogLeftCircle6StepsRstart': 'jog',
    'jogOnPlaceStartAir2StepsLStart': 'jog',
    'jogOnPlaceStartAir2StepsRStart': 'jog',
    'jogOnPlaceStartAir4StepsLStart': 'jog',
    'jogOnPlaceStartFloor2StepsRStart': 'jog',
    'jogOnPlaceStartFloor4StepsRStart': 'jog',
    'jogRightCircle4StepsLstart': 'jog',
    'jogRightCircle4StepsRstart': 'jog',
    'jogRightCircle6StepsLstart': 'jog',
    'jogRightCircle6StepsRstart': 'jog',
    'jumpDown': 'jump',
    'jumpingJack1Reps': 'jump',
    'jumpingJack3Reps': 'jump',
    'kickLFront1Reps': 'kick',
    'kickLFront2Reps': 'kick',
    'kickLSide1Reps': 'kick',
    'kickLSide2Reps': 'kick',
    'kickRFront1Reps': 'kick',
    'kickRFront2Reps': 'kick',
    'kickRSide1Reps': 'kick',
    'kickRSide2Reps': 'kick',
    'lieDownFloor': 'lie_down',
    'punchLFront1Reps': 'punch',
    'punchLFront2Reps': 'punch',
    'punchLSide1Reps': 'punch',
    'punchLSide2Reps': 'punch',
    'punchRFront1Reps': 'punch',
    'punchRFront2Reps': 'punch',
    'punchRSide1Reps': 'punch',
    'punchRSide2Reps': 'punch',
    'rotateArmsBothBackward1Reps': 'rotate_arms',
    'rotateArmsBothBackward3Reps': 'rotate_arms',
    'rotateArmsBothForward1Reps': 'rotate_arms',
    'rotateArmsBothForward3Reps': 'rotate_arms',
    'rotateArmsLBackward1Reps': 'rotate_arms',
    'rotateArmsLBackward3Reps': 'rotate_arms',
    'rotateArmsLForward1Reps': 'rotate_arms',
    'rotateArmsLForward3Reps': 'rotate_arms',
    'rotateArmsRBackward1Reps': 'rotate_arms',
    'rotateArmsRBackward3Reps': 'rotate_arms',
    'rotateArmsRForward1Reps': 'rotate_arms',
    'rotateArmsRForward3Reps': 'rotate_arms',
    # 'runOnPlaceStartAir2StepsLStart': 'run',
    # 'runOnPlaceStartAir2StepsRStart': 'run',
    # 'runOnPlaceStartAir4StepsLStart': 'run',
    # 'runOnPlaceStartFloor2StepsRStart': 'run',
    # 'runOnPlaceStartFloor4StepsRStart': 'run',
    'runOnPlaceStartAir2StepsLStart': 'jog',
    'runOnPlaceStartAir2StepsRStart': 'jog',
    'runOnPlaceStartAir4StepsLStart': 'jog',
    'runOnPlaceStartFloor2StepsRStart': 'jog',
    'runOnPlaceStartFloor4StepsRStart': 'jog',
    'shuffle2StepsLStart': 'shuffle',
    'shuffle2StepsRStart': 'shuffle',
    'shuffle4StepsLStart': 'shuffle',
    'shuffle4StepsRStart': 'shuffle',
    'sitDownChair': 'sit_down',
    'sitDownFloor': 'sit_down',
    'sitDownKneelTieShoes': 'sit_down',
    'sitDownTable': 'sit_down',
    'skier1RepsLstart': 'ski',
    'skier3RepsLstart': 'ski',
    'sneak2StepsLStart': 'sneak',
    'sneak2StepsRStart': 'sneak',
    'sneak4StepsLStart': 'sneak',
    'sneak4StepsRStart': 'sneak',
    'squat1Reps': 'squat',
    'squat3Reps': 'squat',
    'staircaseDown3Rstart': 'climb',
    'staircaseUp3Rstart': 'climb',
    'standUpKneelToStand': 'stand_up',
    'standUpLieFloor': 'stand_up',
    'standUpSitChair': 'stand_up',
    'standUpSitFloor': 'stand_up',
    'standUpSitTable': 'stand_up',
    'throwBasketball': 'throw',
    'throwFarR': 'throw',
    'throwSittingHighR': 'throw',
    'throwSittingLowR': 'throw',
    'throwStandingHighR': 'throw',
    'throwStandingLowR': 'throw',
    'turnLeft': 'turn',
    'turnRight': 'turn',
    'walk2StepsLstart': 'walk_forward',
    'walk2StepsRstart': 'walk_forward',
    'walk4StepsLstart': 'walk_forward',
    'walk4StepsRstart': 'walk_forward',
    'walkBackwards2StepsRstart': 'walk_backward',
    'walkBackwards4StepsRstart': 'walk_backward',
    'walkLeft2Steps': 'walk_left',
    'walkLeft3Steps': 'walk_left',
    'walkLeftCircle4StepsLstart': 'walk_left',
    'walkLeftCircle4StepsRstart': 'walk_left',
    'walkLeftCircle6StepsLstart': 'walk_left',
    'walkLeftCircle6StepsRstart': 'walk_left',
    'walkOnPlace2StepsLStart': 'walk_inplace',
    'walkOnPlace2StepsRStart': 'walk_inplace',
    'walkOnPlace4StepsLStart': 'walk_inplace',
    'walkOnPlace4StepsRStart': 'walk_inplace',
    'walkRightCircle4StepsLstart': 'walk_right',
    'walkRightCircle4StepsRstart': 'walk_right',
    'walkRightCircle6StepsLstart': 'walk_right',
    'walkRightCircle6StepsRstart': 'walk_right',
    'walkRightCrossFront2Steps': 'walk_right',
    'walkRightCrossFront3Steps': 'walk_right',
}

class_map = {
    #1
    'cartwheelLHandStart1Reps': 'cartwheel',
    'cartwheelLHandStart2Reps': 'cartwheel',
    'cartwheelRHandStart1Reps': 'cartwheel',
    #2
    'clap1Reps': 'clap',
    'clap5Reps': 'clap',
    #3
    'clapAboveHead1Reps': 'clapAH',
    'clapAboveHead5Reps': 'clapAH',
    #4
    'depositFloorR': 'depositFR',
    #5
    'depositHighR': 'depositHR',
    #6
    'depositLowR': 'depositLR',
    #7
    'depositMiddleR': 'depositMR',
    #8
    'elbowToKnee1RepsLelbowStart': 'elbow_to_knee',
    'elbowToKnee1RepsRelbowStart': 'elbow_to_knee',
    'elbowToKnee3RepsLelbowStart': 'elbow_to_knee',
    'elbowToKnee3RepsRelbowStart': 'elbow_to_knee',
    #9
    'grabFloorR': 'grabFR',
    #10
    'grabHighR': 'grabHR',
    #11
    'grabLowR': 'grabLR',
    #12
    'grabMiddleR': 'grabMR',
    #13
    'hitRHandHead': 'hit',
    #14
    'hopBothLegs1hops': 'hop',
    'hopBothLegs2hops': 'hop',
    'hopBothLegs3hops': 'hop',
    #15
    'hopLLeg1hops': 'lhop',
    'hopLLeg2hops': 'lhop',
    'hopLLeg3hops': 'lhop',
    #16
    'hopRLeg1hops': 'rhop',
    'hopRLeg2hops': 'rhop',
    'hopRLeg3hops': 'rhop',
    #17
    'jogLeftCircle4StepsRstart': 'jogLC',
    'jogLeftCircle6StepsRstart': 'jogLC',
    #18
    'jogOnPlaceStartAir2StepsLStart': 'jogOP',
    'jogOnPlaceStartAir2StepsRStart': 'jogOP',
    'jogOnPlaceStartAir4StepsLStart': 'jogOP',
    'jogOnPlaceStartFloor2StepsRStart': 'jogOP',
    'jogOnPlaceStartFloor4StepsRStart': 'jogOP',
    #19
    'jogRightCircle4StepsLstart': 'jogRC',
    'jogRightCircle4StepsRstart': 'jogRC',
    'jogRightCircle6StepsLstart': 'jogRC',
    'jogRightCircle6StepsRstart': 'jogRC',
    #20
    'jumpDown': 'jumpD',
    #21
    'jumpingJack1Reps': 'jumpJ',
    'jumpingJack3Reps': 'jumpJ',
    #22
    'kickLFront1Reps': 'kickLF',
    'kickLFront2Reps': 'kickLF',
    #23
    'kickLSide1Reps': 'kickLS',
    'kickLSide2Reps': 'kickLS',
    #24
    'kickRFront1Reps': 'kickRF',
    'kickRFront2Reps': 'kickRF',
    #25
    'kickRSide1Reps': 'kickRS',
    'kickRSide2Reps': 'kickRS',
    #26
    'lieDownFloor': 'lie_down',
    #27
    'punchLFront1Reps': 'punchLF',
    'punchLFront2Reps': 'punchLF',
    #28
    'punchLSide1Reps': 'punchLS',
    'punchLSide2Reps': 'punchLS',
    #29
    'punchRFront1Reps': 'punchRF',
    'punchRFront2Reps': 'punchRF',
    #30
    'punchRSide1Reps': 'punchRS',
    'punchRSide2Reps': 'punchRS',
    #31
    'rotateArmsBothBackward1Reps': 'rotate_armsBB',
    'rotateArmsBothBackward3Reps': 'rotate_armsBB',
    #32
    'rotateArmsBothForward1Reps': 'rotate_armsBF',
    'rotateArmsBothForward3Reps': 'rotate_armsBF',
    #33
    'rotateArmsLBackward1Reps': 'rotate_armsLB',
    'rotateArmsLBackward3Reps': 'rotate_armsLB',
    #34
    'rotateArmsLForward1Reps': 'rotate_armsLF',
    'rotateArmsLForward3Reps': 'rotate_armsLF',
    #35
    'rotateArmsRBackward1Reps': 'rotate_armsRB',
    'rotateArmsRBackward3Reps': 'rotate_armsRB',
    #36
    'rotateArmsRForward1Reps': 'rotate_armsRF',
    'rotateArmsRForward3Reps': 'rotate_armsRF',
    #37
     'runOnPlaceStartAir2StepsLStart': 'run',
     'runOnPlaceStartAir2StepsRStart': 'run',
     'runOnPlaceStartAir4StepsLStart': 'run',
     'runOnPlaceStartFloor2StepsRStart': 'run',
     'runOnPlaceStartFloor4StepsRStart': 'run',
     #38
    'shuffle2StepsLStart': 'shuffle',
    'shuffle2StepsRStart': 'shuffle',
    'shuffle4StepsLStart': 'shuffle',
    'shuffle4StepsRStart': 'shuffle',
    #39
    'sitDownChair': 'sit_downC',
    #40
    'sitDownFloor': 'sit_downF',
    #41
    'sitDownKneelTieShoes': 'sit_downK',
    #42
    'sitDownTable': 'sit_downT',
    #43
    'skier1RepsLstart': 'ski',
    'skier3RepsLstart': 'ski',
    #44
    'sneak2StepsLStart': 'sneak',
    'sneak2StepsRStart': 'sneak',
    'sneak4StepsLStart': 'sneak',
    'sneak4StepsRStart': 'sneak',
    #45
    'squat1Reps': 'squat',
    'squat3Reps': 'squat',
    #46
    'staircaseDown3Rstart': 'climbD',
    #47
    'staircaseUp3Rstart': 'climbU',
    #48
    'standUpKneelToStand': 'stand_upK2S',
    #49
    'standUpLieFloor': 'stand_upLF',
    #50
    'standUpSitChair': 'stand_upSC',
    #51
    'standUpSitFloor': 'stand_upSF',
    #52
    'standUpSitTable': 'stand_upST',
    #53
    'throwBasketball': 'throwB',
    #54
    'throwFarR': 'throwFR',
    #55
    'throwSittingHighR': 'throwST',
    'throwSittingLowR': 'throwST',
    #56
    'throwStandingHighR': 'throwSD',
    'throwStandingLowR': 'throwSD',
    #57
    'turnLeft': 'turnL',
    #58
    'turnRight': 'turnR',
    #59
    'walk2StepsLstart': 'walk_forward',
    'walk2StepsRstart': 'walk_forward',
    'walk4StepsLstart': 'walk_forward',
    'walk4StepsRstart': 'walk_forward',
    #60
    'walkBackwards2StepsRstart': 'walk_backward',
    'walkBackwards4StepsRstart': 'walk_backward',
    #61
    'walkLeft2Steps': 'walk_left',
    'walkLeft3Steps': 'walk_left',
    #62
    'walkLeftCircle4StepsLstart': 'walk_leftLC',
    'walkLeftCircle4StepsRstart': 'walk_leftLC',
    'walkLeftCircle6StepsLstart': 'walk_leftLC',
    'walkLeftCircle6StepsRstart': 'walk_leftLC',
    #63
    'walkOnPlace2StepsLStart': 'walk_inplace',
    'walkOnPlace2StepsRStart': 'walk_inplace',
    'walkOnPlace4StepsLStart': 'walk_inplace',
    'walkOnPlace4StepsRStart': 'walk_inplace',
    #64
    'walkRightCircle4StepsLstart': 'walk_right',
    'walkRightCircle4StepsRstart': 'walk_right',
    'walkRightCircle6StepsLstart': 'walk_right',
    'walkRightCircle6StepsRstart': 'walk_right',
    #65
    'walkRightCrossFront2Steps': 'walk_rightRC',
    'walkRightCrossFront3Steps': 'walk_rightRC',
    }


class_names = list(sorted(list(set(class_map.values()))))

f = open('classes.txt', 'w')
f.write('\n'.join(class_names))
f.close()

def splitData(data, trainRatio, randomise = True):
    I = np.arange(len(data))
    if len(I) <= 0:
        print('emtpy indices')
        return 0, 0
    
    num = int(np.floor(len(I) * trainRatio))
    if randomise:
        I = np.random.permutation(I)
    return data[I[0:num]], data[I[num:]]

def get_files(directory, extension='bvh'):
    return [os.path.join(directory,f) for f in sorted(list(os.listdir(directory)))
    if os.path.isfile(os.path.join(directory,f))
    and f.endswith('.'+extension) and f != 'rest.bvh'] 


def hdm05Data():
    datafolder = 'D:/University/projects/LSTMMotion/code/datasets/CNNMotion/motionsynth_data/data/processed/'
    tdataFolder = '/../data/'

    hdm05_files = get_files(datafolder + 'hdm05')
    hdm05_clips = []
    hdm05_classes = []
    for i, item in enumerate(hdm05_files):
        print('Processing %i of %i (%s)' % (i, len(hdm05_files), item))
        clips, cls = process_file(item)
        #clips, _ = process_file_angles(item)
        hdm05_clips += clips
        hdm05_classes += cls    
    data_clips = np.array(hdm05_clips)
    data_classes = np.array(hdm05_classes)
    np.savez_compressed(os.getcwd() + tdataFolder + 'data_hdm05', clips=data_clips, classes=data_classes)


def msrAction3DData():
    datafolder = 'D:/University/projects/LSTMMotion/code/datasets/CNNMotion/motionsynth_data/data/processed/'
    targetFolder = '/../data/'

    
    classIds = [i for i in range(20)]
    
    #files = [os.getcwd() + folder + f for f in listdir(os.getcwd() + folder) if isfile(join(os.getcwd() + folder, f))]
    files = get_files(datafolder + 'action3D')
    
    AS1 = [2,3,5,6,10,13,18,20]
    AS2 = [1,4,7,8,9,11,12,14]
    AS3 = [6,14,15,16,17,18,19,20]
    trainSub = [1, 3, 5, 7, 9]
    testSub = [2, 4, 6, 8, 10]

    as1Traindata = []
    as1Trainlabels = []
    as1Testdata = []
    as1Testlabels = []

    as2Traindata = []
    as2Trainlabels = []
    as2Testdata = []
    as2Testlabels = []

    as3Traindata = []
    as3Trainlabels = []
    as3Testdata = []
    as3Testlabels = []

    #for i, f in enumerate(files):
    #    fh = open(f, "r")
    #    lineList = fh.readlines()
    #    fh.close()

    #    fh = open(f, "w")
    #    fh.writelines([item for item in lineList[:-1]])
    #    fh.close()

    #return
    
    for i, f in enumerate(files):
        
        m = re.search('a[0-9][0-9]', f)
        if m:
            index = int(m.group(0)[1:])


            clips, _ = process_file(f, window = 20, window_step = 5)

            if index in AS1:
                t = re.search('s[0-9][0-9]', f)
                if t:
                    sind = int(t.group(0)[1:])
                    if sind in trainSub:
                        if len(as1Traindata) != 0:
                            as1Traindata = np.concatenate((as1Traindata, clips), axis=0)
                        else:
                            as1Traindata = clips
                        tl = np.repeat([classIds[index-1]], len(clips), axis=0)
                        as1Trainlabels = np.concatenate((as1Trainlabels, tl), axis=0)
                    else:
                        if len(as1Testdata) != 0:
                            as1Testdata = np.concatenate((as1Testdata, clips), axis=0)
                        else:
                            as1Testdata = clips
                        tl = np.repeat([classIds[index-1]], len(clips), axis=0)
                        as1Testlabels = np.concatenate((as1Testlabels, tl), axis=0)
            if index in AS2:
                t = re.search('s[0-9][0-9]', f)
                if t:
                    sind = int(t.group(0)[1:])
                    if sind in trainSub:
                        if len(as2Traindata) != 0:
                            as2Traindata = np.concatenate((as2Traindata, clips), axis=0)
                        else:
                            as2Traindata = clips
                        tl = np.repeat([classIds[index-1]], len(clips), axis=0)
                        as2Trainlabels = np.concatenate((as2Trainlabels, tl), axis=0)
                    else:
                        if len(as2Testdata) != 0:
                            as2Testdata = np.concatenate((as2Testdata, clips), axis=0)
                        else:
                            as2Testdata = clips
                        tl = np.repeat([classIds[index-1]], len(clips), axis=0)
                        as2Testlabels = np.concatenate((as2Testlabels, tl), axis=0)
            if index in AS3:
                t = re.search('s[0-9][0-9]', f)
                if t:
                    sind = int(t.group(0)[1:])
                    if sind in trainSub:
                        if len(as3Traindata) != 0:
                            as3Traindata = np.concatenate((as3Traindata, clips), axis=0)
                        else:
                            as3Traindata = clips
                        tl = np.repeat([classIds[index-1]], len(clips), axis=0)
                        as3Trainlabels = np.concatenate((as3Trainlabels, tl), axis=0)
                    else:
                        if len(as3Testdata) != 0:
                            as3Testdata = np.concatenate((as3Testdata, clips), axis=0)
                        else:
                            as3Testdata = clips
                        tl = np.repeat([classIds[index-1]], len(clips), axis=0)
                        as3Testlabels = np.concatenate((as3Testlabels, tl), axis=0)



    np.savez_compressed(os.getcwd() + targetFolder + 'action3D_as1train', 
                     clips=as1Traindata, classes=as1Trainlabels)
    np.savez_compressed(os.getcwd() + targetFolder + 'action3D_as1test', 
                     clips=as1Testdata, classes=as1Testlabels)
    np.savez_compressed(os.getcwd() + targetFolder + 'action3D_as2train', 
                     clips=as2Traindata, classes=as2Trainlabels)
    np.savez_compressed(os.getcwd() + targetFolder + 'action3D_as2test', 
                     clips=as2Testdata, classes=as2Testlabels)
    np.savez_compressed(os.getcwd() + targetFolder + 'action3D_as3train', 
                     clips=as3Traindata, classes=as3Trainlabels)
    np.savez_compressed(os.getcwd() + targetFolder + 'action3D_as3test', 
                     clips=as3Testdata, classes=as3Testlabels)

def mhadData():
    datafolder = 'D:/University/projects/LSTMMotion/code/datasets/CNNMotion/motionsynth_data/data/processed/'
    tdataFolder = '/../data/'

    
    classIds = [i for i in range(12)]

    trainSubjects= ['s01','s02','s03','s04','s05','s06','s07']
    testSubjects= ['s08','s09','s10','s11','s12']

    files = get_files(datafolder + 'mhad')

    trainFiles = []
    testFiles = []
    for i, f in enumerate(files):

        m = re.match('.+skl_s[0-9][0-9]_a[0-9][0-9].+', f)
        if m:
            nn = re.match('.+(s0(1|2|3|4|5|6|7)).+', f)
            if nn:
                trainFiles = np.append(trainFiles, f)
            else:
                testFiles = np.append(testFiles, f)
    trainData = []
    trainLabels = []
    testData = []
    testLabels = []
    for i, f in enumerate(trainFiles):
        m = re.search('a[0-9][0-9]', f)
        index = int(m.group(0)[1:]) - 1

        clips, _ = process_file(f)
        trainData += clips
        tl = np.repeat([classIds[index]], len(clips), axis=0)
        trainLabels = np.concatenate((trainLabels, tl), axis=0)
    
    for i, f in enumerate(testFiles):
        m = re.search('a[0-9][0-9]', f)
        index = int(m.group(0)[1:]) - 1
        
        clips, _ = process_file(f)
        testData += clips
        tl = np.repeat([classIds[index]], len(clips), axis=0)
        testLabels = np.concatenate((testLabels, tl), axis=0)   
         
    trainData = np.array(trainData)
    testData = np.array(testData)




    np.savez_compressed(os.getcwd() + tdataFolder + 'data_mhad_train', 
                     clips=trainData, classes=trainLabels)
    np.savez_compressed(os.getcwd() + tdataFolder + 'data_mhad_test', 
                     clips=testData, classes=testLabels)


def nturgb60():
    datafolder = 'D:/University/projects/AdversarialAttack/ActionAttack/data/'
    tdataFolder = '/../data/'

    files = get_files(datafolder + 'nturgb+d_skeletons', 'mat')

    trainSubjects = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
    

    trainData = []
    trainDataLabels = []
    testData = []
    testDataLabels = []

    for i, f in enumerate(files):

        clips, labels = process_mat_file(f)

        m = re.search('P[0-9]+', f)
        subInd = int(m.group(0)[1:])

        if subInd in trainSubjects:
            trainData += clips
            trainDataLabels += labels
        else:
            testData += clips
            testDataLabels += labels


    np.savez_compressed(os.getcwd() + tdataFolder + 'train_data_nturgbd60', 
                     clips=np.array(trainData), classes=np.array(trainDataLabels))

    np.savez_compressed(os.getcwd() + tdataFolder + 'test_data_nturgbd60', 
                     clips=np.array(testData), classes=np.array(testDataLabels))

def nturgb120():
    datafolder = 'D:/University/projects/AdversarialAttack/ActionAttack/data/'
    tdataFolder = '/../data/'


    trainSubjects = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78, 80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103]

    trainData = []
    trainDataLabels = []
    testData = []
    testDataLabels = []

    files = get_files(datafolder + 'nturgb+d_skeletons', 'mat')

    files += get_files(datafolder + 'nturgbd_skeletons_s018_to_s032', 'mat')


    for i, f in enumerate(files):

        clips, labels = process_mat_file(f)

        m = re.search('P[0-9]+', f)
        subInd = int(m.group(0)[1:])

        if subInd in trainSubjects:
            trainData += clips
            trainDataLabels += labels
        else:
            testData += clips
            testDataLabels += labels


    np.savez_compressed(os.getcwd() + tdataFolder + 'train_data_nturgbd120', 
                     clips=np.array(trainData), classes=np.array(trainDataLabels))

    np.savez_compressed(os.getcwd() + tdataFolder + 'test_data_nturgbd120', 
                     clips=np.array(testData), classes=np.array(testDataLabels))

def clsMHADData():
    dataPath = '/../data/'
    tdataPath = '/../data/trainData/jointPositions/60_75/mhad/'

    trainData = np.load(os.getcwd() + dataPath + 'data_mhad_train.npz')['clips']
    trainLabels = np.load(os.getcwd() + dataPath + 'data_mhad_train.npz')['classes']


    X = trainData.reshape(trainData.shape[0], trainData.shape[1], -1)
    
    rng = np.random.RandomState(23456)

    X = np.swapaxes(X, 1, 2).astype(np.float32)

    Xmean = X.mean(axis=2).mean(axis=0)[np.newaxis,:,np.newaxis]

    Xstd = np.array([[[X.std()]]]).repeat(X.shape[1], axis=1)

    

    np.savez_compressed(os.getcwd() + tdataPath + 'train_preprocess_core.npz', Xmean=Xmean, Xstd=Xstd)

    
    X = (X - Xmean) / Xstd
    X = np.swapaxes(X, 1, 2)


    I = np.arange(len(X))
    I = np.random.permutation(I)  

    trainData = X[I]
    trainLabels = trainLabels[I]

    np.savez_compressed(os.getcwd() + tdataPath + '/classTrain', clips=trainData, classes=trainLabels)

    testData = np.load(os.getcwd() + dataPath + 'data_mhad_test.npz')['clips']
    testLabels = np.load(os.getcwd() + dataPath + 'data_mhad_test.npz')['classes']

    X = testData.reshape(testData.shape[0], testData.shape[1], -1)
    
    rng = np.random.RandomState(23456)

    X = np.swapaxes(X, 1, 2).astype(np.float32)

    Xmean = X.mean(axis=2).mean(axis=0)[np.newaxis,:,np.newaxis]

    Xstd = np.array([[[X.std()]]]).repeat(X.shape[1], axis=1)

    

    np.savez_compressed(os.getcwd() + tdataPath + 'test_preprocess_core.npz', Xmean=Xmean, Xstd=Xstd)

    
    X = (X - Xmean) / Xstd
    X = np.swapaxes(X, 1, 2)


    I = np.arange(len(X))
    I = np.random.permutation(I)  

    testData = X[I]
    testLabels = testLabels[I]

    
    np.savez_compressed(os.getcwd() + tdataPath + '/classTest', clips=testData, classes=testLabels)

    print ('train dataset shape: ', trainData.shape)
    print ('test dataset shape: ', testData.shape)

def clsHDM05Data():
    dataPath = '/../data/'
    tdataPath = '/../data/trainData/jointPositions/60_75/hdm05/'
    Xhdm05 = np.load(os.getcwd() + dataPath + 'data_hdm05.npz')['clips']
    labels = np.load(os.getcwd() + dataPath + 'data_hdm05.npz')['classes']

   
    X = Xhdm05.reshape(Xhdm05.shape[0], Xhdm05.shape[1], -1)
    
    rng = np.random.RandomState(23456)

    X = np.swapaxes(X, 1, 2).astype(np.float32)

    Xmean = X.mean(axis=2).mean(axis=0)[np.newaxis,:,np.newaxis]

    Xstd = np.array([[[X.std()]]]).repeat(X.shape[1], axis=1)

    

    np.savez_compressed(os.getcwd() + tdataPath + 'preprocess_core.npz', Xmean=Xmean, Xstd=Xstd)

    
    X = (X - Xmean) / Xstd
    X = np.swapaxes(X, 1, 2)

    trainRatio = 0.8

    I = np.arange(len(X))
    I = np.random.permutation(I)  
    num = int(np.floor(len(I) * trainRatio))

    trainData = X[I[:num]]
    trainLabel = labels[I[:num]]

    testData = X[I[num:]]
    testLabel = labels[I[num:]]




    np.savez_compressed(os.getcwd() + tdataPath + '/classTrain', clips=trainData, classes=trainLabel)
    np.savez_compressed(os.getcwd() + tdataPath + '/classTest', clips=testData, classes=testLabel)

    print ('train dataset shape: ', trainData.shape)
    print ('test dataset shape: ', testData.shape)

def clsNTURGBD120Data():
    dataPath = '/../data/'
    tdataPath = '/../data/trainData/jointPositions/nturgbd120/'
    trainFile = np.load(os.getcwd() + dataPath + 'train_data_nturgbd120.npz')
    Xhdm05 = trainFile['clips']
    labels = trainFile['classes']

 
    X = Xhdm05.reshape(Xhdm05.shape[0], Xhdm05.shape[1], -1)

    rng = np.random.RandomState(23456)

    X = np.swapaxes(X, 1, 2).astype(np.float32)

    Xmean = X.mean(axis=2).mean(axis=0)[np.newaxis,:,np.newaxis]

    Xstd = np.array([[[X.std()]]]).repeat(X.shape[1], axis=1)


    np.savez_compressed(os.getcwd() + tdataPath + 'train_preprocess_core.npz', Xmean=Xmean, Xstd=Xstd)

    
    X = (X - Xmean) / Xstd
    X = np.swapaxes(X, 1, 2)

    trainRatio = 1

    I = np.arange(len(X))
    I = np.random.permutation(I)  
    num = int(np.floor(len(I) * trainRatio))

    trainData = X[I[:num]]
    trainLabel = labels[I[:num]]




    np.savez_compressed(os.getcwd() + tdataPath + '/classTrain', clips=trainData, classes=trainLabel)

    testFile = np.load(os.getcwd() + dataPath + 'test_data_nturgbd120.npz')
    Xhdm05 = testFile['clips']
    labels = testFile['classes']

 
    X = Xhdm05.reshape(Xhdm05.shape[0], Xhdm05.shape[1], -1)

    rng = np.random.RandomState(23456)

    X = np.swapaxes(X, 1, 2).astype(np.float32)

    Xmean = X.mean(axis=2).mean(axis=0)[np.newaxis,:,np.newaxis]

    Xstd = np.array([[[X.std()]]]).repeat(X.shape[1], axis=1)


    np.savez_compressed(os.getcwd() + tdataPath + 'test_preprocess_core.npz', Xmean=Xmean, Xstd=Xstd)

    
    X = (X - Xmean) / Xstd
    X = np.swapaxes(X, 1, 2)

 
    I = np.arange(len(X))
    I = np.random.permutation(I)  
    num = int(np.floor(len(I) * trainRatio))

    testData = X[I[:num]]
    testLabel = labels[I[:num]]


    np.savez_compressed(os.getcwd() + tdataPath + '/classTest', clips=testData, classes=testLabel)

    print ('train dataset shape: ', trainData.shape)
    print ('test dataset shape: ', testData.shape)

def clsNTURGBD60Data():
    dataPath = '/../data/'
    tdataPath = '/../data/trainData/jointPositions/nturgbd60/'
    trainFile = np.load(os.getcwd() + dataPath + 'train_data_nturgbd60.npz')
    Xhdm05 = trainFile['clips']
    labels = trainFile['classes']

 
    X = Xhdm05.reshape(Xhdm05.shape[0], Xhdm05.shape[1], -1)

    rng = np.random.RandomState(23456)

    X = np.swapaxes(X, 1, 2).astype(np.float32)

    Xmean = X.mean(axis=2).mean(axis=0)[np.newaxis,:,np.newaxis]

    Xstd = np.array([[[X.std()]]]).repeat(X.shape[1], axis=1)


    np.savez_compressed(os.getcwd() + tdataPath + 'train_preprocess_core.npz', Xmean=Xmean, Xstd=Xstd)

    
    X = (X - Xmean) / Xstd
    X = np.swapaxes(X, 1, 2)

    trainRatio = 1

    I = np.arange(len(X))
    I = np.random.permutation(I)  
    num = int(np.floor(len(I) * trainRatio))

    trainData = X[I[:num]]
    trainLabel = labels[I[:num]]




    np.savez_compressed(os.getcwd() + tdataPath + '/classTrain', clips=trainData, classes=trainLabel)

    testFile = np.load(os.getcwd() + dataPath + 'test_data_nturgbd60.npz')
    Xhdm05 = testFile['clips']
    labels = testFile['classes']

 
    X = Xhdm05.reshape(Xhdm05.shape[0], Xhdm05.shape[1], -1)

    rng = np.random.RandomState(23456)

    X = np.swapaxes(X, 1, 2).astype(np.float32)

    Xmean = X.mean(axis=2).mean(axis=0)[np.newaxis,:,np.newaxis]

    Xstd = np.array([[[X.std()]]]).repeat(X.shape[1], axis=1)


    np.savez_compressed(os.getcwd() + tdataPath + 'test_preprocess_core.npz', Xmean=Xmean, Xstd=Xstd)

    
    X = (X - Xmean) / Xstd
    X = np.swapaxes(X, 1, 2)

 
    I = np.arange(len(X))
    I = np.random.permutation(I)  
    num = int(np.floor(len(I) * trainRatio))

    testData = X[I[:num]]
    testLabel = labels[I[:num]]


    np.savez_compressed(os.getcwd() + tdataPath + '/classTest', clips=testData, classes=testLabel)

    print ('train dataset shape: ', trainData.shape)
    print ('test dataset shape: ', testData.shape)

def clsAction3DData():
    dataPath = '/../data/'
    tdataPath = '/../data/trainData/jointPositions/action3D/'

    
    dataStandardisation(dataPath, tdataPath, 'action3D_as1train.npz', trainRatio = 1, 
                        coreFileName = 'as1train_preprocess_core.npz', 
                        trainDataFile = 'as1_classTrain',
                        testDataFile = '')

    dataStandardisation(dataPath, tdataPath, 'action3D_as1test.npz', trainRatio = 1, 
                        coreFileName = 'as1test_preprocess_core.npz', 
                        trainDataFile = 'as1_classTest',
                        testDataFile = '')

    dataStandardisation(dataPath, tdataPath, 'action3D_as2train.npz', trainRatio = 1, 
                        coreFileName = 'as2train_preprocess_core.npz', 
                        trainDataFile = 'as2_classTrain',
                        testDataFile = '')

    dataStandardisation(dataPath, tdataPath, 'action3D_as2test.npz', trainRatio = 1, 
                        coreFileName = 'as2test_preprocess_core.npz', 
                        trainDataFile = 'as2_classTest',
                        testDataFile = '')
    dataStandardisation(dataPath, tdataPath, 'action3D_as3train.npz', trainRatio = 1, 
                        coreFileName = 'as3train_preprocess_core.npz', 
                        trainDataFile = 'as3_classTrain',
                        testDataFile = '')

    dataStandardisation(dataPath, tdataPath, 'action3D_as3test.npz', trainRatio = 1, 
                        coreFileName = 'as3test_preprocess_core.npz', 
                        trainDataFile = 'as3_classTest',
                        testDataFile = '')


def dataStandardisation(dataPath, tdataPath, dataFile, trainRatio = 1, 
                        coreFileName = 'preprocess_core.npz', trainDataFile = 'classTrain',
                        testDataFile = 'classTest'):

    data = np.load(os.getcwd() + dataPath + dataFile)['clips']
    labels = np.load(os.getcwd() + dataPath + dataFile)['classes']

   
    X = data.reshape(data.shape[0], data.shape[1], -1)
    
    rng = np.random.RandomState(23456)

    X = np.swapaxes(X, 1, 2).astype(np.float32)

    Xmean = X.mean(axis=2).mean(axis=0)[np.newaxis,:,np.newaxis]

    Xstd = np.array([[[X.std()]]]).repeat(X.shape[1], axis=1)

    

    np.savez_compressed(os.getcwd() + tdataPath + coreFileName, Xmean=Xmean, Xstd=Xstd)

    
    X = (X - Xmean) / Xstd
    X = np.swapaxes(X, 1, 2)


    I = np.arange(len(X))
    I = np.random.permutation(I)  
    num = int(np.floor(len(I) * trainRatio))

    trainData = X[I[:num]]
    trainLabel = labels[I[:num]]

    np.savez_compressed(os.getcwd() + tdataPath + trainDataFile, clips=trainData, classes=trainLabel)
    print ('train dataset shape: ', trainData.shape)

    if trainRatio < 1:
        testData = X[I[num:]]
        testLabel = labels[I[num:]]
    
        np.savez_compressed(os.getcwd() + tdataPath + testDataFile, clips=testData, classes=testLabel)

    
        print ('test dataset shape: ', testData.shape)

def process_bvh_file(filename, window=60, window_step=20):

    print('Processing file %s', filename)



    anim, names, frametime = BVH.load(filename)
    
       
    #convert to 30 fps
    #anim = anim[::4]


    """ Do FK """
    global_positions = Animation.positions_global(anim)


    """ Remove Uneeded Joints """
    positions = global_positions[:,np.array([
            1,  2,  3,  4,  5,
            6,  7,  8,  9, 10,
        0, 12, 13, 14, 15,
        18, 19, 20, 22, 23,
        25, 26, 27, 29, 30])]

    #""" Put on Floor """
    #fid_l, fid_r = np.array([3,4]), np.array([7,8])
    #foot_heights = np.minimum(positions[:,fid_l,1], positions[:,fid_r,1]).min(axis=1)
    #floor_height = softmin(foot_heights, softness=0.5, axis=0)

    #positions[:,:,1] -= floor_height

    #""" Add Reference Joint """
    #trajectory_filterwidth = 3
    #reference = positions[:,0] * np.array([1,0,1])
    #reference = filters.gaussian_filter1d(reference, trajectory_filterwidth, axis=0, mode='nearest')    
    #positions = np.concatenate([reference[:,np.newaxis], positions], axis=1)

    #""" Get Foot Contacts """
    #velfactor, heightfactor = np.array([0.05,0.05]), np.array([3.0, 2.0])

    #feet_l_x = (positions[1:,fid_l,0] - positions[:-1,fid_l,0])**2
    #feet_l_y = (positions[1:,fid_l,1] - positions[:-1,fid_l,1])**2
    #feet_l_z = (positions[1:,fid_l,2] - positions[:-1,fid_l,2])**2
    #feet_l_h = positions[:-1,fid_l,1]
    #feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)

    #feet_r_x = (positions[1:,fid_r,0] - positions[:-1,fid_r,0])**2
    #feet_r_y = (positions[1:,fid_r,1] - positions[:-1,fid_r,1])**2
    #feet_r_z = (positions[1:,fid_r,2] - positions[:-1,fid_r,2])**2
    #feet_r_h = positions[:-1,fid_r,1]
    #feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)

    #""" Get Root Velocity """
    #velocity = (positions[1:,0:1] - positions[:-1,0:1]).copy()

    """ Remove Translation """
    positions[:,:] = positions[:,:] - positions[:,10:11]

    """ Get Forward Direction """
    sdr_l, sdr_r, hip_l, hip_r = 15, 20, 0, 5
    across1 = positions[:,hip_l] - positions[:,hip_r]
    across0 = positions[:,sdr_l] - positions[:,sdr_r]
    across = across0 + across1
    across = across / np.sqrt((across**2).sum(axis=-1))[...,np.newaxis]

    direction_filterwidth = 20
    forward = np.cross(across, np.array([[0,1,0]]))
    forward = filters.gaussian_filter1d(forward, direction_filterwidth, axis=0, mode='nearest')    
    forward = forward / np.sqrt((forward**2).sum(axis=-1))[...,np.newaxis]

    #""" Remove Y Rotation """
    target = np.array([[0,0,1]]).repeat(len(forward), axis=0)
    rotation = Quaternions.between(forward, target)[:,np.newaxis]    
    positions = rotation * positions

    #""" Get Root Rotation """
    #velocity = rotation[1:] * velocity
    #rvelocity = Pivots.from_quaternions(rotation[1:] * -rotation[:-1]).ps

    #""" Add Velocity, RVelocity, Foot Contacts to vector """

    #positions = positions[:-1]
    #positions = positions.reshape(len(positions), -1)
    #positions = np.concatenate([positions, velocity[:,:,0]], axis=-1)
    #positions = np.concatenate([positions, velocity[:,:,2]], axis=-1)
    #positions = np.concatenate([positions, rvelocity], axis=-1)
    #positions = np.concatenate([positions, feet_l, feet_r], axis=-1)

    """ Slide over windows """
    windows = []
    windows_classes = []

    if len(positions) <= window:

        left  = positions[:1].repeat((window-len(positions))//2 + (window-len(positions))%2, axis=0)
        right = positions[-1:].repeat((window-len(positions))//2, axis=0)
        slice = np.concatenate([left, positions, right], axis=0)
        if len(slice) != window: raise Exception()
        windows.append(slice)

        """ Find Class """
        cls = -1
        head, tail = ntpath.split(filename)
        if tail.startswith('HDM'):
            cls_name = os.path.splitext(os.path.split(filename)[1])[0][7:-8]
            cls = class_names.index(class_map[cls_name]) if cls_name in class_map else -1
        windows_classes.append(cls)

        return windows, windows_classes

    for j in range(0, len(positions)-window, window_step):

        """ If slice too small pad out by repeating start and end poses """
        slice = positions[j:j+window]
        if len(slice) < window:
            left  = slice[:1].repeat((window-len(slice))//2 + (window-len(slice))%2, axis=0)
            right = slice[-1:].repeat((window-len(slice))//2, axis=0)
            slice = np.concatenate([left, slice, right], axis=0)
        if len(slice) != window: raise Exception()
        
        windows.append(slice)

        """ Find Class """
        cls = -1
        head, tail = ntpath.split(filename)
        if tail.startswith('HDM'):
            cls_name = os.path.splitext(os.path.split(filename)[1])[0][7:-8]
            cls = class_names.index(class_map[cls_name]) if cls_name in class_map else -1
        windows_classes.append(cls)


    return windows, windows_classes


def process_mat_file(filename, window=60, window_step=20):

    

    print('Processing file %s', filename)

    m = re.search('A[0-9]+', filename)


    index = int(m.group(0)[1:]) - 1
    
    positions = scipy.io.loadmat(filename)["motion"]

    #nanInd = np.isnan(positions)

    #result = np.where(nanInd == True)

    #if any(result) > 0:
    #    pdb.set_trace()

    #joint index lLeg, rLeg, trunk (including root), lArm, rArm
    
    

    """ Remove Translation """
    positions[:,:] = positions[:,:] - positions[:,10:11]

    """ Get Forward Direction """
    sdr_l, sdr_r, hip_l, hip_r = 15, 20, 0, 5
    across1 = positions[:,hip_l] - positions[:,hip_r]
    across0 = positions[:,sdr_l] - positions[:,sdr_r]
    across = across0 + across1
    across = across / np.sqrt((across**2).sum(axis=-1))[...,np.newaxis]

    direction_filterwidth = 20
    forward = np.cross(across, np.array([[0,1,0]]))
    forward = filters.gaussian_filter1d(forward, direction_filterwidth, axis=0, mode='nearest')    
    forward = forward / np.sqrt((forward**2).sum(axis=-1))[...,np.newaxis]

    #""" Remove Y Rotation """
    target = np.array([[0,0,1]]).repeat(len(forward), axis=0)
    rotation = Quaternions.between(forward, target)[:,np.newaxis]    
    positions = rotation * positions

    nanInd = np.isnan(positions)

    result = np.where(nanInd == True)

    if len(result[0]) > 0:
        print('File %s contains NaN, skip it', filename)
        return [], []

    """ Slide over windows """
    windows = []
    windows_classes = []

    if len(positions) <= window:

        left  = positions[:1].repeat((window-len(positions))//2 + (window-len(positions))%2, axis=0)
        right = positions[-1:].repeat((window-len(positions))//2, axis=0)
        slice = np.concatenate([left, positions, right], axis=0)
        if len(slice) != window: raise Exception()
        windows.append(slice)
        windows_classes.append(index)

        #nanInd = np.isnan(windows)

        #result = np.where(nanInd == True)

        #if len(result[0]) > 0:
        #    pdb.set_trace()

        return windows, windows_classes

    for j in range(0, len(positions)-window, window_step):

        """ If slice too small pad out by repeating start and end poses """
        slice = positions[j:j+window]
        if len(slice) < window:
            left  = slice[:1].repeat((window-len(slice))//2 + (window-len(slice))%2, axis=0)
            right = slice[-1:].repeat((window-len(slice))//2, axis=0)
            slice = np.concatenate([left, slice, right], axis=0)
        if len(slice) != window: raise Exception()
        
        windows.append(slice)
        windows_classes.append(index)


    #nanInd = np.isnan(windows)

    #result = np.where(nanInd == True)

    #if len(result[0]) > 0:
    #    pdb.set_trace()
    return windows, windows_classes

def main():
    #classificationData()
    clsHDM05Data()
    #mhadData()
    #clsMHADData()
    #msrAction3DData()
    #clsAction3DData()
    #nturgb60()
    #clsNTURGBD60Data()
    #nturgb120()
    #clsNTURGBD120Data()

if __name__ == '__main__':
    main()