import numpy as np

import os

import pdb

import re


def calAccuracies(folder = ''):

    subFolders = [f for f in list(os.listdir(folder)) if re.search('^batch', f) != None]

    sa = 0
    us = 0
    ab = 0
    abNum = 0
    usNum = 0
    saNum = 0


    for rfolder in subFolders:

        maxsa = 0
        maxus = 0
        maxab = 0

        elements = rfolder.split('_')

        at = elements[1]


        if at == 'ab':
            abNum += 1
        elif at == 'us':
            usNum += 1
        elif at == 'sa':
            saNum += 1
        else:
            print('Unknown attack type')


        files = [f for f in list(os.listdir(folder + '/' + rfolder)) if os.path.splitext(f)[1] == '.npz']


        for file in files:

            felements = os.path.splitext(file)[0].split('_')
            
            sr = felements[-1]
            if sr == 'rets':
                continue
            
            if at == 'ab':
                if float(sr) > maxab:
                    maxab = float(sr)
            elif at == 'us':
                if float(sr) > maxus:
                    maxus = float(sr)
            elif at == 'sa':
                
                if float(sr) > maxsa:
                    maxsa = float(sr)
            else:
                print('Unknown attack type')

        for file in files:

            felements, ext = os.path.splitext(file)

            if ext != '.npz':
                continue

            felements = os.path.splitext(file)[0].split('_')
            
            sr = felements[-1]
            if sr == 'rets':
                continue

            sr = float(sr)
            
            if at == 'ab':
                if sr == maxab:
                    continue
            elif at == 'us':
                if sr == maxus:
                    continue
            elif at == 'sa':
                if sr == maxsa:
                    continue
            else:
                print('Unknown attack type')
                return

            os.remove(folder + '/' + rfolder + '/' + file)

        if at == 'ab':
            ab += maxab
        elif at == 'us':
            us += maxus
        elif at == 'sa':
            sa += maxsa
        else:
            print('Unknown attack type')
        if abNum == 0:
            abNum = 1e-8
        if usNum == 0:
            usNum = 1e-8
        if saNum == 0:
            saNum = 1e-8
    print('ab sr %f, us sr %f, sa sr %f' % (ab / abNum, us/usNum, sa/saNum))





from scipy.stats import entropy

# def calTopNAccuraries(file='', topN = [1, 3, 5]):
#     data = np.load(file)
#
#     rresults = data['rawRets']
#     tlabels = data['tlabels']
#
#
#
#     findices = np.zeros(len(topN))
#     for i in range(len(rresults)):
#         #ent = entropy(rresults[i])
#         ranked = np.argsort(rresults[i])
#         for ind in range(len(topN)):
#             if len(np.where(ranked[:topN[ind]] == tlabels[i])[0]) > 0:
#                 findices[ind] += 1
#
#
#     return findices, len(rresults)
#
#
# def calAllTopN(folder, topN = [1, 3, 5]):
#
#     subFolders = [f for f in list(os.listdir(folder)) if re.search('^batch', f) != None]
#
#     frets = np.zeros(len(topN))
#     totalNum = 0
#
#     for rfolder in subFolders:
#
#
#         elements = rfolder.split('_')
#
#         at = elements[1]
#
#         if at != 'us':
#             continue
#
#
#         files = [f for f in list(os.listdir(folder + '/' + rfolder)) if os.path.splitext(f)[1] == '.npz']
#
#         for file in files:
#             felements = os.path.splitext(file)[0].split('_')
#
#             sr = felements[-1]
#             if sr != 'rets':
#                 continue
#
#             rets, num = calTopNAccuraries(folder + '/' + rfolder + '/' + file, topN)
#
#             frets = frets + rets
#             totalNum = totalNum + num
#
#
#     for i in range(len(topN)):
#         print('Top %d accuracy %f' % (topN[i], 1- frets[i]/totalNum))
#
#
# #from  matplotlib import pyplot as plt
# def jointPerturbationAnalysis(file='', folder = ''):
#
#     data = np.load(file)
#
#     motions = data['clips']
#     orMotions = data['oriClips']
#     plabels = data['classes']
#     tlabels = data['tclasses']
#
#     pickedJoints = [1, 2, 3, 4,
#                     6, 7, 8, 9,
#                     11, 12, 13, 14,
#                     15, 16, 17, 18, 19,
#                     20, 21, 22, 23, 24]
#
#     motions = np.reshape(motions, (motions.shape[0], motions.shape[1], -1, 3))[:, :, pickedJoints]
#     orMotions = np.reshape(orMotions, (orMotions.shape[0], orMotions.shape[1], -1, 3))[:, :, pickedJoints]
#
#     motionVel = np.sqrt(np.sum(np.square(motions[:, 1:, :] - motions[:, :-1, :]), axis=-1))[:, :-1, :]
#     orMotionVel = np.sqrt(np.sum(np.square(orMotions[:, 1:, :] - orMotions[:, :-1, :]), axis=-1))[:, :-1, :]
#
#     motionAcc = np.sqrt(np.sum(np.square(motions[:, 2:, :] - 2 * motions[:, 1:-1, :] + motions[:, :-2, :]), axis=-1))
#     orMotionAcc = np.sqrt(np.sum(np.square(orMotions[:, 2:, :] - 2 * orMotions[:, 1:-1, :] + orMotions[:, :-2, :]), axis=-1))
#
#     perturbations = np.sqrt(np.sum(np.square(np.reshape(motions - orMotions, (motions.shape[0], motions.shape[1], -1, 3))), axis=-1))
#
#     trimed = perturbations[:, 1:-1, :]
#
#     fMotionVel = np.reshape(orMotionVel, (-1, len(pickedJoints)))
#     fMotionAcc = np.reshape(orMotionAcc, (-1, len(pickedJoints)))
#     trimed = np.reshape(trimed, (-1, len(pickedJoints)))
#
#
#     pertCorr = np.corrcoef(trimed, trimed, rowvar = False)[:trimed.shape[1], :trimed.shape[1]]
#
#     #jointNames = ['lShould, lElbow, lWrite, lThumb, lFinger',
#     #              'rShould, rElbow, rWrite, rThumb, rFinger']
#
#     f = plt.figure(figsize=(19, 15))
#     plt.matshow(pertCorr, fignum=f.number)
#     plt.xticks(range(pertCorr.shape[1]), fontsize=14, rotation=45)
#     plt.yticks(range(pertCorr.shape[1]), fontsize=14)
#     cb = plt.colorbar()
#     cb.ax.tick_params(labelsize=20)
#     plt.title('Perturbation Correlation Matrix', fontsize=16);
#     plt.savefig(fname= folder + '/pertCorr.png')
#     plt.close(f)
#
#     pdb.set_trace()
#
#     correlations = np.corrcoef(fMotionVel, trimed, rowvar = False)[:fMotionVel.shape[1], :trimed.shape[1]]
#     f1 = plt.figure(figsize=(19, 15))
#     plt.matshow(correlations, fignum=f1.number)
#     plt.xticks(range(correlations.shape[1]), fontsize=14, rotation=45)
#     plt.yticks(range(correlations.shape[1]), fontsize=14)
#     cb = plt.colorbar()
#     cb.ax.tick_params(labelsize=20)
#     plt.title('Perturbation-vel Correlation Matrix', fontsize=16);
#     plt.savefig(fname = folder + '/pert-vel.png')
#     plt.close(f1)
#
#     correlations = np.corrcoef(fMotionAcc, trimed, rowvar = False)[:fMotionAcc.shape[1], :trimed.shape[1]]
#     f1 = plt.figure(figsize=(19, 15))
#     plt.matshow(correlations, fignum=f1.number)
#     plt.xticks(range(correlations.shape[1]), fontsize=14, rotation=45)
#     plt.yticks(range(correlations.shape[1]), fontsize=14)
#     cb = plt.colorbar()
#     cb.ax.tick_params(labelsize=20)
#     plt.title('Perturbation-acc Correlation Matrix', fontsize=16);
#     plt.savefig(fname = folder + '/pert-acc.png')
#     plt.close(f1)
#
# def pertAnalysis(folder = ''):
#     subFolders = [f for f in list(os.listdir(folder)) if re.search('^batch', f) != None]
#
#     for rfolder in subFolders:
#
#         files = [f for f in list(os.listdir(folder + '/' + rfolder)) if os.path.splitext(f)[1] == '.npz']
#
#         for file in files:
#
#             felements = os.path.splitext(file)[0].split('_')
#
#             sr = felements[-1]
#             if sr == 'rets':
#                 continue
#
#             jointPerturbationAnalysis(folder + '/' + rfolder + '/' + file, folder + '/' + rfolder)
#
#
# #from sklearn.metrics import confusion_matrix
# def confusionMat(file='', folder = './', fileName = 'confuseMat'):
#
#     data = np.load(file)
#
#     motions = data['clips']
#     orMotions = data['oriClips']
#     plabels = data['classes']
#     tlabels = data['tclasses']
#
#
#
#     hitIndices = []
#
#     for i in range(0, len(tlabels)):
#         if plabels[i] != tlabels[i]:
#             hitIndices.append(i)
#
#     hitIndices = np.array(hitIndices)
#
#     tclasses = tlabels[hitIndices]
#     pclasses = plabels[hitIndices]
#
#
#     #if os.path.isfile(folder + 'classes.txt'):
#     #    with open(folder + 'classes.txt') as f:
#     #        content = f.readlines()
#     #        labels = [x.strip() for x in content]
#
#     #labelsInds = np.array(np.union1d(tclasses, pclasses))
#
#
#     #labels = [labels[i] for i in labelsInds]
#
#     #mat = confusion_matrix(tclasses, pclasses)
#
#
#     #f = plt.figure(figsize=(19, 15))
#     #plt.matshow(mat, fignum=f.number)
#     #plt.xticks(range(mat.shape[1]), fontsize=14, rotation=45)
#     #plt.yticks(range(mat.shape[1]), fontsize=14)
#     #cb = plt.colorbar()
#     #cb.ax.tick_params(labelsize=14)
#     ##plt.title('Confusion Matrix on ', fontsize=16);
#     #plt.savefig(fname= folder + '/confusion.png')
#     #plt.close(f)
#
#     if os.path.isfile(folder + 'classes.txt'):
#         with open(folder + 'classes.txt') as f:
#             content = f.readlines()
#             class_names = [x.strip() for x in content]
#
#     class_names = [i for i in range(len(class_names))]
#
#     np.set_printoptions(precision=2)
#
#     ## Plot non-normalized confusion matrix
#     #f, ax = plot_confusion_matrix(tclasses, pclasses, classes=class_names,
#     #                      title='Confusion matrix, without normalization')
#
#     # Plot normalized confusion matrix
#     f, ax = plot_confusion_matrix(tclasses, pclasses, classes=class_names, normalize=True,
#                           title='Normalized confusion matrix')
#
#     plt.savefig(fname= folder + '/' + fileName + '.png')
#     plt.close(f)
#
#
# #from sklearn.utils.multiclass import unique_labels
# def plot_confusion_matrix(y_true, y_pred, classes,
#                           normalize=False,
#                           title=None,
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if not title:
#         if normalize:
#             title = 'Normalized confusion matrix'
#         else:
#             title = 'Confusion matrix, without normalization'
#
#     title = ''
#     # Compute confusion matrix
#     cm = confusion_matrix(y_true, y_pred)
#     # Only use the labels that appear in the data
#     #classes = classes[unique_labels(y_true, y_pred)]
#     classes = [classes[i] for i in np.array(unique_labels(y_true, y_pred), dtype=np.int)]
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#
#     #print(cm)
#
#     fig, ax = plt.subplots(figsize=(19, 15))
#     im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
#     #ax.figure.colorbar(im, ax=ax)
#     # We want to show all ticks...
#     ax.set(xticks=np.arange(cm.shape[1]),
#            yticks=np.arange(cm.shape[0]),
#            # ... and label them with the respective list entries
#            xticklabels=classes, yticklabels=classes,
#            title=title,
#            ylabel='True label',
#            xlabel='Predicted label')
#
#     plt.xlabel('Predicted label', fontsize=18)
#     plt.ylabel('True label', fontsize=18)
#
#     # Rotate the tick labels and set their alignment.
#     plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#              rotation_mode="anchor")
#
#     # Loop over data dimensions and create text annotations.
#     #fmt = '.2f' if normalize else 'd'
#     #thresh = cm.max() / 2.
#     #for i in range(cm.shape[0]):
#     #    for j in range(cm.shape[1]):
#     #        ax.text(j, i, format(cm[i, j], fmt),
#     #                ha="center", va="center",
#     #                color="white" if cm[i, j] > thresh else "black")
#     fig.tight_layout()
#     return fig, ax
#
#
#
# def confusionMatOnDataset(folder = ''):
#
#     subFolders = [f for f in list(os.listdir(folder)) if re.search('^batch', f) != None]
#
#     for rfolder in subFolders:
#
#         files = [f for f in list(os.listdir(folder + '/' + rfolder)) if os.path.splitext(f)[1] == '.npz']
#
#         for file in files:
#
#             felements = os.path.splitext(file)[0].split('_')
#
#             sr = felements[-1]
#             if sr == 'rets':
#                 continue
#
#             jointPerturbationAnalysis(folder + '/' + rfolder + '/' + file, folder + '/' + rfolder)
#
#
# def removeWhiteMargin(folder = ''):
#
#     from PIL import Image
#
#     files = [f for f in list(os.listdir(folder)) if os.path.splitext(f)[1] == '.png']
#
#     for file in files:
#
#         img = Image.open(folder + '/' + file)
#
#         data = img.load()
#
#         minx = int(img.size[0]/2)
#         miny = int(img.size[1]/2)
#         maxx = int(img.size[0]/2)
#         maxy = int(img.size[1]/2)
#
#         for i in range(img.size[0]):
#             for j in range(img.size[1]):
#
#                 if data[i,j] != (255, 255, 255, 255):
#                     if i < minx:
#                         minx = i
#                     if j < miny:
#                         miny = j
#                     if i > maxx:
#                         maxx = i
#                     if j > maxy:
#                         maxy = j
#         img = img.crop((minx-2, miny-2, maxx+2, maxy+2))
#         img.save(folder + '/' + file)

if __name__ == '__main__':
    calAccuracies('../results/ntu60/STGCN/CIASA/')
    #calTopNAccuraries('../data/trainData/jointPositions/hdm05/batch0_ab_clw_0.40_pl_acc-bone_plw_0.60_Adam/adTrainAdExamples_maxFoolRate_batch0_ab_clw_0.40_pl_acc-bone_plw_0.60_Adam_fr_100.00.npz_rets.npz')
    #calAllTopN('../data/trainData/jointPositions/nturgbd60/us')
    #jointPerturbationAnalysis('../data/trainData/jointPositions/hdm05/batch1_ab_clw_0.40_pl_acc-bone_plw_0.60_Adam/adTrainAdExamples_maxFoolRate_batch1_ab_clw_0.40_pl_acc-bone_plw_0.60_Adam_fr_100.00.npz')
    #pertAnalysis('../data/trainData/jointPositions/nturgbd60/')

    dataset = 'ntu60' 
    model = 'hrnn'

    #confusionMat('../results/' +model+'/' + dataset + '/ab_combinedRets.npz', folder='../results/' +model+'/'+dataset + '/', fileName='ab_confuseMat_1')
    #jointPerturbationAnalysis(file='../results/' +model+'/' + dataset + '/ab_combinedRets.npz', folder='../results/' +model+'/'+dataset + '/')
    #confusionMat('../results/' +model+'/' + dataset + '/us_combinedRets.npz', folder='../results/' +model+'/'+dataset + '/', fileName='us_confuseMat')
    #removeWhiteMargin('../results/' +model+'/' + dataset)