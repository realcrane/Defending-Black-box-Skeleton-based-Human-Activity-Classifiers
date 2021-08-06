import pdb

import numpy as np
import pickle as pk
if __name__ == '__main__':

    import argparse

    ap = argparse.ArgumentParser()

    ap.add_argument("-data", "--dataFile", type=str, required=True)
    ap.add_argument("-labels", "--labelFile", type=str, required=True)
    ap.add_argument("-output", "--outputFile", type=str, required=True)


    args = ap.parse_args()

    dataFile = args.dataFile
    labelFile = args.labelFile
    outputFile = args.outputFile

    data = np.load(dataFile)
    with open(labelFile, 'rb') as f:
        labels = pk.load(f)

    # downsample every sample from 300 frames to 60 frames.
    data = data[:, :, ::5, :, :]
    np.savez(outputFile, clips=data, classes=np.array(labels[1]))