# Defending Black-box Skeleton-based Human Activity Classifiers

## Description

This is the source code of our AAAI 2023 paper: Defending Black-box Skeleton-based Human Activity Classifiers.

Paper here: https://arxiv.org/abs/2203.04713

## Getting Started

### Dependencies

Below is the key environment under which the code was developed, not necessarily the minimal requirements:

1. Python 3.9.5
2. pytorch 1.8.1
3. Cuda 10.1

And other libraries such as numpy.

### Installing
No installation needed other than dependencies.

### Warning
The code has not been exhaustively tested. You need to run it at your own risk. The author will try to actively maintain it and fix reported bugs but this can be delayed.

### Executing program

* The code assumes that you have normalised your data and know how to recover it after learning.
* The code by default assumes every dataset is pre-processed so that there are two files: 
  the training data file and the testing data file. 
* The specific parameters of the programme are all in main.py
* The code only provides a 3-layer MLP classifier (in classifiers.ThreeLayerMLP) as an example. Although BEAT can be used to 
  train almost every skeleton-based action classifier, providing all the source is impossible. However, the code is structured
  in the way that you can create/train you own ActionClassifier class (in classifiers.ActionClassifier) where it is easy to embed other classifiers.
* HDM05 dataset is used as an example in this document.
* When you have a classifier, you can first train it by:
    ```
    python main.py -classifier 3layerMLP --routine train --dataset {dataset_name} --trainFile {training data file, default: classTrain.npz} --testFile {training data file, default: classTest.npz} --dataPath {path-for-data} --retPath {path-for-results} -ep 200 -cn 65
    ```
    * Training the classifier will generate two models by default: the minimal loss model minLossModel.pth and the minimal validation loss model minValLossModel.pth under --retPath
    * The source code only comes with a small pre-processed HDM05 dataset (http://resources.mpi-inf.mpg.de/HDM05/), under data/hdm05/, with four files classTrain.npz (training data), classTest.npz (testing data), hdm05_classes.txt (class labels) and preprocess-core.npz (mean and std of the data)
    * Although SMART does not assume any specific data format, for the purpose of the demo, we explain the format of the data provided:
        * classTrain.npz and classTest.npz are normalised data with two fields 'clips' and 'classes'
        * 'clips' is an array of [motionNo, frameNo, 75] (each frame has 75 joint coordinates of 25 joints). 
        * 'classes' are the action labels.
* If you have a trained classifier, you can first test it by:
    ```
    python main.py -classifier 3layerMLP --routine test --dataset {dataset_name} --testFile {training data file, default: classTest.npz} --trainedModelFile {savedModel} --dataPath {path-for-data} -retPath {path-for-results} -cn 65
    ```
* If you have a trained classifier, you can collect all the correctly recognised samples by:
    ```
    python main.py -classifier 3layerMLP --routine gatherCorrectPrediction --dataset {dataset_name} --testFile {training data file, default: classTest.npz} --trainedModelFile {savedModel} --dataPath {path-for-data} -retPath {path-for-results} -cn 65
    ```
    * This process by default generates a file called adClassTrain.npz under --retFolder, which contains all the correctly recognised samples for attack.
* Finally, you can attack the model by:
    ```
    python main.py --routine attack --attackType ab --dataset {dataset_name} -classifier 3layerMLP --epochs 1000 --batchSize 2 --trainedModelFile {savedModel} --trainFile {data file for samples to attack} --dataPath {path-for-data} -retPath {path-for-results} -cn 65
    ```
* After attack, we provide a not-so-structured code snippet for unnormalising the adversarial samples in post-processing.py
* Then you can repeat the above train/testing/sampleGathering using BEAT on your trained classifier:
    ```
    python main.py -classifier ExtendedBayesian --baseClassifier 3layerMLP --routine bayesianTrain --dataset hdm05 -adTrainer EBMATrainer --trainFile classTrain.npz --testFile classTest.npz --dataPath ../data/ --retPath ../results/ -ep 20 -cn 65 --trainedModelFile minValLossModel.pth --bayesianModelNum 5
    python main.py -classifier ExtendedBayesian --baseClassifier 3layerMLP  --routine bayesianTest --dataset hdm05 -adTrainer EBMATrainer --trainFile classTrain.npz --testFile classTest.npz --dataPath ../data/ --retPath ../results/ -ep 20 -cn 65 --trainedModelFile minValLossModel.pth --trainedAppendedModelFile minValLossAppendedModel_adtrained.pth --bayesianModelNum 5
    python main.py -classifier ExtendedBayesian --baseClassifier 3layerMLP  --routine gatherCorrectPrediction --dataset hdm05 -adTrainer EBMATrainer --trainFile classTrain.npz --testFile classTest.npz --dataPath ../data/ --retPath ../results/ -ep 200 -cn 65 --trainedModelFile minValLossModel.pth --trainedAppendedModelFile minValLossAppendedModel_adtrained.pth --bayesianModelNum 5
    ```

### Apologies

Due to the workload, the code is not constructed perfectly. Some code reading is probably needed before you can run the code. 

## Help

Regarding the parameters, just run: 

```
python main.py --help
```

## Authors

He Wang, Yunfeng Diao, Zichang Tan and Guodong Guo

He Wang, h.e.wang@leeds.ac.uk, [Personal website](https://drhewang.com)

Yunfeng Diao, diaoyunfeng@hfut.edu.cn, [Faculty page](http://faculty.hfut.edu.cn/diaoyunfeng/en/index.htm)

Project Webpage: http://drhewang.com/pages/AAHAR.html

## Version History
* 0.1
    * Initial Release
## Citation (Bibtex)
Please cite our papers on action recognition attacks/defense!
1. He Wang*, Yunfeng Diao*, Zichang Tan and Guodong Guo, Defending Black-box Skeleton-based Human Activity Classifiers, the AAAI conference on Aritificial Intelligence (AAAI) 2023.
 
    @InProceedings{Wang_Defending_2023,
    author={Wang, He and Diao, Yunfeng and Tan, Zichang and Guo, Guodong},
    booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
    title={Defending Black-box Skeleton-based Human Activity Classifiers},
    year={2023},
    month={June},
    }
2.  Yunfeng Diao*, He Wang*, Tianjia Shao, Yong-Liang Yang, Kun Zhou, David Hogg, Understanding the Vulnerability of Skeleton-based Human Activity Recognition via Black-box Attack, arxiv 2022.
  
    @misc{Diao_understanding_2022,
    url = {https://arxiv.org/abs/2211.11312},
    author = {Diao, Yunfeng and Wang, He and Shao, Tianjia and Yang, Yong-Liang and Zhou, Kun and Hogg, David},
    title = {Understanding the Vulnerability of Skeleton-based Human Activity Recognition via Black-box Attack},
    publisher = {arXiv},
    year = {2022}
    }

3. He Wang, Feixiang He, Zhexi Peng, Tianjia Shao, Yongliang Yang, Kun Zhou and David Hogg, Understanding the Robustness of Skeleton-based Action Recognition under Adversarial Attack, CVPR 2021

    @InProceedings{Wang_Understanding_2020,
    author={He Wang, Feixiang He, Zhexi Peng, Tianjia Shao, Yongliang Yang, Kun Zhou and David Hogg},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    title={Understanding the Robustness of Skeleton-based Action Recognition under Adversarial Attack},
    year={2021},
    month={June},
    }

4. Yunfeng Diao, Tianjia Shao, Yongliang Yang, Kun Zhou and He Wang, BASAR:Black-box Attack on Skeletal Action Recognition, CVPR 2021

    @InProceedings{Diao_Basar_2020,
    author={Yunfeng Diao, Tianjia Shao, Yongliang Yang, Kun Zhou and He Wang},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    title={BASAR:Black-box Attack on Skeletal Action Recognition},
    year={2021},
    month={June},
    }

## Acknowledgments
This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 899739 CrowdDNA, EPSRC (EP/R031193/1), NSF China (No. 61772462, No. U1736217), RCUK grant CAMERA (EP/M023281/1, EP/T014865/1) and the 100 Talents Program of Zhejiang University.

## License

Copyright (c) 2023, The University of Leeds, UK.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 
