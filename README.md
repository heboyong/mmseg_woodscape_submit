
## Code for Woodscape Fisheye Motion Segmentation Challenge



### 1. Data Prepare
The code for data preparation is in the WoodScape folder, please make sure you have downloaded and unzipped all the data from the official link.
*Then modify all the paths under the folder to match your own settings, and  run:*

##### (1)  Process the synthetic dataset

```shell
python  woodscape/get_syn.py
python  woodscape/get_syn_list.py
```
###### (2) Process the real dataset and get the file list for training and testing

```shell
python  woodscape/get_train_list.py
python  woodscape/get_test_list.py
```

Copy the file list in the data root dir, and make sure that the final data folder has the format as shown :

```
data
│
└───real
│    │  train.txt
│    │  val.txt 
│    └───motion_annotations
│    └───previous_images
│    └───rgb_images
│
└───syn
│    │  train_syn.txt
│    └───motion_annotations
│    └───previous_images
│    └───rgb_images
│   
└───test
│    │  test.txt
│    └───previous_images(test_set)
│    └───rgb_images(test_set)

```

### 2. Requirement  and Package
##### (1). Create a conda environment and activate it.
```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```
##### (2). Install PyTorch 1.7 or higher

```shell
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
```
#### (3). Install mmsegmentation and mmcv
 
 Follow the official tutorial to install mmseg and mmcv: https://github.com/open-mmlab/mmsegmentation/blob/v0.29.1/docs/en/get_started.md#installation
 
 **Please make sure the version of mmseg is 0.29.1.**

### 3. Training Process

Run :
```shell
bash train.sh
```
We first trained a weights file that was used to initialize subsequent models. Then iteratively trained the weights containing only real data three times and three times containing both real data and synthetic data, and obtained a total of six models.

### 4. Testing and Submit
```shell
python model_ensemble.py
```
The final output will be saved in the results folder.
