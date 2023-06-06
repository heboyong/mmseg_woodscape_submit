
## Code for Woodscape Fisheye Motion Segmentation Challenge



#### 1. Data Prepare
*The code for data preparation is in the WoodScape folder, please make sure you have downloaded and unzipped all the data from the official link.*
*Then modify all the paths under the folder to match your own settings, and  run:*

(1)  Process the synthetic dataset

```shell
python  woodscape/get_syn.py
python  woodscape/get_syn_list.py
```
(2) Process the real dataset and get the file list for training and testing

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

#### 2. Requirement  and Package


#### 3. Training Process


#### 3. Testing and Submit
