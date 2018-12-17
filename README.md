# EECS498 Project Repo
Anthony Liang, Julius Yeh, Shih-Heng Yu, Hui-Ching Chen, Hsuan-Kai Chang, Jang Wu, Hsiao-Jou Lin

# Instructions
Several steps are necessary to run this code.

1. Clone this repo using git 
2. Your repo should look like this:
    ```
    final_project_eecs498
      ├── README.md
      ├── checkpoints
      │   ├── inception_v3
      │   └── local_net
      │       └── local_net.param
      ├── config.json 
      ├── data
      │   ├── classes.csv
      │   ├── ssd_features.txt
      │   ├── xyz_train.txt
      │   └── xyz_valid.txt
      ├── dataset.py
      ├── local_net.py
      ├── predict.py
      ├── predict_inception.py
      ├── predict_local_net.py
      ├── preds
      │   ├── predictions.csv
      │   └── result.csv
      ├── starter.py
      ├── train_common.py
      ├── train_inception.py
      ├── train_local_net.py
      └── utils.py
    ```
3. Task 1: Image Classification
- To train network run: 

## Task 2: Localization
1. To train local-net run: `python3 train_local_net.py`. This will save weight params to `checkpoints/local_net/local_net.param`

2. To predict on test data: 
- Run SSD mxnet and generate 2D features on the test dataset.
- Save features in text file called `data/ssd_features.txt`
- Run `python3 predict_local_net.py` to produce `preds/predictions.csv`
