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
      │   ├── train_feats.txt
      │   └── valid_feats.txt
      ├── dataset.py
      ├── local_net.py
      ├── predict.py
      ├── predict_inception.py
      ├── predict_local_net.py
      ├── preds
      │   ├── pred_localization.csv
      │   └── pred_classification.csv
      ├── train_common.py
      ├── train_inception.py
      ├── train_local_net.py
      └── utils.py
    ```

## Task 1: Image Classification

*To use pretrained weights:*
- [Download epoch1 (100MB)](https://drive.google.com/open?id=1HAxcSTTQBy0LLZHBRVzqa2_dPeu0YXFQ)
- [Download epoch2 (100MB)](https://drive.google.com/open?id=19-0uVlXdnW1hJOB4vghzf6vxeKROHeaq)

1. To train network run: `python3 train_inception.py`. This will save weights to `checkpoints/inception_v3/`

2. To predict run: `python3 predict_inception.py` which will save the predictions in `preds/pred_classification.csv`

## Task 2: Localization
1. To train local-net run: `python3 train_local_net.py`. This will save weight params to `checkpoints/local_net/local_net.param`

*Already have pretrained weights, but training should be very quick* 

2. To predict on test data: 
- Run SSD mxnet and generate 2D features on the test dataset.
- Save features in text file called `data/ssd_features.txt`
- Run `python3 predict_local_net.py` to produce `preds/pred_localization.csv`


*Note: If training/prediction doesn't work because of CUDA memory error try decreasing batch size. Training on CPU is very very slow.*
