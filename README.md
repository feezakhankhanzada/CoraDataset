# CoraDataset
A GNN based model for Node classification

This repository demonstartes an implementation of Graph Neural Networks on Cora Dataset using KFold Cross Validation. 

Following are the steps to briefly describe the training and evaluation process:

1. Dataset is downloaded from https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz.

2. The dataset is split into random 10-fold Cross Validation set using StratifiedKFold.

3. Each Training, Validation, and Test set are divided into 2166, 270, 270 length of dataset after applying Cross Valdation.

4. train_mask, val_mask, test_mask are created to feed into the model.

5. We have used the Graph Neural Networks (GNN) for training the dataset for its simplicity and promising results.

6. The model consists of an input layer comprising of 1433 input features, 16 hidden layers and an output layer of seven output prediction of different weights.

7. We created a dictionery named history to store the performance of the model trained and evaluated.

8. The subset accuracy is calculated using accuracy_score function of scikit-learn.

9. The model is being trained for over 200 epoch for each fold providing that the current evaluation loss is less that the previous evaluation loss, the epoch will stop early otherwise.

10. After the 10th fold the history dictionery contains the final test accuracy which is around 88%.

11. The predicitions of 10th Fold Dataset with <paper_id> are stored in Tab Value Separated CSV file named "results.csv"

Following are the steps to run the program:

1. git clone https://github.com/feezakhankhanzada/CoraDataset.git

2. cd Project

3. sudo ./run.sh
