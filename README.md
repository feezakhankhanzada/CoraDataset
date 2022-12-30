# CoraDataset
A GNN based model for Node classification

Dataset downloaded from https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz.

The dataset is split into random 10-fold Cross Validation set using StratifiedKFold.

Each Training, Validation, and Test set are divided into 2166, 270, 270 length of dataset after applying Cross Valdation.

train_mask, val_mask, test_mask are created to feed into the model.

We have used the Graph Neural Networks (GNN) for training the dataset for its simplicity and promising results.

The model consists of an input layer comprising of 1433 input features, 16 hidden layers and an output layer of seven output prediction of different weights.

We created a dictionery named history to store the performance of the model trained and evaluated.

The subset accuracy is calculated using accuracy_score function of scikit-learn.

The model is being trained for over 200 epoch for each fold providing that the current evaluation loss is less that the previous evaluation loss, the epoch will stop early otherwise.

After the 10th fold the history dictionery contains the final test accuracy which is around 88%.

Following are the steps to run the program:

git clone feezakhankhanzada/CoraDataset
cd CoraDataset
sudo ./run.sh
