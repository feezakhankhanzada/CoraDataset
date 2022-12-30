from Dataset import create_folds , create_graph
from Model import GCN
import torch
from typing import Callable, List, Optional, Tuple
from typing_extensions import Literal, TypedDict
import torch
from torch import Tensor
from train import train_step , eval_step
import numpy as np

LossFn = Callable[[Tensor, Tensor], Tensor]

class HistoryDict(TypedDict):
    loss: List[float]
    acc: List[float]
    val_loss: List[float]
    val_acc: List[float]
    preds_train : List[float]
    preds_val : List[float]
    preds_test : List[float]

def train(
    x , 
    edge_index , 
    label , 
    train_mask , 
    val_mask , 
    test_mask , 
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: LossFn = torch.nn.CrossEntropyLoss(),
    max_epochs: int = 200,
    early_stopping: int = 10,
    print_interval: int = 1,
    verbose: bool = True,
) -> HistoryDict:
    history = {"loss": [], "val_loss": [], "acc": [], "val_acc": [] , "preds_train" : [] , "preds_val" : [] , "test_loss" : [] , "test_acc" : [] , "preds_test" : []}
    for epoch in range(max_epochs):
        loss, acc , preds_train = train_step(model,  optimizer, loss_fn , x , edge_index , label , train_mask)
        val_loss, val_acc , preds_val = eval_step(model,  loss_fn, "val" , x , edge_index , label , val_mask , test_mask)
        history["loss"].append(loss)
        history["acc"].append(acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["preds_train"].append(preds_train)
        history["preds_val"].append(preds_val)
        
        if epoch > early_stopping and val_loss > np.mean(history["val_loss"][-(early_stopping + 1) : -1]):
            if verbose:
                print("\nEarly stopping...")

            break

        if verbose and epoch % print_interval == 0:
            print(f"\nEpoch: {epoch}\n----------")
            print(f"Train loss: {loss:.4f} | Train acc: {acc:.4f}")
            print(f"  Val loss: {val_loss:.4f} |   Val acc: {val_acc:.4f}")

    test_loss, test_acc , preds_test = eval_step(model, loss_fn, "test" , x , edge_index , label , val_mask , test_mask)
    
    history["test_loss"].append(test_loss)
    history["test_acc"].append(test_acc)
    history["preds_test"].append(preds_test)
    
    if verbose:
        print(f"\nEpoch: {epoch}\n----------")
        print(f"Train loss: {loss:.4f} | Train acc: {acc:.4f}")
        print(f"  Val loss: {val_loss:.4f} |   Val acc: {val_acc:.4f}")
        print(f" Test loss: {test_loss:.4f} |  Test acc: {test_acc:.4f}")

    return history

if __name__ == "__main__":
    MAX_EPOCHS = 200
    LEARNING_RATE = 0.01
    WEIGHT_DECAY = 5e-4

    x , edge_index , label , le , content = create_graph()

    model = GCN(x.shape[1] , len(label.unique()))
    model.double()
    print(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)

    nfolds = 10 #Defining the number of folds to split the data

    for i in range(nfolds):
        train_mask , val_mask , test_mask , train_ids , val_ids , test_ids = create_folds(i , nfolds)

        history = train(x , edge_index , label , train_mask , val_mask , test_mask , model , optimizer)

    print("The Accuracy of the model is estimated to be of " , history['test_acc'][-1] * 100 , "%")

    result = content.iloc[test_ids]

    result['class_label'] = list(le.inverse_transform(history['preds_test'][0]))

    result = result[['paper_id' , 'class_label']]

    result.to_csv('results.csv', sep ='\t')


    
