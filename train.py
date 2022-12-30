import torch
from torch import Tensor
from sklearn.metrics import accuracy_score

def accuracy(pred, target):
    return (pred == target).sum().item() / target.numel()

def train_step(    
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    loss_fn,
    x , 
    edge_index ,
    label , 
    train_mask
):
    model.train()
    optimizer.zero_grad()
    
    mask = train_mask
    
    #logits = model(train_dataset , train_edge)
    logits = model(x , edge_index)[mask]
    
    preds = logits.argmax(dim=1)
    
    y = label[mask]
    
    #loss = loss_fn(logits, label)
    
    loss = loss_fn(logits , y)
    
    #acc = accuracy(preds, train_label)
    
    acc = accuracy(preds , y)

    acc = accuracy_score(y , preds , normalize=True, sample_weight=None)
    
    loss.backward()
    optimizer.step()
    
    print("train" , preds.shape)
    return loss.item(), acc , preds


@torch.no_grad()
def eval_step(model: torch.nn.Module, 
loss_fn, 
stage , 
x ,
edge_index , 
label ,
val_mask ,
test_mask):

    model.eval()
    
    if stage == 'val':
        print("val")
        mask = val_mask
    elif stage == 'test':
        print("test")
        mask = test_mask
    logits = model(x, edge_index)[mask]
    preds = logits.argmax(dim=1)
    y = label[mask]
    loss = loss_fn(logits, y)

    acc = accuracy(preds, y)

    acc = accuracy_score(y , preds , normalize=True, sample_weight=None)

    print(preds.shape)
    return loss.item(), acc , preds 