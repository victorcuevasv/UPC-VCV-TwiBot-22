from model import BotRGCN, BotGCN
from Dataset import Twibot22
import torch
from torch import nn
from utils import accuracy,init_weights

from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve,auc

from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data, HeteroData

import pandas as pd
from torch_geometric.explain import Explainer, GNNExplainer, CaptumExplainer
from torch_sparse import SparseTensor

device = 'cuda:0'
embedding_size,dropout,lr,weight_decay=32,0.1,1e-2,5e-2


root='./processed_data/'

dataset=Twibot22(root=root,device=device,process=False,save=False)
x,edge_index,edge_type,labels,train_idx,val_idx,test_idx=dataset.dataloader()
### edge_sparse = SparseTensor.from_edge_index(edge_index, edge_type, (x.shape[0], x.shape[0]))
#### Use only for BotGCN model
### edge_sparse=edge_sparse.to(torch.float)

### model=BotRGCN(cat_prop_size=3,embedding_dimension=embedding_size).to(device)
model=BotGCN(cat_prop_size=3,embedding_dimension=embedding_size).to(device)
loss=nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(),
                    lr=lr,weight_decay=weight_decay)


def train(epoch):
    model.train()
    output = model(x,edge_index,edge_type)
    ### output = model(x,edge_sparse)
    loss_train = loss(output[train_idx], labels[train_idx])
    acc_train = accuracy(output[train_idx], labels[train_idx])
    acc_val = accuracy(output[val_idx], labels[val_idx])
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    print('Epoch: {:04d}'.format(epoch+1),
        'loss_train: {:.4f}'.format(loss_train.item()),
        'acc_train: {:.4f}'.format(acc_train.item()),
        'acc_val: {:.4f}'.format(acc_val.item()),)
    return acc_train,loss_train

def test():
    model.eval()
    output = model(x,edge_index,edge_type)
    ### output = model(x,edge_sparse)
    loss_test = loss(output[test_idx], labels[test_idx])
    acc_test = accuracy(output[test_idx], labels[test_idx])
    output=output.max(1)[1].to('cpu').detach().numpy()
    label=labels.to('cpu').detach().numpy()
    f1=f1_score(label[test_idx],output[test_idx])
    #mcc=matthews_corrcoef(label[test_idx], output[test_idx])
    precision=precision_score(label[test_idx],output[test_idx])
    recall=recall_score(label[test_idx],output[test_idx])
    fpr, tpr, thresholds = roc_curve(label[test_idx], output[test_idx], pos_label=1)
    Auc=auc(fpr, tpr)
    print("Test set results:",
            "test_loss= {:.4f}".format(loss_test.item()),
            "test_accuracy= {:.4f}".format(acc_test.item()),
            ## "precision= {:.4f}".format(precision.item()),
            "precision= {:.4f}".format(precision),
            ## "recall= {:.4f}".format(recall.item()),
            "recall= {:.4f}".format(recall),
            ## "f1_score= {:.4f}".format(f1.item()),
            "f1_score= {:.4f}".format(f1),
            ##"mcc= {:.4f}".format(mcc.item()),
            "auc= {:.4f}".format(Auc.item()),
            )
    
model.apply(init_weights)

epochs=50
for epoch in range(epochs):
    train(epoch)
    
test()


explainer = Explainer(
    model=model,
    # algorithm=GNNExplainer(epochs=200),
    algorithm=GNNExplainer(epochs=50),
    # algorithm=CaptumExplainer('IntegratedGradients'),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='multiclass_classification',
        task_level='node',
        return_type='log_probs',
    ),
)
node_index = 10
explanation = explainer(x, edge_index, index=node_index, edge_type=edge_type)
### explanation = explainer(x, edge_sparse, index=node_index)
print(f'Generated explanations in {explanation.available_explanations}')

path = 'feature_importance.png'
explanation.visualize_feature_importance(path, top_k=10)
print(f"Feature importance plot has been saved to '{path}'")

path = 'subgraph.pdf'
explanation.visualize_graph(path)
print(f"Subgraph visualization plot has been saved to '{path}'")


