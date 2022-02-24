import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def ConfusionMat(x,y,ae,model,plot=True):
    with torch.no_grad():
        x, _ = ae(x)
        logits = model(x)
        pred = torch.argmax(F.log_softmax(logits,dim=1),dim=1)
        cm = confusion_matrix(y,pred)
    
    if plot:
        plt.figure(figsize=(15,9))
        ax = sns.heatmap(cm, annot=True, fmt='d', linewidths=.5, square=True)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
    return cm

ae = Arxiv()
cl = Classifier()
ae.load_state_dict(torch.load('saves/ae.pt'))
cl.load_state_dict(torch.load('saves/cl.pt'))
cm = ConfusionMat(x_test, y_test, ae, cl)