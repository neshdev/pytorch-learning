import torch
from torch.nn import Sequential, MaxPool2d, ReLU, Conv2d, Dropout2d, Linear

class NeshNetwork(torch.nn.Module):
    def __init__(self,input_dim:[int]=[1,28,28],num_classes:int=10):
        super(NeshNetwork,self).__init__()
        in_c0 = input_dim[0]
        width = input_dim[1]
        heigt = input_dim[2]

        kernel_size = (3,3)
        out_c0 = 27
        out_c1 = 27

        self.convs =  Sequential(
            Conv2d(in_c0,out_c0,kernel_size,padding=1),
            ReLU(),
            MaxPool2d(kernel_size,stride=3),
            Conv2d(out_c0,out_c1,kernel_size,padding=1),
            ReLU(),
            MaxPool2d(kernel_size,stride=3),
        )
        
        s = self.convs(torch.zeros([1] + input_dim))
        cols = s.shape[1:].numel()
        
        self.classifier = Sequential(
            Dropout2d(),
            Linear(cols,1024),
            ReLU(),
            Linear(1024,1024),
            ReLU(),
            Dropout2d(),
            Linear(1024,num_classes)    
        )
    def forward(self,X: torch.Tensor) -> torch.Tensor:
        self.train()
        s = self.convs(X)
        cols = s.shape[1:].numel()
        t = s.view(s.size(0),cols)
        preds = self.classifier(t)
        return preds
    
    def predict(net, X: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        with torch.no_grad():
            scores = net(X)
            return (torch.nn.Softmax()(scores),scores.argmax(dim=1,keepdim=True))
            
        