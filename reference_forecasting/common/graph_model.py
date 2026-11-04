"""Target-only spatial adapter around the unchanged copied temporal architecture."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.linear_model import Ridge


class TargetGraphKAN(nn.Module):
    def __init__(self,backbone,widths,adjacency,config,device):
        super().__init__()
        self.widths=list(widths)
        self.offsets=np.cumsum([0]+self.widths).tolist()
        # Node 0 is always Brentwood. Only its output is supervised/predicted.
        target_width=self.widths[0]
        self.temporal=backbone(target_width,config,device)
        self.register_buffer("target_weights",torch.as_tensor(adjacency[0],dtype=torch.float32))
        self.projections=nn.ModuleList([nn.Identity()]+[nn.Linear(w,target_width) for w in self.widths[1:]])
        self.graph_transform=nn.Linear(target_width,target_width)
        self.norm=nn.LayerNorm(target_width)

    def forward(self,x):
        own=x[...,:self.widths[0]]
        projected=[projection(x[...,self.offsets[i]:self.offsets[i+1]])
                   for i,projection in enumerate(self.projections)]
        nodes=torch.stack(projected,dim=2)
        context=torch.einsum("n,btnd->btd",self.target_weights,nodes)
        enriched=self.norm(own+F.gelu(self.graph_transform(context)))
        return self.temporal(enriched)

    def forecast(self,x):
        return self(x.to(self.target_weights.device))


def target_ridge_class(total_width,target_width,window):
    """Keep Ridge strictly Brentwood-only even though neural input contains neighbor windows."""
    class TargetOnlyRidge(Ridge):
        def own_design(self,x):
            values=np.asarray(x)
            if values.ndim!=2 or values.shape[1]!=window*total_width:
                raise ValueError("Unexpected packed spatial Ridge design.")
            return values.reshape(len(values),window,total_width)[...,:target_width].reshape(len(values),-1)

        def fit(self,x,y,sample_weight=None):
            return super().fit(self.own_design(x),y,sample_weight=sample_weight)

        def predict(self,x):
            return super().predict(self.own_design(x))
    return TargetOnlyRidge
