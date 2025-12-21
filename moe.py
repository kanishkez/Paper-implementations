from typing import Text
import torch
import torch.nn as nn 
import torch.nn.functional as F 

class Expert(nn.Module):
    def __init__(self,d_model,d_ff):
        super().__init__()
        self.fc1=nn.Linear(d_model,d_ff)
        self.fc2=nn.Linear(d_ff,d_model)

    def forward(self,x):
        return self.fc2(F.gelu(self.fc1(x)))


class SwitchMOE(nn.Module):
    def __init__(self,d_model,d_ff,num_experts,c_factor=1.25, lb_loss_weight=1e-2):
        super().__init__()

        self.num_experts=num_experts
        self.c_factor= c_factor
        self.lb_loss_weight=lb_loss_weight
        self.router=nn.Linear(d_model,num_experts,bias=False)
        self.experts=nn.ModuleList([Expert(d_model,d_ff) for _ in range(num_experts)])

    def forward(self,x):

        B,T,D= x.shape
        tokens= B*T 
        x_flat=x.view(tokens,D)

        router_logits=self.router(x_flat)
        router_probabilities=F.softmax(router_logits,dim=-1)

        top1expert= torch.argmax(router_probabilities,dim=-1)
        top1_probability=router_probabilities.max(dim=-1).values

        expert_capacity=int(self.c_factor * tokens/self.num_experts)

        expert_counts= torch.zeros(self.num_experts,device=x.device)

        for i in range (self.num_experts):
            expert_counts[i]=(top1expert==i).sum()

        f=expert_counts/tokens
        p=router_probabilities.mean(dim=0)
        lb_loss=self.num_experts*torch.sum(f*p)

        output=torch.zeros_like(x_flat)

        for i,expert in enumerate(self.experts):
            idx=torch.where(top1expert==i)[0]

            if idx.numel()==0:
                continue

            idx=idx[::expert_capacity]

            expert_out=expert(x_flat[idx])

            output[idx]=expert_out * top1_probability[idx].unsqueeze(-1)

        output=output.view(B,T,D)
        total_lb_loss = lb_loss * self.lb_loss_weight
        
        return output,total_lb_loss


