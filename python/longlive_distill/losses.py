import torch
import torch.nn.functional as F

def relation(x):
    x=F.normalize(x.float(),dim=-1);return x@x.transpose(-1,-2)
def distillation_loss(student,teacher,weights):
    output=F.mse_loss(student["noise_pred"].float(),teacher["noise_pred"].float())
    s,t=student["noise_pred"].float(),teacher["noise_pred"].float();temporal=F.mse_loss(s[:,:,1:]-s[:,:,:-1],t[:,:,1:]-t[:,:,:-1]) if s.shape[2]>1 else output.new_zeros(())
    feature=output.new_zeros(());pairs=min(len(student.get("hidden_states",[])),len(teacher.get("hidden_states",[])))
    if pairs:
        si=torch.linspace(0,len(student["hidden_states"])-1,pairs).long();ti=torch.linspace(0,len(teacher["hidden_states"])-1,pairs).long()
        feature=torch.stack([F.mse_loss(relation(student["hidden_states"][a]),relation(teacher["hidden_states"][b])) for a,b in zip(si,ti)]).mean()
    total=weights["output"]*output+weights["temporal"]*temporal+weights["feature"]*feature
    return total,{"output":output.item(),"temporal":temporal.item(),"feature":feature.item(),"total":total.item()}
