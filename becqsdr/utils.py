import torch

def decompose_tensor_torch(T):
    trace = T.trace()
    isotropic = (trace / 3.0) * torch.eye(3, device=T.device)
    diag = torch.diag(torch.diag(T))
    deviatoric_diag = diag - isotropic
    off_diag = T - diag
    return trace, deviatoric_diag, off_diag

def tensor_loss_torch(T_true, T_pred, alpha=1.0, beta=1.0, gamma=1.0):
    trace_true, dev_true, off_true = decompose_tensor_torch(T_true)
    trace_pred, dev_pred, off_pred = decompose_tensor_torch(T_pred)

    trace_loss = (trace_true - trace_pred)**2
    dev_loss = torch.mean((dev_true - dev_pred)**2)
    off_diag_loss = torch.mean((off_true - off_pred)**2)

    return alpha * trace_loss + beta * dev_loss + gamma * off_diag_loss
