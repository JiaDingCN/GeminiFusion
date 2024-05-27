from torch import nn
from torch.optim import AdamW, SGD


def get_optimizer(
    model: nn.Module, optimizer: str, lr: float, weight_decay: float = 0.01
):
    wd_params, nowd_params, decoder_params = [], [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            if "decode_head" in n:
                decoder_params.append(p)
            else:
                if p.dim() == 1:
                    nowd_params.append(p)
                else:
                    wd_params.append(p)

    params = [
        {"params": wd_params},
        {"params": nowd_params, "weight_decay": 0},
        {"params": decoder_params, "lr": lr * 10},
    ]

    if optimizer == "adamw":
        return AdamW(
            params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay
        )
    else:
        return SGD(params, lr, momentum=0.9, weight_decay=weight_decay)
