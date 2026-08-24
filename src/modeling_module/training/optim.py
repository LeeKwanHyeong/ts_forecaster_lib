import torch.optim as optim

def build_optimizer_and_scheduler(model, cfg):
    opt = optim.AdamW(model.parameters(), lr = cfg.lr, weight_decay = cfg.weight_decay)
    scheduler_name = str(getattr(cfg, "lr_scheduler", "cosine")).lower()
    if scheduler_name == "cosine":
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, cfg.t_max)
    elif scheduler_name == "constant":
        sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda _: 1.0)
    else:
        raise ValueError(
            f"Unsupported lr_scheduler={scheduler_name!r}; use 'cosine' or 'constant'."
        )
    return opt, sched
