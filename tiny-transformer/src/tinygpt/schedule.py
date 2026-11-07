"""Learning-rate schedules: linear warmup followed by cosine decay."""
import math


def warmup_cosine(step, total_steps, base_lr, warmup=0, min_lr_ratio=0.1):
    """LR for 1-indexed `step`. Linear ramp to base_lr, then cosine to base_lr*min_lr_ratio."""
    if warmup > 0 and step <= warmup:
        return base_lr * step / warmup
    if total_steps <= warmup:
        return base_lr
    progress = min(1.0, (step - warmup) / (total_steps - warmup))
    floor = base_lr * min_lr_ratio
    return floor + 0.5 * (base_lr - floor) * (1 + math.cos(math.pi * progress))
