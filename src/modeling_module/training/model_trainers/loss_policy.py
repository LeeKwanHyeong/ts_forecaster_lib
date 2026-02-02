from modeling_module.training.config import TrainingConfig


def infer_loss_mode(train_cfg: TrainingConfig) -> str:
    """
    PatchTST train과 동일한 'auto/infer' 해석 규칙을 ExoTST에도 적용.
    - DistributionLoss 계열이면 dist
    - Quantile은 ExoTST에서 별도 head가 없으면 미지원(현재는 에러)
    - 그 외 point
    """
    loss_mode = str(getattr(train_cfg, "loss_mode", "auto")).lower()
    if loss_mode not in ("auto", "infer"):
        return loss_mode

    loss_obj = getattr(train_cfg, "loss", None)
    loss_name = getattr(loss_obj, "__class__", type("x", (object,), {})).__name__

    if (loss_name == "DistributionLoss") or bool(getattr(loss_obj, "is_distribution_output", False)):
        return "dist"
    if loss_name in ("MQLoss", "QuantileLoss"):
        return "quantile"
    return "point"