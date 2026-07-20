"""Stable public loss selectors.

Only ``Normal`` and ``StudentT`` are supported by the public training and
checkpoint contract. Other values accepted by the underlying implementation
are rejected by :func:`modeling_module.train` before data materialization.
"""

from modeling_module._internal.loss_runtime import DistributionLoss

__all__ = ["DistributionLoss"]
