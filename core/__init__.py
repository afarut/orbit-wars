"""Ядро инференса Orbit Wars — фичи, тула intercept и утилиты запуска.

Публичная поверхность:
  * :func:`utils.build_mlp`                  — фабрика MLP.
  * :func:`utils.validate_launch`            — проверка прямой траектории запуска флота.
  * :mod:`features`                          — obs -> тензоры (+ метаданные для декода).
  * :func:`intercept.intercept_angle`        — упреждающий угол / ETA на движущуюся цель.

Сеть :class:`model.PolicyValueNet` живёт отдельным модулем верхнего уровня ``model``
(он импортирует ``core``), поэтому здесь НЕ ре-экспортируется.
"""

from .features import EncodedObs, FeatureConfig, encode
from .intercept import Target, fleet_speed, intercept_angle, predict_position
from .utils import LaunchPlan, build_mlp, validate_launch

__all__ = [
    "build_mlp",
    "LaunchPlan",
    "validate_launch",
    "encode",
    "EncodedObs",
    "FeatureConfig",
    "Target",
    "fleet_speed",
    "intercept_angle",
    "predict_position",
]
