from .NoiseSGD import NoiseSGD
from .NoiseAdam import NoiseAdam

# __all__ = ["NormalLearnRate", "LearningRateNoise", "NoiseSGD", "NoiseAdam"]
custom_imports = dict(imports=['mmcls.core.optimizer.NoiseSGD',
                               'mmcls.core.optimizer.NoiseAdam'], allow_failed_imports=False)