from typing import *


class GuidanceIntervalSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance with interval.
    """

    def _inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs):
        # print("22222!!!!!")
        # print(cfg_strength)
        # print(cfg_interval)
        if cfg_interval[0] <= t <= cfg_interval[1]:
            # print("model type is {}".format(model.dtype  ))
            # print("x_t type is {}".format(x_t.dtype  ))
            # print("t type is {}".format(t.dtype  ))
            # print("cond type is {}".format(cond.dtype  ))
            pred = super()._inference_model(model, x_t, t, cond, **kwargs)
            neg_pred = super()._inference_model(model, x_t, t, neg_cond, **kwargs)
            return (1 + cfg_strength) * pred - cfg_strength * neg_pred
        else:
            return super()._inference_model(model, x_t, t, cond, **kwargs)
