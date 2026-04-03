from typing import *
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from .base import Sampler
from .classifier_free_guidance_mixin import ClassifierFreeGuidanceSamplerMixin
from .guidance_interval_mixin import GuidanceIntervalSamplerMixin
import math
from ...modules import sparse as sp
import torch.nn.functional as F
from trellis.representations.mesh.point_mesh_dist import point_to_face_only, point_mesh_face_distance, mean_nn_distance
from .point_sdf_opt import sdf_zero_level_loss, trilinear_sdf_at_points, sdf_zero_level_l2, sample_sdf_with_idx_grid, sample_sdf_with_idx_grid_coarse_kernel
import kaolin


def _expand_t_like_x(t, x):
    if not torch.is_tensor(t):
        t = torch.tensor(t, dtype=x.dtype, device=x.device)
    # 允许 t 是标量或 [B]，统一成 [B,1,1,...]
    if t.ndim == 0:
        t = t.view(1)
    if t.ndim == 1:  # [B]
        while t.ndim < x.ndim:
            t = t.view(t.shape[0], *([1] * (x.ndim - 1)))
    elif t.ndim < x.ndim:
        # 已经是 [B,1,1,...] 的话会在这里走完
        while t.ndim < x.ndim:
            t = t.view(*t.shape, 1)
    return t


class FlowEulerSampler(Sampler):
    """
    Generate samples from a flow-matching model using Euler sampling.

    Args:
        sigma_min: The minimum scale of noise in flow.
    """
    def __init__(
        self,
        sigma_min: float,
    ):
        self.sigma_min = sigma_min

    def _eps_to_xstart(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * eps) / (1 - t)

    def _xstart_to_eps(self, x_t, t, x_0):
        assert x_t.shape == x_0.shape
        return (x_t - (1 - t) * x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _v_to_xstart_eps(self, x_t, t, v):
        assert x_t.shape == v.shape
        eps = (1 - t) * v + x_t
        x_0 = (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * v
        return x_0, eps

    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        # print("1111!!!!!")
        t = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        
        if cond is not None and cond.shape[0] == 1 and x_t.shape[0] > 1:
            cond = cond.repeat(x_t.shape[0], *([1] * (len(cond.shape) - 1)))
        return model(x_t, t, cond, **kwargs)

    def _get_model_prediction(self, model, x_t, t, cond=None, **kwargs):

        pred_v = self._inference_model(model, x_t, t, cond, **kwargs)
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        return pred_x_0, pred_eps, pred_v

    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ):
        """
        Sample x_{t-1} from the model using Euler method.
        
        Args:
            model: The model to sample from.
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            cond: conditional information.
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
        """
        pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
        pred_x_prev = x_t - (t - t_prev) * pred_v
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})



    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        # print(aa)
        for t, t_prev in tqdm(t_pairs, desc="Sampling", disable=not verbose):
            out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
            sample = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)
        ret.samples = sample
        # print(sample)
        return ret



    def q_sample(
        self, x_start: torch.Tensor, t: float, noise: Optional[torch.Tensor] = None
    ):
        """
        Forward diffusion process (q sampling) at time t.

        Args:
            x_start: Original clean image [N x C x ...].
            t: Current timestep.
            noise: Optional pre-generated noise. If None, will generate new noise.
        """
        if noise is None:
            if isinstance(x_start, sp.SparseTensor):
                noise = sp.SparseTensor(
                    coords=x_start.coords, feats=torch.randn_like(x_start.feats)
                )
                print(
                    f"[{t}] sampled noise std: {noise.feats.std()} {noise.feats.min()} {noise.feats.max()}"
                )
            else:
                noise = torch.randn_like(x_start)

        # Scale noise according to timestep
        scaled_noise = (self.sigma_min + (1 - self.sigma_min) * t) * noise
        
        return (1 - t) * x_start + scaled_noise, noise

    @torch.no_grad()
    def sample_lascomp(self,
        flow_model,
        decoder,
        encoder,
        noise,
        mask,
        partial_voxel,
        alpha_eta,
        optimization_step,
        lr_sample,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        **kwargs):
        sample = noise
        voxel_mask = partial_voxel.bool()
        mask_pos = (partial_voxel == 1).float()

        t_seq = np.linspace(1.0, 0.0, int(steps + 1), dtype=np.float32)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = [(t_seq[i], t_seq[i + 1]) for i in range(len(t_seq) - 1)]
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})

        for t, t_prev in tqdm(t_pairs, desc="Sampling", disable=not verbose):
            t      = _expand_t_like_x(t, sample)
            t_prev = _expand_t_like_x(t_prev, sample)
            pred_x_0, pred_eps, pred_v = self._get_model_prediction(flow_model, sample, t, cond, **kwargs)
         
            if t > 0.5:
               
                x_0 = sample -  t * pred_v


                x_1 = sample + (1-t) * pred_v
                eta_t = max(0.0, min(1.0, alpha_eta*t**1.0))
                eps = torch.randn_like(sample)
                x1_tilde = (1.0 - eta_t) ** 0.5 * x_1 + ( eta_t )** 0.5 * eps 
                x1_tilde = x1_tilde * mask + (1-mask) * torch.randn_like(sample)             
            

                pred_voxel = (decoder(x_0) > 0.0).float()
                
                

                print("time is {}, voxel num before insert gt:{}".format(t, torch.sum(pred_voxel)))
                pred_voxel[voxel_mask] = 1.0

                

                print("time is {}, voxel num after insert gt:{}".format(t, torch.sum(pred_voxel)))
                x_0_new = encoder(pred_voxel, sample_posterior=False)        
                
                if optimization_step==0:
                    sample = (1 - t) * x_0_new + t * x1_tilde
                    pred_x_0, pred_eps, pred_v = self._get_model_prediction(flow_model, sample, t, cond, **kwargs)
                    sample = sample - (t-t_prev)*pred_v
                else:
                    sample = (1 - t) * x_0_new + t * x1_tilde
 
            else:
                x_0 = sample -  t * pred_v
                x_1 = sample + (1-t) * pred_v

                pred_voxel = (decoder(x_0) > 0.0).float()
                print("time is {}, voxel num before insert gt:{}".format(t, torch.sum(pred_voxel)))
                pred_voxel[voxel_mask] = 1.0
                print("time is {}, voxel num after insert gt:{}".format(t, torch.sum(pred_voxel)))
                
                x_0_new = encoder(pred_voxel, sample_posterior=False)

                if optimization_step==0:
                    sample = (1 - t) * x_0_new + t * x_1
                    pred_x_0, pred_eps, pred_v = self._get_model_prediction(flow_model, sample, t, cond, **kwargs)
                    sample = sample - (t-t_prev)*pred_v
                else:
                    sample = (1 - t) * x_0_new + t * x_1

            if optimization_step != 0:
                pred_x_0, pred_eps, pred_v = self._get_model_prediction(flow_model, sample, t, cond, **kwargs)

                with torch.enable_grad():
                        
                    sample = torch.autograd.Variable(sample.detach(), requires_grad=True)

                    optimizer = torch.optim.Adam([
                        {"params": [sample], "lr": lr_sample}
                    ])


                    for _ in range(optimization_step):                   
                        x_0 = sample -  t * pred_v.detach()
                    
                        logits = decoder(x_0)
                        bce_all  = F.binary_cross_entropy_with_logits(
                            logits, partial_voxel.float(), reduction='none'
                        )
                        loss = (bce_all * mask_pos).sum() / mask_pos.sum()
                        print("loss == {}".format(loss))                    
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                pred_x_0, pred_eps, pred_v = self._get_model_prediction(flow_model, sample, t, cond, **kwargs)
                sample = sample - (t - t_prev)*pred_v

                
        ret.samples = sample
        return ret
    

    
    @torch.no_grad()
    def sample_inverse(self,
        model,
        gt,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        **kwargs):


        sample = gt

        t_seq = np.linspace(1, 0, steps + 1)
        
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)

        t_seq = t_seq[::-1]
        
        t_pairs = [(t_seq[i], t_seq[i+1]) for i in range(len(t_seq)-1)]
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})

        ret.pred_x_t.append(sample)

        for t, t_prev in tqdm(t_pairs, desc="Inverse", disable= not verbose):

        
            pred_x_0, pred_eps, v1 = self._get_model_prediction(model, sample, t, cond, **kwargs)
          

            x_euler = sample - (t - t_prev) * v1 
            _, _, v2 = self._get_model_prediction(model, x_euler, t_prev, cond, **kwargs)

            sample = sample - 0.5 * (t - t_prev) * (v1 + v2)

            print(
                f"After sample_once [t: {t},",
                sample.min(),
                sample.max(),
                sample.std()
            )
            
            ret.pred_x_t.append(sample)

        return ret
        



        


class FlowEulerRepaintSampler(FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with RePaint inpainting strategy.
    This implementation follows the RePaint paper (https://arxiv.org/abs/2201.09865).

    Args:
        sigma_min: The minimum scale of noise in flow.
    """

    def __init__(
        self,
        sigma_min: float,
    ):
        super().__init__(sigma_min=sigma_min)

    def q_sample(
        self, x_start: torch.Tensor, t: float, noise: Optional[torch.Tensor] = None
    ):
        """
        Forward diffusion process (q sampling) at time t.

        Args:
            x_start: Original clean image [N x C x ...].
            t: Current timestep.
            noise: Optional pre-generated noise. If None, will generate new noise.
        """
        if noise is None:
            if isinstance(x_start, sp.SparseTensor):
                noise = sp.SparseTensor(
                    coords=x_start.coords, feats=torch.randn_like(x_start.feats)
                )
                print(
                    f"[{t}] sampled noise std: {noise.feats.std()} {noise.feats.min()} {noise.feats.max()}"
                )
            else:
                noise = torch.randn_like(x_start)

        # Scale noise according to timestep
        scaled_noise = (self.sigma_min + (1 - self.sigma_min) * t) * noise
        # return x_start + scaled_noise, noise
        # return (1 - (1 - self.sigma_min) * t) * x_start + scaled_noise, noise
        return (1 - t) * x_start + scaled_noise, noise

    def q_sample_from_to(
        self, x_t: torch.Tensor, t: float, t_next: float, resample_method: int = 1
    ):
        """
        Sample from x_t at time t to x_{t_next} at time t_next (t < t_next) using Rectified Flow scheme.
        Used for resampling between timesteps.
        Intrinsically it's a forward diffusion process
        """
        # For rectified flow, we scale both the signal and noise
        if isinstance(x_t, sp.SparseTensor):
            noise = sp.SparseTensor(
                coords=x_t.coords, feats=torch.randn_like(x_t.feats)
            )
        else:
            noise = torch.randn_like(x_t)

        if resample_method == 1:
            # or we can term x_t as x_0
            return self.q_sample(x_t, t_next - t)[0]
        elif resample_method == 3:
            # formula: x_0 * (1 - t) + (self.sigma_min + (1 - self.sigma_min) * t) * eps = x_t
            # this way, x_0 = (x_{t-1} - (self.sigma_min + (1 - self.sigma_min) * t_next) * eps) / (1 - t_next)
            # x_t = x_0 * (1 - t) + (self.sigma_min + (1 - self.sigma_min) * t) * eps
            #     = (1 - t) / (1 - t_next) * x_{t-1} - (1 - t) / (1 - t_next) * (self.sigma_min + (1 - self.sigma_min) * t_next) * eps) + (self.sigma_min + (1 - self.sigma_min) * t) * eps

            x_next = (
                (1 - t_next) / (1 - t) * x_t
                - (1 - t_next)
                / (1 - t)
                * (self.sigma_min + (1 - self.sigma_min) * t)
                * noise
                + (self.sigma_min + (1 - self.sigma_min) * t_next) * noise
            )
            return x_next

        # Scale signal according to time ratio
        # x_next = t_next / t * x_t
        # Add scaled noise
        # noise_scale = torch.sqrt(t_next**2 - (t_next/t)**2 * t**2)
        # x_next = x_next + noise_scale * noise

        # For transition t -> t_next:
        # 1. Scale signal by ratio of signal coefficients at t_next vs t
        signal_scale = (1 - t_next) / (1 - t)

        # 2. Scale noise considering both:
        #    - The transition ratio of noise coefficients
        #    - Maintaining sum of coefficients = 1
        noise_scale = (self.sigma_min + (1 - self.sigma_min) * t_next) * (
            1 - signal_scale
        )

        x_next = signal_scale * x_t + noise_scale * noise
        return x_next

    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        mask: torch.Tensor,
        cond: Optional[Any] = None,
        **kwargs,
    ):
        """
        Sample x_{t-1} from the model using RePaint-modified Euler method.

        Args:
            model: The model to sample from.
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            mask: Binary mask indicating regions to inpaint [N x 1 x ...].
            cond: conditional information.
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
        """
        # Get model predictions
        pred_x_0, pred_eps, pred_v = self._get_model_prediction(
            model, x_t, t, cond, **kwargs
        )

        if isinstance(x_t, sp.SparseTensor):
            print(
                f"[t: {t}, t_prev: {t_prev}] predv feats: {pred_v.feats.std()} {pred_v.feats.min()} {pred_v.feats.max()}"
            )

        # Standard Euler step
        pred_x_prev = x_t - (t - t_prev) * pred_v

        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

    @torch.no_grad()
    def sample(
        self,
        model,
        noise: torch.Tensor,
        mask: torch.Tensor,
        known_x0: Union[torch.Tensor, sp.SparseTensor],
        cond: Optional[Any] = None,
        resample_times: int = 1,
        resample_method: int = 1,
        steps: int = 25,
        rescale_t: float = 3.0,
        verbose: bool = True,
        **kwargs,
    ):
        """
        Generate inpainted samples using RePaint algorithm.

        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            mask: Binary mask indicating regions to inpaint [N x 1 x ...].
            known_x0: Known regions of the image to preserve.
            resample_times: Number of resampling steps for each timestep.
            cond: conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the final inpainted samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        device = noise.device
        batch_size = noise.shape[0]

        # Initialize with known content if provided

        
        sample = noise * mask + known_x0 * (1 - mask)

        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})

        # Generate timestep sequence
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))

        # Sampling loop
        for t, t_next in tqdm(t_pairs, disable=not verbose):
            is_last_timestep = t_next == 0
            print(resample_method)

            # Multiple resampling steps at each timestep
            for r in reversed(range(resample_times)):
                is_last_resample_step = r == 0

                # Resample masked regions with noise
                # Add noise to known regions according to timestep
                # Q-sample step, sample x_{t-1}^{known} for known region
                noised_known, _ = (
                    self.q_sample(known_x0, t_next)
                    if not is_last_timestep
                    else (known_x0, 0)
                )

                # if isinstance(sample, sp.SparseTensor):
                #     print(
                #         "After q_sample and remix",
                #         sample.feats.min(),
                #         sample.feats.max(),
                #     )
                # else:
                #     print("After q_sample and remix", sample.min(), sample.max())

                # Single sampling step
                # Sample x_{t-1}^{unknown} from x_t
                # in this formulation, the known region will also be changed.
                step_output = self.sample_once(
                    model, sample, t, t_next, mask=mask, cond=cond, **kwargs
                )

                sample = step_output.pred_x_prev

                # Remix to get x_{t-1}
                sample = sample * mask + noised_known * (1 - mask)
                if isinstance(sample, sp.SparseTensor):
                    print(
                        f"After sample_once [t: {t}, r: {r}]",
                        sample.feats[mask.bool()[:, 0]].min(),
                        sample.feats[mask.bool()[:, 0]].max(),
                        sample.feats[~mask.bool()[:, 0]].min(),
                        sample.feats[~mask.bool()[:, 0]].max(),
                    )
                else:
                    print(
                        f"After sample_once [t: {t}, r: {r}]",
                        sample.min(),
                        sample.max(),
                    )

                # Forward diffusion from x_{t-1} to x_t
                # If not the last resample step and not the last timestep,
                # sample noise from t_next to t for next resample iteration
                if not (is_last_resample_step or is_last_timestep):
                    sample = self.q_sample_from_to(
                        sample, t_next, t, resample_method=resample_method
                    )
                    if isinstance(sample, sp.SparseTensor):
                        print(
                            f"After sample_from_to [t: {t}, r: {r}]",
                            sample.feats[mask.bool()[:, 0]].min(),
                            sample.feats[mask.bool()[:, 0]].max(),
                            sample.feats[~mask.bool()[:, 0]].min(),
                            sample.feats[~mask.bool()[:, 0]].max(),
                        )
                    else:
                        print(
                            f"After sample_from_to [t: {t}, r: {r}]",
                            sample.min(),
                            sample.max(),
                        )

                if is_last_timestep:
                    break

            # Store intermediate results
            ret.pred_x_t.append(sample)
            ret.pred_x_0.append(step_output.pred_x_0)

        # Final inpainting mask
        sample = sample * mask + known_x0 * (1 - mask)
        ret.samples = sample
        return ret


class FlowEulerCfgSampler(ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, **kwargs)




class FlowEulerGuidanceIntervalSampler(GuidanceIntervalSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance and interval.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            cfg_interval: The interval for classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, cfg_interval=cfg_interval, **kwargs)


class FlowEulerRepaintGuidanceIntervalSampler(
    GuidanceIntervalSamplerMixin, FlowEulerRepaintSampler
):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance and interval.
    """

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        mask,
        known_x0,
        cond,
        neg_cond,
        resample_times: int = 10,
        resample_method: int = 1,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs,
    ):
        """
        Generate samples from the model using Euler method.

        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            cfg_interval: The interval for classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(
            model,
            noise,
            mask,
            known_x0,
            cond,
            resample_times,
            resample_method,
            steps,
            rescale_t,
            verbose,
            neg_cond=neg_cond,
            cfg_strength=cfg_strength,
            cfg_interval=cfg_interval,
            **kwargs,
        )


def get_schedule_jump(t_T, n_sample, jump_length, jump_n_sample,
                      jump2_length=1, jump2_n_sample=1,
                      jump3_length=1, jump3_n_sample=1,
                      start_resampling=100000000):

    jumps = {}
    for j in range(0, t_T - jump_length, jump_length):
        jumps[j] = jump_n_sample - 1
    

    

    jumps2 = {}
    for j in range(0, t_T - jump2_length, jump2_length):
        jumps2[j] = jump2_n_sample - 1

    # print(jumps2)

    jumps3 = {}
    for j in range(0, t_T - jump3_length, jump3_length):
        jumps3[j] = jump3_n_sample - 1

    # print(jumps3)

    # print(aaa)

    t = t_T
    ts = []
    ts.append(t)

    while t >= 1:
        t = t-1
        ts.append(t)
        

        if (
            t + 1 < t_T - 1 and
            t <= start_resampling
        ):
            for _ in range(n_sample - 1):
                t = t + 1
                ts.append(t)

                if t >= 0:
                    t = t - 1
                    ts.append(t)

        if (
            jumps3.get(t, 0) > 0 and
            t <= start_resampling - jump3_length
        ):
            jumps3[t] = jumps3[t] - 1
            for _ in range(jump3_length):
                t = t + 1
                ts.append(t)

        if (
            jumps2.get(t, 0) > 0 and
            t <= start_resampling - jump2_length
        ):
            jumps2[t] = jumps2[t] - 1
            for _ in range(jump2_length):
                t = t + 1
                ts.append(t)
            jumps3 = {}
            for j in range(0, t_T - jump3_length, jump3_length):
                jumps3[j] = jump3_n_sample - 1

        if (
            jumps.get(t, 0) > 0 and
            t <= start_resampling - jump_length
        ):
            jumps[t] = jumps[t] - 1
            for _ in range(jump_length):
                t = t + 1
                ts.append(t)
            jumps2 = {}
            for j in range(0, t_T - jump2_length, jump2_length):
                jumps2[j] = jump2_n_sample - 1

            jumps3 = {}
            for j in range(0, t_T - jump3_length, jump3_length):
                jumps3[j] = jump3_n_sample - 1

    ts.append(-1)

    _check_times(ts, -1, t_T)

    return np.array(ts)/float(1000)

def _check_times(times, t_0, t_T):
    # Check end
    assert times[0] > times[1], (times[0], times[1])

    # Check beginning
    assert times[-1] == -1, times[-1]

    # Steplength = 1
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)

    # Value range
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= t_T, (t, t_T)


def make_repaint_times_stepbystep(start=1.0, end=0.0, step=0.02,
                       drop_before_jump=0.2, jump_back=0.1,
                       repeats=10, ndigits=6, stop_repeat_below=0.0):
    """
    生成时间序列：按 step 递减；每累计下降 drop_before_jump 后在同一位置回溯 jump_back，
    并重复 repeats 次；当 t < stop_repeat_below 时，不再 repeat，直接下降到 end。
    回跳（往更噪走）也按 step 逐步上行，而不是一次跳到目标。
    """
    if not (0 <= end < start <= 1.0):
        raise ValueError("require 0 <= end < start <= 1.0")
    if step <= 0:
        raise ValueError("step must be > 0")

    r = round
    eps = 10**(-ndigits)

    def add(tlist, t):
        t = r(float(t), ndigits)
        if not tlist or abs(tlist[-1] - t) > eps:
            tlist.append(t)

    times = []
    anchor = float(start)
    add(times, anchor)

    while anchor - drop_before_jump > end + eps:
        boundary = anchor - drop_before_jump  # 本段要下降到的阈值（更小 t）

        # 1) 从当前 t（一般等于 anchor）按 step 降到 boundary
        t = times[-1]
        while t - step > boundary + eps:
            t -= step
            add(times, t)
        add(times, boundary)   # 精准落到阈值
        t = boundary

        # --- 到达停止 repeat 的阈值？若是则直接下降到 end 并返回 ---
        if boundary <= stop_repeat_below + eps:
            while t - step > end + eps:
                t -= step
                add(times, t)
            add(times, end)
            return times

        # 2) 在该阈值处做 repeats 次回跳（上行和下行都按 step 走）
        for _ in range(repeats):
            t_jump = min(anchor, boundary + jump_back)  # 目标更噪的时刻（更大 t）

            # 2a) 先按 step 从 boundary -> t_jump（上行）
            while t + step < t_jump - eps:
                t += step
                add(times, t)
            add(times, t_jump)    # 精准到达 t_jump
            t = t_jump

            # 2b) 再按 step 从 t_jump -> boundary（下行）
            while t - step > boundary + eps:
                t -= step
                add(times, t)
            add(times, boundary)
            t = boundary

        # 3) 进入下一段（新的 anchor）
        anchor = boundary
        add(times, anchor)

    # 最后一段：从当前 anchor 直接到 end（无 repeat）
    t = times[-1]
    while t - step > end + eps:
        t -= step
        add(times, t)
    add(times, end)
    return times

def make_repaint_times(start=1, end=0.0, step=0.02,
                       drop_before_jump=2, jump_back=0.2,
                       repeats=5, ndigits=6, stop_repeat_below=0.):
    """
    生成时间序列：按 step 递减；每累计下降 drop_before_jump 后在同一位置回溯 jump_back，
    并重复 repeats 次；当 t < stop_repeat_below 时，不再 repeat，直接下降到 end。
    """
    if not (0 <= end < start <= 1.0):
        raise ValueError("require 0 <= end < start <= 1.0")
    if step <= 0:
        raise ValueError("step must be > 0")

    r = round
    eps = 10**(-ndigits)

    def add(tlist, t):
        t = r(t, ndigits)
        if not tlist or abs(tlist[-1] - t) > eps:
            tlist.append(t)

    times = []
    anchor = float(start)
    add(times, anchor)

    while anchor - drop_before_jump > end + eps:
        boundary = anchor - drop_before_jump  # 本段要下降到的阈值

        # 1) 先下降到 boundary
        t = times[-1]
        while t - step > boundary + eps:
            t = t - step
            add(times, t)
        add(times, boundary)  # 精准落到阈值

        # --- 触发“停止 repeat”条件：直接去 end 并返回 ---
        if boundary <= stop_repeat_below + eps:
            t = times[-1]
            while t - step > end + eps:
                t = t - step
                add(times, t)
            add(times, end)
            return times

        # 2) 否则在该阈值处做 repeats 次回跳
        for _ in range(repeats):
            t_jump = min(anchor, boundary + jump_back)  # 不超过 anchor
            add(times, t_jump)  # 回到更噪
            t = t_jump
            while t - step > boundary + eps:
                t = t - step
                add(times, t)
            add(times, boundary)  # 回到阈值

        # 3) 进入下一段
        anchor = boundary
        add(times, anchor)

    # 最后一段：从当前 anchor 直接到 end（无 repeat）
    t = times[-1]
    while t - step > end + eps:
        t = t - step
        add(times, t)
    add(times, end)
    return times

from torch import nn
FP16_MODULES = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.Linear,
    sp.SparseConv3d,
    sp.SparseInverseConv3d,
    sp.SparseLinear,
)
def convert_module_to_bf16(l):
    """
    Convert primitive modules to bfloat16, 类似 convert_module_to_f16()
    """
    if isinstance(l, FP16_MODULES):
        for p in l.parameters():
            p.data = p.data.bfloat16()

def force_module_dtype_(model: nn.Module, dtype: torch.dtype):
    # 先常规 .to()（大多数会跟上）
    model.to(dtype=dtype)
    # 再保险地逐一替换没跟上的 param/buffer
    for m in model.modules():
        for name, p in list(m._parameters.items()):
            if p is not None and p.dtype != dtype:
                m._parameters[name] = nn.Parameter(p.detach().to(dtype), requires_grad=p.requires_grad)
        for name, b in list(m._buffers.items()):
            if b is not None and b.dtype != dtype:
                m._buffers[name] = b.detach().to(dtype)
    return model


def main():
    t_seq = make_repaint_times()
    print(t_seq)

if __name__ == "__main__":
    main()
