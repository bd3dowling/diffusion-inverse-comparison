"""Conditioning method definitions and registry."""

from abc import ABC, abstractmethod
import torch
from strenum import StrEnum

__CONDITIONING_METHOD__ = {}


class ConditioningMethodName(StrEnum):
    VANILLA = "vanilla"
    PROJECTION = "projection"
    MANIFOLD_CONSTRAINT_GRADIENT = "mcg"
    POSTERIOR_SAMPLING = "ps"
    POSTERIOR_SAMPLING_PLUS = "ps+"
    TWEEDIE_MOMENT_PROJECTION = "tmp"


def register_conditioning_method(name: ConditioningMethodName):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls

    return wrapper


def get_conditioning_method(name: ConditioningMethodName, operator, noiser, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](operator=operator, noiser=noiser, **kwargs)


class ConditioningMethod(ABC):
    def __init__(self, operator, noiser, **kwargs):
        self.operator = operator
        self.noiser = noiser

    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)

    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
        if self.noiser.__name__ == "gaussian":
            difference = measurement - self.operator.forward(x_0_hat, **kwargs)
            norm = torch.linalg.norm(difference)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        elif self.noiser.__name__ == "poisson":
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement - Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        else:
            raise NotImplementedError

        return norm_grad, norm

    @abstractmethod
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass


@register_conditioning_method(name=ConditioningMethodName.VANILLA)
class Identity(ConditioningMethod):
    # just pass the input without conditioning
    def conditioning(self, x_t, **kwargs):
        return x_t, torch.Tensor([0])


@register_conditioning_method(name=ConditioningMethodName.PROJECTION)
class Projection(ConditioningMethod):
    def conditioning(self, x_t, noisy_measurement, **kwargs):
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement, **kwargs)
        return x_t, torch.Tensor([0])


@register_conditioning_method(name=ConditioningMethodName.MANIFOLD_CONSTRAINT_GRADIENT)
class ManifoldConstraintGradient(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get("scale", 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, noisy_measurement, **kwargs):
        # posterior sampling
        norm_grad, norm = self.grad_and_value(
            x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs
        )
        x_t -= norm_grad * self.scale

        # projection
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement, **kwargs)
        return x_t, norm


@register_conditioning_method(name=ConditioningMethodName.POSTERIOR_SAMPLING)
class PosteriorSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get("scale", 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm_grad, norm = self.grad_and_value(
            x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs
        )
        x_t -= norm_grad * self.scale
        return x_t, norm


@register_conditioning_method(name=ConditioningMethodName.POSTERIOR_SAMPLING_PLUS)
class PosteriorSamplingPlus(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.num_sampling = kwargs.get("num_sampling", 5)
        self.scale = kwargs.get("scale", 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm = 0
        for _ in range(self.num_sampling):
            # TODO: use noiser?
            x_0_hat_noise = x_0_hat + 0.05 * torch.rand_like(x_0_hat)
            difference = measurement - self.operator.forward(x_0_hat_noise)
            norm += torch.linalg.norm(difference) / self.num_sampling

        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        x_t -= norm_grad * self.scale
        return x_t, norm


@register_conditioning_method(name=ConditioningMethodName.TWEEDIE_MOMENT_PROJECTION)
class TweedieMomentProjection(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.num_sampling = kwargs.get('num_sampling', 5)

    def conditioning(self, x_t, measurement, estimate_x_0, ratio, v, noise_std, **kwargs):

        def estimate_h_x_0(x_t):
            x_0 = estimate_x_0(x_t)
            return self.operator.forward(x_0, **kwargs), x_0
    
        h_x_0, vjp = torch.autograd.functional.vjp(estimate_h_x_0, x_t, self.operator.forward(torch.ones_like(x_t), **kwargs))
        difference = measurement - h_x_0
        norm = torch.linalg.norm(difference)
        C_yy = self.operator.forward(vjp, **kwargs) + noise_std**2 / ratio
        _, ls = torch.autograd.functional.vjp(estimate_h_x_0, x_t, difference / C_yy)
        x_0 = estimate_x_0(x_t)

        # h_x_0, vjp_estimate_h_x_0, x_0 = torch.func.vjp(estimate_h_x_0, x_t, has_aux=True)
        # C_yy = self.operator.forward(vjp_estimate_h_x_0(self.operator.forward(torch.ones_like(x_0), **kwargs))[0], **kwargs) + noise_std**2 / ratio
        # difference = measurement - h_x_0
        # norm = torch.linalg.norm(difference)
        # ls = vjp_estimate_h_x_0(difference / C_yy)[0]

        x_0 = x_0 + ls

        return x_0, norm
