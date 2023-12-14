from torch.autograd.functional import hvp
from torch.optim import Optimizer
import torch
import numpy as np
from torch.optim.optimizer import _use_grad_for_differentiable


##################
##  First Order ##
##################

class Adam(torch.optim.Adam):
    name = 'Adam'
    def __init__(self, *args, **kwargs):
        super(Adam, self).__init__(*args, **kwargs)

    def step(self, closure=None):
        return super(Adam, self).step(closure)

    def set_f(self, model, data, target, criterion):
        return


class SGD(torch.optim.SGD):
    name = 'SGD'
    def __init__(self, *args, **kwargs):
        super(SGD, self).__init__(*args, **kwargs)

    def step(self, closure=None):
        return super(SGD, self).step(closure)

    def set_f(self, model, data, target, criterion):
        return


class COptimizer(Optimizer):
    name = 'COptimizer'
    def __init__(self, *args, **kwargs):
        self.f = None
        self.has_f = False
        super(COptimizer, self).__init__(*args, **kwargs)

    def set_f(self, model, data, target, criterion):
        if self.has_f:
            names = list(n for n, _ in model.named_parameters())

            def f(*params):
                out: torch.Tensor = torch.func.functional_call(model, {n: p for n, p in zip(names, params)}, data)
                return criterion(out, target)

            self.f = f


class StormOptimizer(COptimizer):
    # Storing the parameters required in defaults dictionary
    # lr-->learning rate
    # c-->parameter to be swept over logarithmically spaced grid as per paper
    # w and k to be set as 0.1 as per paper
    # momentum-->dictionary storing model params as keys and their momentum term as values
    #            at each iteration(denoted by 'd' in paper)
    # gradient--> dictionary storing model params as keys and their gradients till now in a list as values
    #            (denoted by '∇f(x,ε)' in paper)
    # sqrgradnorm-->dictionary storing model params as keys and their sum of norm ofgradients till now
    #             as values(denoted by '∑G^2' in paper)

    def __init__(self, params, lr=0.1, c=100, momentum={}, gradient={}, sqrgradnorm={}):
        defaults = dict(lr=lr, c=c, momentum=momentum, sqrgradnorm=sqrgradnorm, gradient=gradient)
        super(StormOptimizer, self).__init__(params, defaults)

    # Returns the state of the optimizer as a dictionary containing state and param_groups as keys
    def __setstate__(self, state):
        super(StormOptimizer, self).__setstate__(state)

    # Performs a single optimization step for parameter updates
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        # param_groups-->a dict containing all parameter groups
        for group in self.param_groups:
            # Retrieving from defaults dictionary
            learn_rate = group['lr']
            factor = group['c']
            momentum = group['momentum']
            gradient = group['gradient']
            sqrgradnorm = group['sqrgradnorm']

            # Update step for each parameter present in param_groups
            for p in group['params']:
                # Calculating gradient('∇f(x,ε)' in paper)
                if p.grad is None:
                    continue
                dp = p.grad.data

                # Storing all gradients in a list
                if p in gradient:
                    gradient[p].append(dp)
                else:
                    gradient.update({p: [dp]})

                # Calculating and storing ∑G^2in sqrgradnorm
                if p in sqrgradnorm:
                    sqrgradnorm[p] = sqrgradnorm[p] + torch.pow(torch.norm(dp), 2)
                else:
                    sqrgradnorm.update({p: torch.pow(torch.norm(dp), 2)})

                # Updating learning rate('η' in paper)
                power = 1.0 / 3.0
                scaling = torch.pow((0.1 + sqrgradnorm[p]), power)
                learn_rate = learn_rate / (float)(scaling)

                # Calculating 'a' mentioned as a=cη^2 in paper(denoted 'c' as factor here)
                a = min(factor * learn_rate ** 2.0, 1.0)

                # Calculating and storing the momentum term(d'=∇f(x',ε')+(1-a')(d-∇f(x,ε')))
                if p in momentum:
                    momentum[p] = gradient[p][-1] + (1 - a) * (momentum[p] - gradient[p][-2])
                else:
                    momentum.update({p: dp})

                # Update of model parameter p
                p.data = p.data - learn_rate * momentum[p]
                learn_rate = group['lr']

        return loss


class Adaptive_SGD(COptimizer):
    def __init__(self, params, lr=1e-1, f=lambda x: x, T=100):
        defaults = dict(lr=lr, f=f, T=T)
        super(Adaptive_SGD, self).__init__(params, defaults)
        self.has_f = False
        self.f = f
        self.lr = lr
        self.T = T
        self.count = 0

    def step(self, **kwargs):
        lr = self.lr / self.f(1 + self.count // self.T)
        for group in self.param_groups:
            for p in group["params"]:
                p.data -= lr * p.grad
        self.count += 1


##################
## Second Order ##
##################

class HVP_RVR(COptimizer):
    def __init__(self, params, b=0.1,
                 sigma1=1, sigma2=1, l1=1, l2=1, eps=1e-2, lr=1e-1,
                 adaptive=False, T=100, func=lambda x: x):
        defaults = dict(b=b, sigma1=sigma1, sigma2=sigma2, l1=l1, l2=l2, eps=eps, lr=lr, adaptive=adaptive,
                        func=func, T=T)
        super(HVP_RVR, self).__init__(params, defaults)
        self.f = None
        self.has_f = True
        self.g = None
        self.parameters = params
        self.b = b
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.l1 = l1
        self.l2 = l2
        self.eps = eps
        self.lr = lr
        self.adaptive = adaptive
        self.func = func
        self.T = T
        self.count = 0
        self.device = 'cuda'
        self.deltas = None

    @torch.no_grad()
    def step(self, **kwargs):
        self.SGD()

    def gradient_estimator(self, x, x_prev, g_prev, b, sigma2, l2, eps):
        if torch.rand(1) < b or g_prev is None:
            return [p.grad for p in x]
        else:
            K = 5 * (sigma2 ** 2 + l2 * eps) / (b * eps ** 2) * sum(
                torch.norm(x1 - x2).pow(2) for x1, x2 in zip(x, x_prev))
            """print(K,sigma2,l2,eps,sum(
                torch.norm(x1 - x2).pow(2) for x1, x2 in zip(x, x_prev)),sum(
                torch.norm(x1 ).pow(2) for x1, x2 in zip(x, x_prev)),sum(
                torch.norm( x2).pow(2) for x1, x2 in zip(x, x_prev)))"""
            K = min(K, 10)
            b = x_prev
            g = g_prev
            for i in range(1, int(K)):
                a = [(i / K) * x1 + (1 - i / K) * x2 for x1, x2 in zip(x, x_prev)]
                h = hvp(self.f, tuple(b), tuple((x1 - x2) for x1, x2 in zip(a, b)))[1]
                g = [g1 + h1 for g1, h1 in zip(g, h)]
                b = a
            return g

    def SGD(self):
        if self.adaptive:
            lr = self.lr / self.func(1 + self.count // self.T)
        else:
            lr = self.lr
        self.parameters = [p for name in self.param_groups for p in name["params"]]
        if self.g is None:
            x_prev = None
        else:
            if self.adaptive:
                lr_prev = self.lr / self.func(1 + (self.count - 1) // self.T)
            else:
                lr_prev = self.lr
            x_prev = [p + lr_prev * g for p, g in zip(self.parameters, self.g)]

        self.g = self.gradient_estimator(self.parameters, x_prev, self.g, self.b, self.sigma2, self.l2,
                                         self.eps)

        for group in self.param_groups:
            for p, delta in zip(group["params"], self.g):
                p.data -= lr * delta
        self.count += 1


class SCRN(COptimizer):
    def __init__(self, params, T_out=2, T_eps=15, lr=0.05,
                 rho=1, c_=1, eps=1e-6):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        defaults = dict(T_out=T_out, T_eps=T_eps, lr=lr, rho=rho, c_=c_, eps=eps)
        super(SCRN, self).__init__(params, defaults)
        self.hes = None
        self.T_eps = T_eps
        self.T_out = T_out
        self.l_ = 1 / (20 * lr)
        self.lr = lr
        self.rho = rho
        self.c_ = c_
        self.eps = eps
        self.params = params
        self.f = None
        self.has_f = True
        self.log = []
        self.name = 'SCRN'
        self.mask = None
        self.val = (-1 / 100) * torch.sqrt(torch.tensor(self.eps ** 3 / self.rho)).to(self.device)
    def check_delta_m(self, param, delta_ms, grad, data):
        torch._foreach_sub_(delta_ms, self.val)
        delta_ms = [torch.tensor(1 if t > 0.5 else 0, dtype=torch.int8).to(self.device) for t in delta_ms]
        delta_ms = torch._foreach_mul(self.mask, delta_ms)
        if any(delta_ms):
            deltas = self.cubic_final(param, self.eps, grad, delta_ms)
            torch._foreach_addcmul_(data, deltas, delta_ms)
            torch._foreach_neg_(delta_ms)
            torch._foreach_add_(self.mask, delta_ms)
    @torch.no_grad()
    def step(self, **kwargs):
        self.mask = [torch.tensor(1).to(self.device) for group in self.param_groups for _ in group['params']]
        self.l_ = 1 / (20 * self.lr)
        param = [p for group in self.param_groups for p in group['params']]
        data = [p.data for p in param]
        grad = [p.grad if p.grad is not None else torch.zeros(p.data.shape) for p in param]
        for iter in range(self.T_out):
            deltas, delta_ms = self.cubic_regularization(param, self.eps, grad)

            # for group in self.param_groups:
            #     for p, delta in zip(group["params"], deltas):
            #         if self.mask[cnt]:
            #             p.data += delta
            #         cnt += 1
            torch._foreach_mul_(deltas, self.mask)

            torch._foreach_add_(data, deltas)
            self.check_delta_m(param, delta_ms, grad, data)
            if all(m < 0.5 for m in self.mask):
                break

    # Algorithm 4 Cubic-Subsolver via Gradient Descent

    def cubic_final(self, param, eps, grad, delta_ms):
        # ∆ ← 0, g_m ← g, mu ← 1/(20l)
        delta = [torch.zeros(g.size()).to(self.device) for g in grad]
        # delta = torch._foreach_zero(grad)
        grad_m = torch._foreach_mul(grad, delta_ms)
        mu = 1.0 / (20.0 * self.l_)
        # a = torch._foreach_pow(grad_m, 2)
        # a = torch.stack(a).sum().sqrt().item()
        a = torch.max(torch.stack(torch._foreach_norm(grad_m)))

        while a >= eps / 2:
            # delta = [d - mu * g for g, d in zip(grad_m, delta)]
            torch._foreach_mul_(grad_m, -mu)
            torch._foreach_add_(delta, grad_m)
            grad_m = hvp(self.f, tuple(param), tuple(delta))[1]

            # g_m ← g + B[∆] + ρ/2||∆||
            # grad_m = [(g + h + self.rho / 2 * torch.norm(d) * d) for g, d, h in zip(grad, delta, hdp)]
            torch._foreach_add_(grad_m, grad)
            tmp = torch._foreach_norm(delta)
            torch._foreach_mul_(tmp, self.rho / 2)
            torch._foreach_addcmul_(grad_m, delta, tmp)
            torch._foreach_add_(grad_m, delta_ms)
            norms = torch._foreach_norm(grad_m)
            a = torch._foreach_sub(norms, eps / 2)
            torch._foreach_sign_(a)
            # torch._foreach_add_(a, 1)
            # torch._foreach_div_(a, 2)
            a = [torch.tensor(1 if t > 0.5 else 0, dtype=torch.int8).to(self.device) for t in a]
            torch._foreach_mul_(norms, a)
            a = torch.max(torch.stack(norms))

            # a = torch._foreach_pow(grad_m, 2)
            # a = torch.stack(a).sum().sqrt().item()
            # a = torch.max(torch.stack(torch._foreach_norm(grad_m)))

        return delta

    def cubic_regularization(self, param, eps, grad):
        # g_norm = [torch.norm(g) for g in grad]
        a = torch._foreach_norm(grad)
        a_mask = torch._foreach_sub(a, ((self.l_ ** 2) / self.rho))
        torch._foreach_neg_(a_mask)
        torch._foreach_sign_(a_mask)
        # torch._foreach_add_(a_mask, 1)
        # torch._foreach_div_(a_mask, 2)
        a_mask = [torch.tensor(1 if t > 0.5 else 0, dtype=torch.int8).to(self.device) for t in a_mask]
        m1 = any(a_mask)
        m2 = not all(a_mask)
        if m1:
            torch._foreach_add_(a, a_mask)

            # if a >= ((self.l_ ** 2) / self.rho):
                # B[g]
            hgp = hvp(self.f, tuple(param), tuple(grad))[1]
            # (gT B[g]) / (ρ||g||2)
            # temp = [g.reshape(-1) @ h.reshape(-1) / self.rho / (a ** 2) for g, h in zip(grad, hgp)]
            torch._foreach_mul_(hgp, grad)
            hgp = [t.sum() for t in hgp]
            a_pow = torch._foreach_pow(a, 2)
            # epsilon = 1e-10
            # torch._foreach_add(a_pow, epsilon)
            torch._foreach_mul_(a_pow, - self.rho)
            torch._foreach_div_(hgp, a_pow)
            # -temp + sqrt(temp^2 + 2 ||g_norm||/ρ)
            # R_c = [(-t + torch.sqrt(t.pow(2) + 2 * a / self.rho)) for t in temp]
            hgp_pow = torch._foreach_pow(hgp, 2)
            a_rho = torch._foreach_mul(a, 2 / self.rho)
            torch._foreach_add_(hgp_pow, a_rho)
            torch._foreach_sqrt_(hgp_pow)
            torch._foreach_add_(hgp, hgp_pow)
            # ∆ ← −Rc g/||g||
            # delta = [-r * g / a for r, g in zip(R_c, grad)]
            torch._foreach_div_(hgp, torch._foreach_neg(a))
            delta1 = torch._foreach_mul(grad, hgp)

        # ****************
        if m2:
            # ∆ ← 0, σ ← c sqrt(ρε)/l, mu ← 1/(20l)
            delta = [torch.zeros(g.size()).to(self.device) for g in grad]
            # delta = torch._foreach_zeros(grad)

            sigma = self.c_ * (eps * self.rho) ** 0.5 / self.l_
            mu = 1.0 / (20.0 * self.l_)
            # v ← random vector in R^d in uniform distribution
            vec = [torch.rand(g.size()).to(self.device) for g in grad]
            torch._foreach_pow_(vec, 2)
            torch._foreach_add_(vec, 1)
            torch._foreach_div_(vec, torch._foreach_norm(vec))
            # vec = [(torch.rand(g.size()) * 2 + torch.ones(g.size())).to(self.device) for g in grad]
            # vec = [v / torch.norm(v) for v in vec]
            # g_ ← g + σv
            # g_ = [g + sigma * v for g, v in zip(grad, vec)]
            torch._foreach_mul_(vec, sigma)
            torch._foreach_add_(vec, grad)
            for _ in range(self.T_eps):
                # B[∆]
                hdp = hvp(self.f, tuple(param), tuple(delta))[1]
                # ∆ ← ∆ − μ(g + B[∆] + ρ/2||∆||∆)
                # delta = [(d - mu * (g + h + self.rho / 2 * torch.norm(d) * d)) for g, d, h in zip(g_, delta, hdp)]
                torch._foreach_add_(hdp, vec)
                tmp = torch._foreach_norm(delta)
                torch._foreach_mul_(tmp, self.rho / 2)
                torch._foreach_addcmul_(hdp, delta, tmp)
                torch._foreach_mul_(hdp, -mu)
                torch._foreach_add_(delta, hdp)
        if m1 and m2:
            torch._foreach_mul_(delta, a_mask)
            torch._foreach_sub_(a_mask, 1)
            torch._foreach_neg_(a_mask)
            torch._foreach_mul_(delta1, a_mask)
            torch._foreach_add_(delta, delta1)
        elif m1:
            delta = delta1
        hdp = hvp(self.f, tuple(param), tuple(delta))[1]
        # delta_m = [torch.sum(g * d) + torch.sum(1 / 2 * d * h) + self.rho / 6 * torch.pow(torch.norm(d), 3) for g, d, h
        #            in zip(grad, delta, hdp)]
        torch._foreach_mul_(hdp, delta)
        delta_m = torch._foreach_mul(delta, grad)
        torch._foreach_add_(delta_m, hdp)
        delta_m = [t.sum() for t in delta_m]
        tmp = torch._foreach_norm(delta)
        torch._foreach_pow_(tmp, 3)
        torch._foreach_mul_(tmp, self.rho / 6)
        torch._foreach_add_(delta_m, tmp)

        return delta, delta_m


class SCRN_Momentum(SCRN):
    def __init__(self, params, T_out=3, momentum=0.9, T_eps=10, lr=0.05, rho=1, c_=1, eps=1e-9):
        super(SCRN_Momentum, self).__init__(params, T_out, T_eps, lr, rho, c_, eps)
        self.old_delta = [torch.zeros(p.size()).to(self.device) for group in self.param_groups for p in group['params']]
        self.name = 'SCRN_Momentum'
        self.momentum = momentum

    @torch.no_grad()
    def step(self, **kwargs):
        self.mask = [torch.tensor(1.0).to(self.device) for group in self.param_groups for _ in group['params']]
        self.l_ = 1 / (20 * self.lr)
        param = [p for group in self.param_groups for p in group['params']]
        grad = [p.grad if p.grad is not None else torch.zeros(p.data.shape) for p in param]
        data = [p.data for p in param]
        for iter in range(self.T_out):
            deltas, delta_ms = self.cubic_regularization(param, self.eps, grad)
            torch._foreach_mul_(self.old_delta, self.momentum)
            torch._foreach_add_(self.old_delta, deltas, self.mask)
            torch._foreach_addcmul_(data, self.old_delta, self.mask)

            torch._foreach_sub_(delta_ms, self.val)
            delta_ms = [torch.tensor(1 if t > 0.5 else 0, dtype=torch.int8).to(self.device) for t in delta_ms]
            delta_ms = torch._foreach_mul(self.mask, delta_ms)
            if any(delta_ms):
                deltas = self.cubic_final(param, self.eps, grad, delta_ms)
                torch._foreach_addcmul_(data, deltas, delta_ms)
                torch._foreach_neg_(delta_ms)
                torch._foreach_add_(self.mask, delta_ms)
            if all(m < 0.5 for m in self.mask):
                break

class LBFGS(torch.optim.LBFGS):
    name = 'LBFGS'
    def __init__(self, *args, **kwargs):
        super(LBFGS, self).__init__(*args, **kwargs)

    def step(self, closure=None):
        return super(LBFGS, self).step(closure)

    def set_f(self, model, data, target, criterion):
        return
