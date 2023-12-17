from torch.autograd.functional import vhp
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


##################
## Second Order ##
##################

class SCRN(COptimizer):
    def __init__(self, params, T_out=1, T_eps=10, lr=0.05,
                 rho=1, c_=1, eps=0.05, using_final=False):
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
        self.t = 1000
        self.using_final = using_final
        self.mask = [torch.tensor(1).to(self.device) for group in self.param_groups for _ in group['params']]

        self.val = ((-1 / 100) * torch.sqrt(torch.tensor(self.eps ** 3 / self.rho))).to(self.device)

    @torch.no_grad()
    def step(self, **kwargs):
        self.l_ = 1 / (20 * self.lr)
        param = [p for group in self.param_groups for p in group['params']]
        data = [p.data for p in param]
        grad = [p.grad if p.grad is not None else torch.zeros(p.data.size()).to(self.device) for p in param]
        for iter in range(self.T_out):
            deltas, delta_ms = self.cubic_regularization(param, self.eps, grad)
            if self.using_final:
                torch._foreach_sub_(delta_ms, self.val)
                delta_ms = [torch.tensor(1 if t > 0.5 else 0, dtype=torch.int8).to(self.device) for t in delta_ms]
                delta_ms = torch._foreach_mul(self.mask, delta_ms)
                if any(delta_ms):
                    deltas_f = self.cubic_final(param, self.eps, grad, delta_ms)
                    torch._foreach_addcmul_(data, deltas_f, delta_ms)
                    torch._foreach_mul_(deltas, self.mask)
                    torch._foreach_add_(data, deltas)
                    torch._foreach_neg_(delta_ms)
                    torch._foreach_add_(self.mask, delta_ms)
                else:
                    torch._foreach_mul_(deltas, self.mask)
                    torch._foreach_add_(data, deltas)

                if all(m < 0.5 for m in self.mask):
                    break
            else:
                torch._foreach_add_(data, deltas)

    def cubic_final(self, param, eps, grad, delta_ms):
        # ∆ ← 0, g_m ← g, mu ← 1/(20l)
        delta = [torch.zeros(g.size()).to(self.device) for g in grad]
        grad_m = [g.detach().clone() for g in grad]
        delta_ms = [d.detach().clone() for d in delta_ms]
        mu = 1.0 / (20.0 * self.l_)
        a = torch._foreach_norm(grad_m)
        torch._foreach_mul_(a, delta_ms)
        a = torch.max(torch.stack(a))
        t = self.t
        while t > 0 and a > eps / 2:
            t -= 1
            torch._foreach_mul_(grad_m, -mu)
            torch._foreach_addcmul_(delta, grad_m, delta_ms)
            grad_m = vhp(self.f, tuple(param), tuple(delta))[1]

            # g_m ← g + B[∆] + ρ/2||∆||∆
            torch._foreach_add_(grad_m, grad)
            tmp = torch._foreach_norm(delta)
            torch._foreach_mul_(tmp, self.rho / 2)
            torch._foreach_addcmul_(grad_m, delta, tmp)
            norms = torch._foreach_norm(grad_m)

            a = torch._foreach_sub(norms, eps / 2)
            torch._foreach_sign_(a)

            a = [torch.tensor(1 if t > 0.5 else 0, dtype=torch.int8).to(self.device) for t in a]
            torch._foreach_mul_(delta_ms, a)
            torch._foreach_mul_(norms, delta_ms)
            # a = torch.max(torch.stack(norms))
            a = torch.max(torch.stack(norms))

        return delta

    def cubic_regularization(self, param, eps, grad):
        a = torch._foreach_norm(grad)
        a_mask = torch._foreach_sub(a, ((self.l_ ** 2) / self.rho))
        torch._foreach_neg_(a_mask)
        torch._foreach_sign_(a_mask)

        a_mask = [torch.tensor(1 if t > 0.5 else 0, dtype=torch.int8).to(self.device) for t in a_mask]
        m1 = any(a_mask)
        m2 = not all(a_mask)
        if m1:
            torch._foreach_add_(a, a_mask)

            # B[g]
            hgp = vhp(self.f, tuple(param), tuple(grad))[1]
            # (gT B[g]) / (ρ||g||2)
            torch._foreach_mul_(hgp, grad)
            hgp = [t.sum() for t in hgp]
            a_pow = torch._foreach_pow(a, 2)

            torch._foreach_mul_(a_pow, - self.rho)
            torch._foreach_div_(hgp, a_pow)

            hgp_pow = torch._foreach_pow(hgp, 2)
            a_rho = torch._foreach_mul(a, 2 / self.rho)
            torch._foreach_add_(hgp_pow, a_rho)
            torch._foreach_sqrt_(hgp_pow)
            torch._foreach_add_(hgp, hgp_pow)
            # ∆ ← −Rc g/||g||
            torch._foreach_div_(hgp, torch._foreach_neg(a))
            delta1 = torch._foreach_mul(grad, hgp)

        # ****************
        if m2:
            # ∆ ← 0, σ ← c sqrt(ρε)/l, mu ← 1/(20l)
            delta = [torch.zeros(g.size()).to(self.device) for g in grad]

            sigma = self.c_ * (eps * self.rho) ** 0.5 / self.l_
            mu = 1.0 / (20.0 * self.l_)
            # v ← random vector in R^d in uniform distribution
            vec = [torch.rand(g.size()).to(self.device) for g in grad]
            torch._foreach_pow_(vec, 2)
            torch._foreach_add_(vec, 1)
            torch._foreach_div_(vec, torch._foreach_norm(vec))

            # g_ ← g + σv
            # g_ = [g + sigma * v for g, v in zip(grad, vec)]
            torch._foreach_mul_(vec, sigma)
            torch._foreach_add_(vec, grad)
            for _ in range(self.T_eps):
                # B[∆]
                hdp = vhp(self.f, tuple(param), tuple(delta))[1]
                # ∆ ← ∆ − μ(g + B[∆] + ρ/2||∆||∆)
                torch._foreach_mul_(hdp, mu)
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
        if self.using_final:
            hdp = vhp(self.f, tuple(param), tuple(delta))[1]
            torch._foreach_mul_(hdp, delta)
            delta_m = torch._foreach_mul(delta, grad)
            torch._foreach_add_(delta_m, hdp)
            delta_m = [t.sum() for t in delta_m]
            tmp = torch._foreach_norm(delta)
            torch._foreach_pow_(tmp, 3)
            torch._foreach_mul_(tmp, self.rho / 6)
            torch._foreach_add_(delta_m, tmp)
        else:
            delta_m = None
        return delta, delta_m


class SCRN_Momentum(SCRN):
    def __init__(self, params, T_out=1, momentum=0.9, T_eps=10, lr=0.05, rho=1, c_=1, eps=0.05):
        super(SCRN_Momentum, self).__init__(params, T_out, T_eps, lr, rho, c_, eps)
        self.old_delta = [torch.zeros(p.size()).to(self.device) for group in self.param_groups for p in group['params']]
        self.name = 'SCRN_Momentum'
        self.momentum = momentum
        self.mask = [torch.tensor(1).to(self.device) for group in self.param_groups for _ in group['params']]

    @torch.no_grad()
    def step(self, **kwargs):
        self.l_ = 1 / (20 * self.lr)
        param = [p for group in self.param_groups for p in group['params']]
        grad = [p.grad if p.grad is not None else torch.zeros(p.data.size()).to(self.device) for p in param]
        data = [p.data for p in param]
        for iter in range(self.T_out):
            deltas, delta_ms = self.cubic_regularization(param, self.eps, grad)
            if self.using_final:
                new_old_delta = torch._foreach_mul(self.old_delta, self.mask)

                torch._foreach_sub_(delta_ms, self.val)
                delta_ms = [torch.tensor(1 if t > 0.5 else 0, dtype=torch.int8).to(self.device) for t in delta_ms]
                delta_ms = torch._foreach_mul(self.mask, delta_ms)
                neg_mask_f = None
                if any(delta_ms):
                    deltas_f = self.cubic_final(param, self.eps, grad, delta_ms)

                    new_old_delta = torch._foreach_mul(self.old_delta, delta_ms)
                    torch._foreach_mul_(new_old_delta, self.momentum)
                    neg_mask_f = torch._foreach_neg(delta_ms)
                    torch._foreach_add_(neg_mask_f, 1)
                    torch._foreach_mul_(self.old_delta, neg_mask_f)
                    torch._foreach_add_(self.old_delta, new_old_delta)
                    torch._foreach_addcmul_(self.old_delta, deltas_f, delta_ms)
                    torch._foreach_addcmul_(data, self.old_delta, delta_ms)
                torch._foreach_mul_(new_old_delta, self.momentum)
                neg_mask = torch._foreach_neg(self.mask)
                torch._foreach_add_(neg_mask, 1)
                torch._foreach_mul_(self.old_delta, neg_mask)
                torch._foreach_add_(self.old_delta, new_old_delta)
                torch._foreach_addcmul_(self.old_delta, deltas, self.mask)
                torch._foreach_addcmul_(data, self.old_delta, self.mask)

                if neg_mask_f is not None:
                    torch._foreach_add_(self.mask, neg_mask_f)
                if all(m < 0.5 for m in self.mask):
                    break
            else:
                torch._foreach_mul_(self.old_delta, self.momentum)
                torch._foreach_add_(self.old_delta, deltas)
                torch._foreach_add_(data, self.old_delta)
