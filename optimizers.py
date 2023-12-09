from torch.autograd.functional import hvp
from torch.optim import Optimizer
import torch
import numpy as np


##################
##  First Order ##
##################

class Adam(torch.optim.Adam):
    def __init__(self, *args, **kwargs):
        super(Adam, self).__init__(*args, **kwargs)

    def step(self, closure=None):
        return super(Adam, self).step(closure)


class SGD(torch.optim.SGD):
    def __init__(self, *args, **kwargs):
        super(SGD, self).__init__(*args, **kwargs)

    def step(self, closure=None):
        return super(SGD, self).step(closure)


class StormOptimizer(Optimizer):
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


class Adaptive_SGD(Optimizer):
    def __init__(self, params, lr=1e-1, f=lambda x: x, T=100):
        defaults = dict(lr=lr, f=f, T=T)
        super(Adaptive_SGD, self).__init__(params, defaults)
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

class HVP_RVR(Optimizer):
    def __init__(self, params, b=0.1,
                 sigma1=1, sigma2=1, l1=1, l2=1, eps=1e-2, lr=1e-1, mode='SGD'
                 , adaptive=False, T=100, func=lambda x: x):
        defaults = dict(b=b, sigma1=sigma1, sigma2=sigma2, l1=l1, l2=l2, eps=eps, lr=lr, mode=mode, adaptive=adaptive,
                        func=func, T=T)
        super(HVP_RVR, self).__init__(params, defaults)
        self.f = None
        self.g = None
        self.parameters = params
        self.b = b
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.l1 = l1
        self.l2 = l2
        self.eps = eps
        self.lr = lr
        self.mode = None

        if mode == 'SGD':
            self.mode = self.SGD
        elif mode == "SCRN":
            self.mode = self.SCRN

        self.adaptive = adaptive
        self.func = func
        self.T = T
        self.count = 0
        self.device = 'cuda'
        self.deltas = None

    def set_f(self, f):
        self.f = f

    def step(self, **kwargs):
        self.mode()

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

    def SCRN(self, **kwargs):
        self.parameters = [p for name in self.param_groups for p in name["params"]]
        if self.g is None or self.deltas is None:
            x_prev = None
        else:
            x_prev = [p.data - g for p, g in zip(self.parameters, self.deltas)]

        self.g = self.gradient_estimator(self.parameters, x_prev, self.g, self.b, self.sigma2, self.l2,
                                         self.eps)
        self.deltas = self.cubic_regularization(self.eps, self.g, self.l1, 5 * self.l2)
        for group in self.param_groups:
            for p, delta in zip(group["params"], self.deltas):
                p.data += delta

    def cubic_regularization(self, eps, grad, l_, rho, c_=1, T_eps=10):
        g_norm = [torch.norm(g) for g in grad]
        a = sum(g_norm)
        if a >= ((l_ ** 2) / rho):
            hgp = hvp(self.f, tuple(p for group in self.param_groups for p in group['params']),
                      tuple(p.grad for group in self.param_groups for p in group['params']))[1]
            temp = [g.reshape(-1) @ h.reshape(-1) / rho / a.pow(2) for g, h in zip(grad, hgp)]
            R_c = [(-t + torch.sqrt(t.pow(2) + 2 * a / rho)) for t in temp]
            delta = [-r * g / a for r, g in zip(R_c, grad)]
        else:
            delta = [torch.zeros(g.size()).to(self.device) for g in grad]
            sigma = c_ * (eps * rho) ** 0.5 / l_
            mu = 1.0 / (20.0 * l_)
            vec = [(torch.rand(g.size()) * 2 + torch.ones(g.size())).to(self.device) for g in grad]
            vec = [v / torch.norm(v) for v in vec]
            g_ = [g + sigma * v for g, v in zip(grad, vec)]
            for j in range(T_eps):
                hdp = hvp(self.f, tuple(p for group in self.param_groups for p in group['params']),
                          tuple(delta))[1]
                delta = [(d - mu * (g + h + rho / 2 * torch.norm(d) * d)) for g, d, h in zip(g_, delta, hdp)]
                # g_m = [(g + h + self.rho / 2 * torch.norm(d) * d) for g, d, h in zip(g_, delta2, hdp)]
                # d2_norm = [torch.norm(d) for d in g_m]
        return delta


class SCRN(Optimizer):
    def __init__(self, params, T_eps=10, l_=1,
                 rho=1, c_=1, eps=1e-2, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        defaults = dict(T_eps=T_eps, l_=l_, rho=rho, c_=c_, eps=eps)
        super(SCRN, self).__init__(params, defaults)
        self.hes = None
        self.T_eps = T_eps
        self.l_ = l_
        self.rho = rho
        self.c_ = c_
        self.eps = eps
        self.params = params
        self.f = None
        self.device = device
        self.log = []
        self.name = 'SCRN'

    def set_l(self, l, rho):
        self.l_ = l
        self.rho = rho

    def set_f(self, f):
        self.f = f

    def step(self, **kwargs):
        grad = [p.grad for group in self.param_groups for p in group['params']]
        deltas, delta_m = self.cubic_regularization(self.eps, grad)

        for group in self.param_groups:
            for p, delta in zip(group["params"], deltas):
                p.data += delta

        if delta_m >= (-1 / 100) * torch.sqrt(torch.pow(self.eps, 3) / self.rho):
            deltas = self.cubic_final(self.eps, grad)
            for group in self.param_groups:
                for p, delta in zip(group["params"], deltas):
                    p.data += delta
            return True

    # Algorithm 4 Cubic-Subsolver via Gradient Descent
    def cubic_regularization(self, eps, grad):
        g_norm = [torch.norm(g) for g in grad]
        a = sum(g_norm)
        if a >= ((self.l_ ** 2) / self.rho):
            # B[g]
            hgp = hvp(self.f, tuple(p for group in self.param_groups for p in group['params']),
                      tuple(p.grad for group in self.param_groups for p in group['params']))[1]
            # (gT B[g]) / (ρ||g||2)
            temp = [g.reshape(-1) @ h.reshape(-1) / self.rho / a.pow(2) for g, h in zip(grad, hgp)]

            # -temp + sqrt(temp^2 + 2 ||g_norm||/ρ)
            R_c = [(-t + torch.sqrt(t.pow(2) + 2 * a / self.rho)) for t in temp]

            # ∆ ← −Rc g/||g||
            delta = [-r * g / a for r, g in zip(R_c, grad)]
            self.log.append(('1', a.item(), sum([torch.norm(d) for d in delta]).item()))
        else:
            # ∆ ← 0, σ ← c sqrt(ρε)/l, mu ← 1/(20l)
            delta = [torch.zeros(g.size()).to(self.device) for g in grad]
            sigma = self.c_ * (eps * self.rho) ** 0.5 / self.l_
            mu = 1.0 / (20.0 * self.l_)
            # v ← random vector in R^d in uniform distribution
            vec = [(torch.rand(g.size()) * 2 + torch.ones(g.size())).to(self.device) for g in grad]
            vec = [v / torch.norm(v) for v in vec]
            # g_ ← g + σv
            g_ = [g + sigma * v for g, v in zip(grad, vec)]
            for _ in range(self.T_eps):
                # B[∆]
                hdp = hvp(self.f, tuple(p for group in self.param_groups for p in group['params']),
                          tuple(delta))[1]
                # ∆ ← ∆ − μ(g + B[∆] + ρ/2||∆||∆)
                delta = [(d - mu * (g + h + self.rho / 2 * torch.norm(d) * d)) for g, d, h in zip(g_, delta, hdp)]

        delta_m = grad * delta + 1 / 2 * delta.T * hdp + self.rho / 6 * torch.pow(torch.norm(delta), 3)

        self.log.append(('2', a.item(), sum([torch.norm(d) for d in delta]).item()))

        return delta, delta_m

    def cubic_final(self, eps, grad):
        # ∆ ← 0, g_m ← g, mu ← 1/(20l)
        delta = [torch.zeros(g.size()).to(self.device) for g in grad]
        grad_m = grad
        mu = 1.0 / (20.0 * self.l_)
        grad_norm = [torch.norm(g) for g in grad]

        while torch.sum(grad_norm) >= eps / 2:
            delta -= mu * grad_m

            hdp = hvp(self.f, tuple(p for group in self.param_groups for p in group['params']),
                      tuple(delta))[1]

            # g_m ← g + B[∆] + ρ/2||∆||
            grad_m = [(g + h + self.rho / 2 * torch.norm(d) * d) for g, d, h in zip(grad, delta, hdp)]
            grad_norm = [torch.norm(g) for g in grad_m]
        return delta

    def save_log(self, path='classifier_logs/classifier_logs/', flag_param=False):
        if flag_param:
            name = self.name + "_l_" + str(self.l) + "_rho_" + str(self.rho)
        else:
            name = self.name
        f = open(path + name, 'w')
        for l in self.log:
            f.write(str(l) + '\n')
        f.close()


class SCRN_Momentum(SCRN):
    def __init__(self, params, momentum=0.9, T_eps=10, l_=1, rho=1, c_=1, eps=1e-9):
        defaults = dict(T_eps=T_eps, l_=l_, rho=rho, c_=c_, eps=eps)
        super(SCRN_Momentum, self).__init__(params, defaults)
        self.old_delta = [torch.zeros(p.size()).to(self.device) for group in self.param_groups for p in group['params']]
        self.name = 'SCRN_Momentum'
        self.momentum = momentum
        self.old_delta = None
        self.training = True

    def step(self, **kwargs):
        if not self.training:
            return
        grad = [p.grad for group in self.param_groups for p in group['params']]
        deltas, delta_m = self.cubic_regularization(self.eps, grad)
        self.old_delta = [d1 * self.momentum + d2 for d1, d2 in zip(self.old_delta, deltas)]
        for group in self.param_groups:
            for p, delta in zip(group["params"], self.old_delta):
                p.data += delta

        if delta_m >= (-1 / 100) * torch.sqrt(torch.pow(self.eps, 3) / self.rho):
            deltas = self.cubic_final(self.eps, grad)
            for group in self.param_groups:
                for p, delta in zip(group["params"], deltas):
                    p.data += delta
            self.training = False


class SVRCRN(Optimizer):
    def __init__(self, params, T_eps=10, l_=1,
                 rho=1, c_=1, eps=1e-2, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        defaults = dict(T_eps=T_eps, l_=l_, rho=rho, c_=c_, eps=eps)
        super(SVRCRN, self).__init__(params, defaults)
        self.hes = None
        self.T_eps = T_eps
        self.l_ = l_
        self.rho = rho
        self.c_ = c_
        self.eps = eps
        self.params = params
        self.f = None
        self.device = device
        self.log = []
        self.name = 'SCRN'
        self.deltas = None
        self.vt = None

    def set_l(self, l, rho):
        self.l_ = l
        self.rho = rho

    def set_f(self, f):
        self.f = f

    def reset_vt(self):
        self.vt = None

    def step(self, **kwargs):
        if torch.rand(1) < 0.2:
            self.vt = None
        if self.vt is None:
            self.vt = [p.grad for group in self.param_groups for p in group['params']]
        else:
            # bug not working
            p_grad = [p.grad for group in self.param_groups for p in group['params']]
            old_param = [p - d for p, d in
                         zip([p for group in self.param_groups for p in group['params']], self.deltas)]

            old_grad = torch.autograd.grad(self.f(*old_param), old_param)
            self.log.append((sum([torch.norm(o) for o in old_grad]), sum([torch.norm(o) for o in p_grad]),
                             sum([torch.norm(o) for o in self.vt])))
            self.vt = [g1 - g2 + g3 for g1, g2, g3 in zip(p_grad, old_grad, self.vt)]
        grad = self.vt
        self.deltas = self.cubic_regularization(self.eps, grad)
        for group in self.param_groups:
            for p, delta in zip(group["params"], self.deltas):
                p.data += delta

    def cubic_regularization(self, eps, grad):
        g_norm = [torch.norm(g) for g in grad]
        a = sum(g_norm)
        if a >= ((self.l_ ** 2) / self.rho):
            hgp = hvp(self.f, tuple(p for group in self.param_groups for p in group['params']),
                      tuple(p.grad for group in self.param_groups for p in group['params']))[1]
            temp = [g.reshape(-1) @ h.reshape(-1) / self.rho / a.pow(2) for g, h in zip(grad, hgp)]
            R_c = [(-t + torch.sqrt(t.pow(2) + 2 * a / self.rho)) for t in temp]
            delta = [-r * g / a for r, g in zip(R_c, grad)]
            self.log.append(('1', a.item(), sum([torch.norm(d) for d in delta]).item()))
        else:
            delta = [torch.zeros(g.size()).to(self.device) for g in grad]
            sigma = self.c_ * (eps * self.rho) ** 0.5 / self.l_
            mu = 1.0 / (20.0 * self.l_)
            vec = [(torch.rand(g.size()) * 2 + torch.ones(g.size())).to(self.device) for g in grad]
            vec = [v / torch.norm(v) for v in vec]
            g_ = [g + sigma * v for g, v in zip(grad, vec)]
            for j in range(self.T_eps):
                hdp = hvp(self.f, tuple(p for group in self.param_groups for p in group['params']),
                          tuple(delta))[1]
                delta = [(d - mu * (g + h + self.rho / 2 * torch.norm(d) * d)) for g, d, h in zip(g_, delta, hdp)]
                # g_m = [(g + h + self.rho / 2 * torch.norm(d) * d) for g, d, h in zip(g_, delta2, hdp)]
                # d2_norm = [torch.norm(d) for d in g_m]
            self.log.append(('2', a.item(), sum([torch.norm(d) for d in delta]).item()))
        return delta

    def save_log(self, path='classifier_logs/classifier_logs/', flag_param=False):
        if flag_param:
            name = self.name + "_l_" + str(self.l) + "_rho_" + str(self.rho)
        else:
            name = self.name
        f = open(path + name, 'w')
        for l in self.log:
            f.write(str(l) + '\n')
        f.close()


class SVRC(Optimizer):
    def __init__(self, params, l_=1,
                 rho=100, fp=1e-1, T_eps=10, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        defaults = dict(l_=l_, rho=rho, fp=fp, T_eps=T_eps)
        super(SVRC, self).__init__(params, defaults)
        self.hes = None
        self.l_ = l_
        self.rho = rho
        self.eps = min(self.l_ / (4 * self.rho), self.l_ ** 2 / (4 * self.rho))
        self.fp = fp
        self.Mt = 4 * self.rho
        self.T = 25 * self.rho ** 0.5 / self.eps ** 1.5
        # self.T_eps = int(T_eps*self.l_/(self.Mt*np.sqrt(self.eps/self.rho)))
        self.T_eps = T_eps
        self.params = params
        self.f = None
        self.device = device
        self.log = []
        self.name = 'SVRC'
        self.vt = None
        self.deltas = None

    def set_f(self, f):
        self.f = f

    def reset_vt(self):
        self.vt = None

    def step(self, **kwargs):
        if self.vt is None:
            self.vt = [p.grad for group in self.param_groups for p in group['params']]
        else:
            # bug not working
            p_grad = [p.grad for group in self.param_groups for p in group['params']]
            old_param = [p - d for p, d in
                         zip([p for group in self.param_groups for p in group['params']], self.deltas)]

            old_grad = torch.autograd.grad(self.f(*old_param), old_param)
            self.log.append((sum([torch.norm(o) for o in old_grad]), sum([torch.norm(o) for o in p_grad]),
                             sum([torch.norm(o) for o in self.vt])))
            self.vt = [g1 - g2 + g3 for g1, g2, g3 in zip(p_grad, old_grad, self.vt)]
        self.deltas = self.cubic_regularization(self.vt, self.Mt, 1 / (16 * self.l_),
                                                np.sqrt(self.eps / self.rho), 0.5, self.fp / self.T / 3)
        for group in self.param_groups:
            for p, delta in zip(group["params"], self.deltas):
                p.data += delta

    def cubic_regularization(self, beta, tau, eta, zeta, eps, phi):
        n = sum([torch.norm(g) for g in beta])
        hgp = hvp(self.f, tuple(p for group in self.param_groups for p in group['params']),
                  tuple(beta))[1]
        temp = [g.reshape(-1) @ h.reshape(-1) / tau / n.pow(2) for g, h in zip(beta, hgp)]
        R_c = [(-t + torch.sqrt(t.pow(2) + 2 * n / tau)) for t in temp]
        x = [-r * g / n for r, g in zip(R_c, beta)]
        if self.cubic_function(beta, tau, x) <= -(1 - eps) * tau * (zeta ** 3) / 12:
            return x

        T_eps = self.T_eps
        sigma = (tau ** 2) * (zeta ** 3) * eps / (self.l_ + tau * zeta) / 576  # beta === rho?
        vec = [(torch.rand(g.size()) * 2 + torch.ones(g.size())).to(self.device) for g in beta]
        vec = [v / torch.norm(v) for v in vec]
        beta_ = [g + sigma * v for g, v in zip(beta, vec)]

        for i in range(T_eps):
            x = [a - eta * a1 for a, a1 in zip(x, self.cubic_grad(beta_, tau, x))]
            if self.cubic_function(beta_, tau, x) <= -(1 - eps) * tau * (zeta ** 3) / 12:
                return x
        return x

    def cubic_grad(self, beta, tau, x):
        hxp = hvp(self.f, tuple(p for group in self.param_groups for p in group['params']),
                  tuple(x))[1]
        nx = sum([torch.norm(a) for a in x])
        return [b + h + tau * nx * a / 2
                for a, b, h in zip(x, beta, hxp)]

    def cubic_function(self, beta, tau, x):
        hxp = hvp(self.f, tuple(p for group in self.param_groups for p in group['params']),
                  tuple(x))[1]
        nx = sum([torch.norm(a) for a in x])
        return sum([b.reshape(-1) @ a.reshape(-1) + a.reshape(-1) @ h.reshape(-1) / 2 + tau * (nx ** 3) / 6
                    for a, b, h in zip(x, beta, hxp)])

    def save_log(self, path='', flag_param=False, name=''):
        f = open(path + name, 'w')
        for l in self.log:
            f.write(str(l) + '\n')
        f.close()
