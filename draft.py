# here I've tried to implement the BPTT from Pascanu, 2014

import torch


class RNN:
    def __init__(self, input_size, dim):
        self.input_size = input_size
        self.dim = dim

        self.W_hx = torch.ones(self.dim)  # dummy initialization
        self.W_hh = torch.ones(self.dim)
        self.W_hy = torch.ones(self.dim)
        self.b_h = torch.ones(self.dim)
        self.b_y = torch.ones(self.dim)

        self.x = torch.zeros(self.input_size)
        self.h = torch.zeros(self.input_size)  # same length as the input
        self.y = torch.zeros(self.input_size)
        self.loss = torch.zeros(self.input_size)

    def forward(self, x):
        self.x = x  # needed for derivatives
        T = self.input_size  # T = len(x)
        h_prev = torch.zeros()
        for t in range(T):
            self.h[t] = torch.sigmoid(self.W_hx * x[t] + self.W_hh * h_prev + self.b_h)
            self.y[t] = torch.sigmoid(self.W_hy * self.h + self.b_y)
            h_prev = self.h
        return self.y, self.h

    def backward(self, targets):
        self.loss = torch.square(self.y - targets)  # squares of error

        def dloss_dy(i):
            return 2 * (self.y[i] - targets[i])

        def dy_dWhy(i):
            return self.y[i] * (1 - self.y[i]) * self.h[i]

        def dloss_dWhy(i):
            return dloss_dy(i) * dy_dWhy(i)

        def dy_dh(i):
            return self.y[i] * (1 - self.y[i]) * self.W_hy

        def dloss_dh(i):
            return dloss_dy(i) * dy_dh(i)

        def idh_dWhx(i):  # immediate derivative
            return self.h[i] * (1 - self.h[i]) * self.h[i - 1]

        def idh_dWhh(i):
            return self.h[i] * (1 - self.h[i]) * self.x[i]

        def dh_dh(i):  # gradient of self.h[t+1] wrt self.h[t]
            return self.h[i + 1] * (1 - self.h[i + 1]) * self.W_hh

        T = self.input_size
        gW_hy = dloss_dWhy(T)
        gh = dloss_dh(T)
        gW_hx = gh * idh_dWhx(T)
        gW_hh = gh * idh_dWhh(T)

        for t in range(T - 1, 0, -1):
            gh = gh * dh_dh(t) + dloss_dh(t)
            gW_hy = gW_hy + dloss_dWhy(t)
            gW_hx = gW_hx + gh * idh_dWhx(t)
            gW_hh = gW_hh + gh * idh_dWhh(t)


class HORNN:
    def __init__(self, input_size, dim, order):
        self.input_size = input_size
        self.dim = dim
        self.order = order

        self.W_hx = torch.ones(self.dim)
        self.W_hh = [torch.ones(self.dim)] * order  # it's now a list of W_hh
        self.W_hy = torch.ones(self.dim)
        self.b_h = torch.ones(self.dim)
        self.b_y = torch.ones(self.dim)

        self.x = torch.zeros(self.input_size)
        self.h = torch.zeros(self.input_size)
        self.y = torch.zeros(self.input_size)
        self.loss = torch.zeros(self.input_size)

    def forward(self, x):
        self.x = x
        T = self.input_size
        h_prev = [torch.zeros()] * self.order  # it's now also a list of previous states
        for t in range(T):
            s = torch.zeros(self.input_size)
            for p in self.order:
                s += self.W_hh[p] * h_prev[p]
            self.h[t] = torch.sigmoid(self.W_hx * x[t] + s + self.b_h)
            self.y[t] = torch.sigmoid(self.W_hy * self.h + self.b_y)
            h_prev = h_prev[1:]  # keeping the last p states
            h_prev.append(self.h[t])
        return self.y, self.h

    def backward(self, targets):
        self.loss = torch.square(self.y - targets)

        def dloss_dy(i):
            return 2 * (self.y[i] - targets[i])

        def dy_dWhy(i):
            return self.y[i] * (1 - self.y[i]) * self.h[i]

        def dloss_dWhy(i):
            return dloss_dy(i) * dy_dWhy(i)

        def dy_dh(i):
            return self.y[i] * (1 - self.y[i]) * self.W_hy

        def dloss_dh(i):
            return dloss_dy(i) * dy_dh(i)

        def idh_dWhx(i):
            return self.h[i] * (1 - self.h[i]) * self.h[i - 1]

        def idh_dWhh(i):
            return self.h[i] * (1 - self.h[i]) * self.x[i]

        def dh_dh(i):  # gradient of self.h[t+1] wrt self.h[t]
            return self.h[i + 1] * (1 - self.h[i + 1]) * self.W_hh

        T = self.input_size
        gW_hy = dloss_dWhy(T)
        gh0 = dloss_dh(T)
        gh = [0] * (self.order - 1) + [gh0]  # pending zeros to array of gh's
        gs = [0] * (self.order - 1)  # array of s_k's from the progress report

        gW_hx = gh0 * idh_dWhx(T)
        gW_hh = gh0 * idh_dWhh(T)

        for t in range(T - 1, 0, -1):
            M = gh[0]
            for i in range(self.order - 1):
                M = M * gs[i] + gh[i + 1]

            s = dh_dh(t)
            gs = gs[1:]
            gs.append(s)

            l = dloss_dh(t)
            M = M * s + l
            gh = gh[1:]
            gh.append(M)

            r = idh_dWhh(t)
            gW_hh = gW_hh + M * r

            gW_hy = gW_hy + dloss_dWhy(t)
            gW_hx = gW_hx + M * idh_dWhx(t)

        # for now gW_hh is update for only W_hh[0]
        # soon will add for each W_hh[i]
