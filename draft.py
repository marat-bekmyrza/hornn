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

        self.h = torch.zeros(self.input_size)  # same length as the input
        self.y = torch.zeros(self.input_size)

    def forward(self, x):
        T = self.input_size  # T = len(x)
        h_prev = torch.zeros()
        for t in range(T):
            self.h[t] = torch.sigmoid(self.W_hx * x[t] + self.W_hh * h_prev + self.b_h)
            self.y[t] = torch.sigmoid(self.W_hy * self.h + self.b_y)
            h_prev = self.h
        return self.y, self.h

    def backward(self):
        T = self.input_size
        gW_hy = grad(loss[T], W_hy)
        gh = grad(loss[T], self.h[T])
        gW_hx = gh * imgrad(self.h[T], W_hx)
        gW_hh = gh * imgrad(self.h[T], W_hh)

        for t in range(T-1, 0, -1):
            gh = gh * grad(self.h[t+1], self.h[t]) + grad(loss[t], self.h[t])
            gW_hy = gW_hy + grad(loss[t], W_hy)
            gW_hx = gW_hx + gh * imgrad(self.h[t], W_hx)
            gW_hh = gW_hh + gh * imgrad(self.h[t], W_hx)





print(torch.matmul(torch.tensor([2.0, 2.0]), torch.tensor([[1.0, 4.0], [3.0, 3.0]])))


"""
p = 3  # number of order, just an example

# At the beginning, initialisations:
l0 = 1  # suppose l0 = 1
M0 = l0
arrayOfM = [0]*(p-1) + [M0]
arrayOfs = [0]*(p-1)
r0 = 1  # suppose r0 = 1
gW = M0 * r0

T = 6  # number of time steps, just an example
# At each time step (steps are from T-1 to 1, or k from 1 to T-1):
for k in range(1, T-1):
   # here we find current M
   #########################
   M = arrayOfM[0]
   for i in range(p-1):
      M = M * arrayOfs[i] + arrayOfM[i+1]
   s = 1  # suppose s_(k-1,k) = 1
   arrayOfs = arrayOfs[1:]
   arrayOfs.append(s)
   l = 1  # suppose l_k = 1
   M = M * s + l
   arrayOfM = arrayOfM[1:]
   arrayOfM.append(M)
   #########################

   # here we update accumulating sum
   r = 1  # suppose r_k = 1
   gW = gW + M * r

"""