import torch



class RNN:
    def __init__(self, dim=1):
        self.W_hx = torch.zeros(dim)
        self.W_hh = torch.zeros(dim)
        self.W_hy = torch.zeros(dim)
        self.b_h = torch.zeros(dim)
        self.b_y = torch.zeros(dim)

        self.h = torch.zeros(dim)
        self.y = torch.zeros(dim)

    def forward(self, x, h):
        self.h = torch.sigmoid(self.W_hx * x + self.W_hh * h + self.b_h)
        self.y = torch.sigmoid(self.W_hy * self.h + self.b_y)

    def backward(self):
        



print(torch.zeros(1))


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