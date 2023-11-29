#projector builder function for VICReg global loss

def Projector(embedding, dimension):
  return nn.Sequential(
      nn.Linear(embedding, dimension),
      nn.ReLU(True),
      nn.Linear(dimension, dimension),
      nn.BatchNorm1d(dimension),
      nn.ReLU(True),
      nn.Linear(dimension, dimension, bias = False)
  )

#useful later during loss calculation
def off_diagonal(x):
  n, m = x.shape
  assert n == m
  return x.flatten()[:-1].view(n-1, n+1)[:, 1:].flatten()

