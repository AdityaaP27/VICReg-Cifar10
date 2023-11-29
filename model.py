



class VICReg(nn.Module):

  def __init__(self,batchsize):
    super(VICReg, self).__init__()
    self.batchsize = batchsize
    self.embedding = 64
    self.num_features = 256
    self.backbone = ResNet_CIFAR(block = BasicBlock, img_channels = 3, num_layers = 18, num_classes = 10)
    self.backbone.fc = nn.Identity()
    self.projector = Projector(self.embedding, self.num_features)


  def forward(self, x, y):
    x = self.projector(self.backbone(x))
    y = self.projector(self.backbone(y))

    repr_loss = F.mse_loss(x,y)

    x = x-x.mean(dim = 0)
    y = y-y.mean(dim = 0)


    std_x = torch.sqrt(x.var(dim=0) + 0.0001)
    std_y = torch.sqrt(y.var(dim=0) + 0.0001)


    std_loss = torch.mean(F.relu(1-std_x))/2 + torch.mean(F.relu(1 - std_y))/2

    cov_x = (torch.matmul(x.T, x))/(self.batchsize - 1)
    cov_y = (torch.matmul(y.T, y))/(self.batchsize - 1)

    cov_loss = off_diagonal(cov_x).pow_(2).sum().div(self.num_features) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

    loss = (25*repr_loss + 25*std_loss + 1*cov_loss)

    return loss
