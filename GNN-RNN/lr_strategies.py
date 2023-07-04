from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR


# learning rate adjustment strategy
class WarmUp:
    # warm up
    def __init__(self, warm_up_epochs, decay_rate):
        # initialization
        self.warm_up_epochs = warm_up_epochs
        self.decay_rate = decay_rate

    def warm_up_lr(self, epoch):
        if epoch < self.warm_up_epochs:
            return (epoch + 1) / self.warm_up_epochs
        else:
            return self.decay_rate ** (epoch - self.warm_up_epochs)

    def get_scheduler(self,optimizer):
        scheduler = LambdaLR(optimizer,self.warm_up_lr,last_epoch=-1)
        return scheduler
