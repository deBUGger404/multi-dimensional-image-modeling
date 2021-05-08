import torch
from utils import *

class model_train:
    def __init__(self, model, optimizer, criterion, lr_scheduler, device, train_loader, test_loader, n_epochs):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.n_epochs = n_epochs
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler

    def train(self,epoch):
        acc = AverageMeter()
        losses = AverageMeter()
        
        self.model.train()
        for step, (x, y) in enumerate(self.train_loader):
            if torch.cuda.is_available():
                x, y = x.to(device), y.to(device)

            bs = x.size(0)
            self.optimizer.zero_grad()
            logits = self.model(x.float())

            metrics = binary_acc(logits, y)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()
            losses.update(loss.item(), bs)
            acc.update(metrics, bs)
        print(f'Epoch {epoch}: | Training_Loss: {losses.avg:.5f} \
                    | Training_Acc: {acc.avg:.3f}')
        return losses.avg

    def validate(self, epoch):
        acc = AverageMeter()
        losses = AverageMeter()
        
        self.model.eval()
        with torch.no_grad():
            for step, (x, y) in enumerate(self.test_loader):
                if torch.cuda.is_available():
                    x, y = x.to(device), y.to(device)
                bs = x.size(0)
                logits = self.model(x.float())
                metrics = binary_acc(logits, y)
                loss = self.criterion(logits, y)
                losses.update(loss.item(), bs)
                acc.update(metrics, bs)

        print(f'Epoch {epoch}: | Validation_Loss: {losses.avg:.5f} | Validation_Acc: {acc.avg:.3f}')
        return acc.avg, losses.avg

    def run(self):
        self.model = torch.nn.DataParallel(self.model)
        self.best_top1 = 0.
        self.early_stopping = EarlyStopping(patience=15, verbose=True)
        for epoch in range(self.n_epochs):
            # training
            train_loss = self.train(epoch)
            # validation
            top1, val_loss = self.validate(epoch)
            self.best_top1 = max(self.best_top1, top1)
            self.early_stopping(val_loss, self.model)

            if self.early_stopping.early_stop:
                print("Early stopping")
                break
            self.lr_scheduler.step()

        print("Final best acc_score = {:.4%}".format(self.best_top1))
        return self.model