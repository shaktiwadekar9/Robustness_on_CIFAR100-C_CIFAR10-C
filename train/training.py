import torch
from torch import nn
import torch.optim as optim
import os
import tqdm

class Train(nn.Module):
    """
    Main Training class. Has all the methods related to training a model
    
    Args:
        cfg: all configurations 
        model: Neural Network model
        device: 'cuda' or 'cpu'
    """

    def __init__(self, cfg, model, device):
        super(Train, self).__init__()

        self.cfg = cfg
        self.model = model
        self.device = device
        self.epochs = self.cfg['epochs']
        self.bt_sz = self.cfg['trBatch']
        self.optimizer = self.set_optimizer()
        self.criterion = self.set_criterion()
        self.scheduler = self.set_lrscheduler()

    def set_lrscheduler(self):
        """ Returns learning rate schedular
        """
        if self.cfg['scheduler']['name']=='StepLR':
            return optim.lr_scheduler.StepLR(self.optimizer,
                                            step_size=self.cfg['scheduler']['step_size'],
                                            gamma=self.cfg['scheduler']['gamma']) #lr =gamma*lr
        elif self.cfg['scheduler']['name']=='MultiStepLR':
            print(f"Using MultiStepLR. Milestones: {self.cfg['scheduler']['milestones']}")
            return optim.lr_scheduler.MultiStepLR(self.optimizer,
                                            milestones=self.cfg['scheduler']['milestones'],
                                            gamma=self.cfg['scheduler']['gamma'])

        else:
            raise ValueError('Only StepLR implemented')


    
    def set_optimizer(self):
        """Returns Optimizer
        """

        if self.cfg['optimizer']['name']=='adam':
            return optim.Adam(self.model.parameters(),
                                lr=self.cfg['optimizer']['lr'],
                                weight_decay=self.cfg['optimizer']['wd'])

        elif self.cfg['optimizer']['name']=='rmsprop':
            return optim.RMSprop(self.model.parameters(),
                                lr=self.cfg['optimizer']['lr'],
                                alpha=self.cfg['optimizer']['alpha'],
                                momentum=self.cfg['optimizer']['momentum'],
                                weight_decay=self.cfg['optimizer']['wd'])
        
        elif self.cfg['optimizer']['name']=='sgd':
            return optim.SGD(self.model.parameters(),
                            lr=self.cfg['optimizer']['lr'],
                            momentum=self.cfg['optimizer']['momentum'],
                            weight_decay=self.cfg['optimizer']['wd'])

        else:
            raise ValueError('Only adam optimizer implemented')

    def set_criterion(self):
        """ Returns loss function
        """

        if self.cfg['loss_fun']=='cross_entropy_loss':
            return nn.CrossEntropyLoss()
        else:
            raise ValueError('Only CrossEntropyLoss implemented')


    def train_att(self, mode, loader, epoch):
        """ Main training loop

        Can be called directly by using Train.train_att()
        OR
        Can be called for training from checkpoint using Train.from_ckpt()

        Args:
            mode: 'train'
            loader: trainloader
            epoch: Current epoch 
        Returns:
            loss and accuracy
        """
        
        self.model.to(self.device).train()  # To ensure all layers are in training mode

        epoch_loss = 0

        for batch_idx, data in enumerate(loader):
            
            # get inputs and labels
            inputs = data[0].to(self.device, non_blocking=True)
            labels = data[1].to(self.device, non_blocking=True)
            
            # Zero the parameter gradients 
            self.optimizer.zero_grad()

            # Forward
            outputs = self.model(inputs)
            
            loss = self.criterion(outputs, labels)
            
            # Backward
            loss.backward()

            # Optimze
            self.optimizer.step()

            # Capture metric/loss
            epoch_loss += loss.item()

            if batch_idx % self.cfg['print_rate_bt'] == 0:
                print(f'{mode} Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}')


        # learning scheduler
        self.scheduler.step()
        print(f"New Learning rate: {self.optimizer.param_groups[0]['lr']}")
        # Find accuracy
        acc = self.accuracy(mode, loader)
        

        return epoch_loss/(batch_idx+1), acc


    def eval_att(self, mode, loader, epoch):
        """ Main evaluation loop 

        Created separately due to memory increase issues when doing evaluation in training loop with .eval() mode

        Args:
            mode: 'eval'
            loader: testloader/valloader
            epoch: Current epoch 
        Returns:
            loss and accuracy
        """
        
        self.model.to(self.device).eval()

        epoch_loss = 0
        
        with torch.no_grad():
            for batch_idx, data in enumerate(loader):
                
                # get inputs and labels
                inputs = data[0].to(self.device, non_blocking=True)
                labels = data[1].to(self.device, non_blocking=True)
                
                # Zero the parameter gradients 
                self.optimizer.zero_grad()

                # Forward
                outputs = self.model(inputs)
                
                loss = self.criterion(outputs, labels)

                # Capture metric/loss
                epoch_loss += loss.item()

                if batch_idx % self.cfg['print_rate_bt'] == 0:
                    print(f'{mode} Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}')

        # Find accuracy
        acc = self.accuracy(mode, loader)  

        return epoch_loss/(batch_idx+1), acc


    def test_att(self, mode, loader, epoch):
        """ Main testing loop

        Args:
            mode: 'test'
            loader: testloader
            epoch: Current epoch 
        Returns:
            accuracy
        """
        
        return self.accuracy(mode, loader)


    def save_ckpt(self, epoch, ckpt_path):
        """Saves model after every epoch"""
        torch.save(
                {
                'model':self.model.state_dict(),
                'optimizer':self.optimizer.state_dict(),
                'epoch': epoch+1,
                'criterion': self.criterion
                },
                ckpt_path
                )

    def save_best(self, epoch, best_path, best_result):
        torch.save(
                {
                'model':self.model.state_dict(),
                'optimizer':self.optimizer.state_dict(),
                'epoch': epoch+1,
                'criterion': self.criterion,
                'best_result': best_result
                },
                best_path
                )
                


    def from_ckpt(self):
        """Start training from checkpoint epoch+1

        To start training from epoch=0, directly use Train.train_att()
        To start training from checkpoint epoch, use Train.from_checkpoint()
        """
        fullpath = os.path.join(self.cfg['ckpt']['path'], 
                                self.cfg['ckpt']['name'])
        checkpoint = torch.load(fullpath)
        self.model.load_state_dict(checkpoint['model'])
        print("Model loaded!")
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print("Optimizer loaded!")
        start_epoch = checkpoint['epoch'] - 1
        print(f"start_epoch: {start_epoch}")
        self.criterion = checkpoint['criterion']

        #self.train_att()
        return start_epoch

    def from_best(self, fullpath):
        """Start training from checkpoint epoch+1

        To start training from epoch=0, directly use Train.train_att()
        To start training from checkpoint epoch, use Train.from_checkpoint()
        """
        checkpoint = torch.load(fullpath)
        #print(checkpoint['model'])
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer = checkpoint['optimizer']
        start_epoch = checkpoint['epoch'] - 1
        self.criterion = checkpoint['criterion']

        

    def accuracy(self, mode, loader):
        """Calculate accuracy

        Args:
            mode: train or val or test
            loader: trainloader, valloader, testloader
        Returns:
            Accuracy
        """

        correct=0
        total=0

        self.model.to(self.device).eval()

        with torch.no_grad():
            for data in loader:

                images = data[0].to(self.device, non_blocking=True)
                labels = data[1].to(self.device, non_blocking=True)

                outputs = self.model(images)

                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)

                correct += (predicted==labels).sum().item()
        

        accuracy = correct/total*100
        #print(f" {mode} accuracy: {accuracy:.3} %")

        return accuracy


    def test_bestmodel(self, fullpath, loader):
        """ Calculate best model's accuracy

        Args: 
            fullpath: fullpath to best model's .pth/.pt file
            loader: trainloader/testloader
        Returns:
            Accuracy
        """
        
        self.from_best(fullpath)

        return self.accuracy('test', loader)









