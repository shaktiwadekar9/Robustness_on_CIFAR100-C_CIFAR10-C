import torch
from torch import nn
import torchvision.models as vmodels

import argparse
import yaml
import torchsummary as summary
import time
import random
import numpy as np
import shutil

from utils.all_datasets_vet import load_cifar10, load_cifar100
from utils.plots import plot
from utils.saving import savemetrics
from utils.get_all_models import get_model

from train.training import Train
#from test import robustness
import robustness


import os 
# Parsing command line arguments
parser = argparse.ArgumentParser(description='Main file')
parser.add_argument("-cp", "--cfg_pth", type=str,
                    default="configs/resnet50.yml",
                    help="configuration file path")

parser.add_argument("-g","--gpu", type=str,
                    default='0', help="GPU number 0/1/2")

args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# Main function
def main():
    
    # Load configuration
    with open(args.cfg_pth, 'r') as stream:
        cfg = yaml.safe_load(stream)
    print(f"Configurations:{cfg}")

    # model names
    ckptmodel = os.path.join(cfg['ckpt']['path'], cfg['ckpt']['name'])
    bestmodel = os.path.join(cfg['bestmodel']['path'], cfg['bestmodel']['name'])
    
    # For reproducibility
    random.seed(cfg['random_seed'])
    np.random.seed(cfg['random_seed'])
    torch.manual_seed(cfg['random_seed'])

    # Load dataset and trainloader
    if cfg['train_db']== 'CIFAR100':
        trainloader, valloader, testloader = load_cifar100(cfg)
        print("CIFAR100 dataset loaded!")
    elif cfg['train_db']== 'CIFAR10':
        trainloader, valloader, testloader = load_cifar10(cfg)
        print("CIFAR10 dataset loaded!")
    else:
        raise NotImplementedError("Only CIFAR100/-C and CIFAR10/-C supported")

    # Load model
    model = get_model(cfg)

    # Run model on multiple gpus
    model = torch.nn.DataParallel(model)
    model = model.cuda()

    # Train and eval
    Training = Train(cfg, model, device='cuda')
    
    start_epoch = 0


    all_tr_loss = []
    all_tr_acc = []
    all_val_loss = [] 
    all_val_acc = []
    best_val_acc = 0
    all_epoch_start = time.time()
    
    # Train loop
    try:
        for idx, epoch in enumerate(range(start_epoch, cfg['epochs'])):

            start = time.time()
            tr_loss, tr_acc = Training.train_att(mode='train', loader=trainloader, epoch=epoch)
            end = time.time()
            print(f"Train Epoch: {epoch}, Loss: {tr_loss}, Acc: {tr_acc:.4}")
            print(f"Epoch time: {(end-start)/60} minutes")

            # Save checkpoint
            if not os.path.exists(cfg['ckpt']['path']):
                os.makedirs(cfg['ckpt']['path'])
                print("Directory for saving checkpoint created!")
            Training.save_ckpt(epoch, ckptmodel) 
            print("Checkpoint Saved!")


            # Eval loop executed conditionally
            if epoch % cfg['val_rate'] == 0:
                start = time.time()
                val_loss, val_acc = Training.eval_att(mode='eval', loader=valloader, epoch=epoch)
                end = time.time()
                print(f"Eval Epoch: {epoch}, Loss: {val_loss}, Acc: {val_acc:.4}")
                print(f"Eval epoch time: {(end-start)/60} minutes")

                if epoch==0:# Do it once at the start to test testloop
                    test_acc = Training.test_att(mode='test', loader=testloader, epoch=epoch)
                    print(f"Test Accuracy: {test_acc:.3}")
                
                # Save the best model based on validation accuracy
                if val_acc > best_val_acc:
                    # check if save folder exists, if not create
                    if not os.path.exists(cfg['bestmodel']['path']):
                        os.makedirs(cfg['bestmodel']['path'])
                        print("Directory for saving bestmodel created!")
                    Training.save_best(epoch, bestmodel, val_acc) 
                    print("New best Model saved!")
                    best_val_acc = val_acc
                
                print(f"Best Val Acc: {best_val_acc:.4}")

            # Record data 
            all_tr_loss.append(tr_loss)
            all_tr_acc.append(tr_acc)
            all_val_loss.append(val_loss)
            all_val_acc.append(val_acc)
            


        all_epoch_end = time.time()
        print(f"Total training time: {(all_epoch_end-all_epoch_start)/60} minutes")

        # Test data accuracy
        test_acc = Training.test_att(mode='test', loader=testloader, epoch=epoch)
        print(f"Final epoch model Test Accuracy: {test_acc:.4}")

        # Best model accuracy
        best_acc = Training.test_bestmodel(bestmodel, testloader)
        print(f"Best model Test Accuracy: {best_acc:.4}")

        # Calculate mCE on CIFAR100-C/CIFAR10-C
        if cfg['train_db']== 'CIFAR100': 
            mCE = robustness.mCE_cifar100(cfg, model, device='cuda')
        elif cfg['train_db']== 'CIFAR10': 
            mCE = robustness.mCE_cifar10(cfg, model, device='cuda')
        else:
            raise NotImplementedError("Only CIFAR100-c and cifar10-c supported for robustness evaluation")
        print(f"Robustness: {cfg['robustness']['name']} mCE: {mCE}")


        # Print train statistics
        #print(f"Training losses: {all_tr_loss}")
        #print(f"Training accuracies: {all_tr_acc}")
        #print(f"Validation losses: {all_val_loss}")
        #print(f"Validation accuracies: {all_val_acc}")

        # Save losses and accuracies
        savemetrics(all_tr_loss,
                    all_tr_acc,
                    all_val_loss,
                    all_val_acc,
                    best_acc,
                    mCE,
                    cfg['plotpath']
                    )
        
        # Record Train statistics
        plot(cfg, all_tr_loss, all_val_loss, 'Loss_CA')
        plot(cfg, all_tr_acc, all_val_acc, 'Acc_CA')

        # Copying config file
        # shutil.copy(src, dest)
        shutil.copy(args.cfg_pth, cfg['plotpath'])
        print("Config file copied!")

    except KeyboardInterrupt:
        
        print("KeyboardInterrupt detected!")
        print("Plotting and saving acc and loss values till now!")
        all_epoch_end = time.time()
        print(f"Total training time: {(all_epoch_end-all_epoch_start)/60} minutes")
        # Record Train statistics
        #print(f"Training losses: {all_tr_loss}")
        #print(f"Training accuracies: {all_tr_acc}")
        #print(f"Validation losses: {all_val_loss}")
        #print(f"Validation accuracies: {all_val_acc}")
        plot(cfg, all_tr_loss, all_val_loss, 'Loss_CA')
        plot(cfg, all_tr_acc, all_val_acc, 'Acc_CA')

        # Best model accuracy
        best_acc = Training.test_bestmodel(bestmodel, testloader)
        print(f"Best model Test Accuracy: {best_acc:.4}")

        # Calculate mCE on CIFAR100-C/CIFAR10-C
        if cfg['train_db']== 'CIFAR100': 
            mCE = robustness.mCE_cifar100(cfg, model, device='cuda')
        elif cfg['train_db']== 'CIFAR10': 
            mCE = robustness.mCE_cifar10(cfg, model, device='cuda')
        else:
            raise NotImplementedError("Only CIFAR100-c and cifar10-c supported for robustness evaluation")
        print(f"Robustness: {cfg['robustness']['name']} mCE: {mCE}")
        
        # Save losses and accuracies
        print("Saving all metrics and best accuracy.....")
        savemetrics(all_tr_loss,
                    all_tr_acc,
                    all_val_loss,
                    all_val_acc,
                    best_acc,
                    mCE,
                    cfg['plotpath']
                    )

        # Copying config file
        # shutil.copy(src, dest)
        shutil.copy(args.cfg_pth, cfg['plotpath'])
        print("Config file copied!")


if __name__ == '__main__':
    main()
