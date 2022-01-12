# Test robustness of model using CIFAR100-c dataset
# References:
# 1. https://github.com/hendrycks/robustness/blob/master/ImageNet-C/test.py
# 2. https://github.com/psh150204/AugMix/blob/master/main.py
import os
import torch
import numpy as np
import yaml

import torchvision.datasets as datasets
import torchvision.transforms as transforms

#import torchvision.models as models
import torch.utils.model_zoo as model_zoo

from utils.get_all_models import get_model


def show_performance_cifar(model, dataloader, 
                            distortion_name=None, 
                            device='cuda'):

    # Calculate error
    model.to(device).eval()  # Put model in eval mode

    err, correct, total = 0,0,0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(data)

            _, pred = torch.max(output.data, 1)
            correct += (pred==target).sum().item()
            total += target.size(0)

    err = 1 - correct / total
    correct = correct / total
    #print(f"Total correct prediction (%): {correct*100}")

    if distortion_name is not None: # For robustness
        print(f"Distortion: {distortion_name}, Err: {err}")
        print(f"Distortion: {distortion_name}, Correct: {correct}")
        print(f"Total images in {distortion_name}: {total}")
        

    return err, correct



def cal_mCE(model, dataset_root, 
            dataset_transforms, 
            dataset_name,
            device='cuda'):
    
    # All the distortions: Total 15
    distortions = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                    'defocus_blur', 'glass_blur', 'motion_blur',
                    'zoom_blur', 'snow', 'frost',
                    'brightness', 'contrast', 'elastic_transform',
                    'pixelate', 'jpeg_compression', 'speckle_noise',
                    'gaussian_blur', 'spatter', 'saturate']

    if dataset_name=="cifar100":
        test_data = datasets.CIFAR100("./dataset", 
                                        train=False, 
                                        transform=dataset_transforms,
                                        download=True)
    elif dataset_name=="cifar10":
        test_data = datasets.CIFAR10("./dataset", 
                                        train=False, 
                                        transform=dataset_transforms,
                                        download=True)
    else:
        raise NotImplementedError("Only for CIFAR100 and CIFAR10")
    
    # Standard dataset accuracy:
    standard_test_loader = torch.utils.data.DataLoader(test_data,
                                                        batch_size=32,
                                                        shuffle=False,
                                                        num_workers=8,
                                                        pin_memory=True)
    err, correct = show_performance_cifar(model,
                                        standard_test_loader,
                                        device=device
                                        )
    
    print(f"Standard Err (%): {err*100}")
    print(f"Standard Correct (%): {correct*100}")

    # Calculate errors: mCE
    errors = []
    corrects = []
    for distortion_name in distortions:
        
        full_data_pth = os.path.join(dataset_root, f"{distortion_name}.npy")
        full_labels_pth = os.path.join(dataset_root, "labels.npy")

        test_data.data = np.load(full_data_pth)
        test_data.targets = torch.LongTensor(np.load(full_labels_pth))

        testloader = torch.utils.data.DataLoader(test_data,
                                                batch_size=32,
                                                shuffle=False,
                                                num_workers=8,
                                                pin_memory=True)

        # error rate for a distortion
        err, correct = show_performance_cifar(model,
                                            testloader,
                                            distortion_name,
                                            device=device
                                            )
        
        # Collect all distortion rates to calculate mCE later
        errors.append(err)
        corrects.append(correct)

        print('Distortion: {:15s} | CE (unnormalized) (%): {:.2f}'.format(distortion_name, 100*err))


    # Calculate and print mCE
    print('mCE (unnormalized) (%): {:.2f}'.format(100 * np.mean(errors)))

    return 100 * np.mean(errors)


def load_best_model(cfg, model):

    bestmodelpth = os.path.join(cfg['bestmodel']['path'], cfg['bestmodel']['name'])
    bestmodel = torch.load(bestmodelpth) # load .pth file
    model.load_state_dict(bestmodel['model'])
    print("Best model loaded!")

    return model


def mCE_cifar100(cfg, model, device='cuda'):
    
    dataset_root = cfg['robustness']['dataset']

    # load best model
    model = load_best_model(cfg, model) 

    print("Calculating Errors on CIFAR100 and CIFAR100-C")
    dataset_name = "cifar100"
    cifar_transforms = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.50707516,  0.48654887,  0.44091784),
                                            (0.26733429,  0.25643846,  0.27615047))
                        ])
#    test_dataset = torchvision.datasets.CIFAR100(root='./dataset',
#                                                train=False,
#                                                download=True,
#                                                transform=cifar_transforms)
 
    return cal_mCE(model, dataset_root, 
                    dataset_transforms=cifar_transforms, 
                    dataset_name=dataset_name,
                    device=device) 


def mCE_cifar10(cfg, model, device='cuda'):

    # load dataset
    dataset_root = cfg['robustness']['dataset']

    # load best model
    model = load_best_model(cfg, model) 

    # Calculate err
    print("Calculating Errors on CIFAR10 and CIFAR10-C")
    dataset_name = "cifar10"
    cifar_transforms = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.49139968,  0.48215841,  0.44653091),
                                            (0.24703223,  0.24348513,  0.26158784))
                        ])
#    test_dataset = torchvision.datasets.CIFAR100(root='./dataset',
#                                                train=False,
#                                                download=True,
#                                                transform=cifar_transforms)
 
    return cal_mCE(model, dataset_root, 
                     dataset_transforms=cifar_transforms, 
                    dataset_name=dataset_name,
                    device=device) 



    



#if __name__== "__main__":
#
#    # Load config file to load correct model
#    with open(args.cfg_pth, 'r') as stream:
#        cfg = yaml.safe_load(stream)
#    print(f"Configurations: {cfg}")
#    
#    # Create model
#    model = get_model(cfg)
#    model = torch.nn.DataParallel(model).cuda() # modules.layername saved if dataparallel was used while saving the model. Therefore need to wrap again with DataParallel when loading weights
#    print("Model created!")
#
#    # calculate mCE on cifar
#    print(f"Evaluating ...")
#    mCE_cifar10(cfg, model, device='cuda')


