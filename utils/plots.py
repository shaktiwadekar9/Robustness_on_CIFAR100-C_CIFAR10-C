import matplotlib.pyplot as plt
import os

def plot(cfg, tr_data, val_data, metric):

    plt.plot(range(1, len(tr_data)+1), tr_data, 'g', label=f'Train_{metric}')
    plt.plot(range(1, len(val_data)+1), val_data, 'b', label=f'Val_{metric}')

    plt.title(f'Train and Val {metric}')
    plt.xlabel('Epochs')
    plt.ylabel(f'{metric}')
    plt.legend()
    
    savedir = os.path.join(cfg['plotpath'], 'graphs')
    if not os.path.exists(savedir):
        os.makedirs(savedir)
        print("Directory for saving loss and metric graphs created")
    savepath = os.path.join(savedir, f'{metric}.png')

    plt.savefig(savepath)
    plt.close()

