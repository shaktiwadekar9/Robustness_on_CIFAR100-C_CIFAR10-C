import csv
import os

def savemetrics(all_tr_loss,
                all_tr_acc,
                all_val_loss,
                all_val_acc,
                best_acc,
                mCE,
                path):
    """
    Saving all the metrics and best models accuracy in a text file

    Args:
        losses and accuracies
    Return:
        None
    """
    filepath = os.path.join(path, 'loss_and_acc.txt')
    with open(filepath, 'w') as f:

        headers = ['tr_loss', 'tr_acc', 'val_loss', 
                    'val_acc', 'best_acc', 'mCE' ]
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        dict_w = {'tr_loss': all_tr_loss,
                    'tr_acc': all_tr_acc,
                    'val_loss': all_val_loss,
                    'val_acc': all_val_acc,
                    'best_acc': best_acc,
                    'mCE': mCE
                    }

        writer.writerow(dict_w)

    print("All the metrics saved!")
