import matplotlib.pyplot as plt 
import seaborn as sns

#plt.style.use('https://gist.githubusercontent.com/thissop/d1967ecb352011a4580e2b2274959a89/raw/fe22f835ecb734523e88884bd30c751ca6511cf2/stylish.mplstyle')

sns.set_context("paper") # font_scale=
sns.set_palette('deep') #
seaborn_colors = sns.color_palette('deep') #

plt.rcParams['font.family'] = 'serif'
plt.rcParams["mathtext.fontset"] = "dejavuserif"

def plot_training_diagnostics(loss_history, save_path:str=None):
    r'''
    _Plot diagnostic plots from training process._
    
    Parameters
    ----------      

    loss_history : `dict`
        By default, Ankit's original train_model functionality returned a one-item list with a Keras history object. My version now returns the dictionary from that one item, and that is what should be provided to this function. 

    save_path : `str`
        If defined, plots will be saved at this directory together. 

    Returns
    -------

    figures : `list`
        List of figures 

    '''
    import matplotlib.pyplot as plt 
    import seaborn as sns
    import numpy as np
    import os

    train_keys = ['loss', 'dice_coef', 'dice_loss', 'specificity', 'sensitivity', 'accuracy']

    x = np.arange(0, len(loss_history['loss']))

    figures = []

    for train_key in train_keys:
        fig, ax = plt.subplots()

        val_key = f'val_{train_key}'
        ax.plot(x, loss_history[train_key], label=train_key)
        ax.plot(x, loss_history[val_key], label=val_key)
        ax.legend(loc='best')

        ax.set(xlabel='Training Epoch', ylabel=train_key.title().replace('_', ' '))

        #$plt.tight_layout()

        figures.append(fig)
        
        if save_path is not None: 
            plt.savefig(os.path.join(save_path, f'{train_key}.png'), dpi=200)

        plt.clf()
    
    return figures

        