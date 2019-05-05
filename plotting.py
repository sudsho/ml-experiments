# matplotlib + seaborn helpers
import matplotlib.pyplot as plt
import seaborn as sns

def init():
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = (8, 5)
    plt.rcParams['axes.titlesize'] = 14

def confusion(cm, labels):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('predicted')
    ax.set_ylabel('true')
    return ax
