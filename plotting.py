# matplotlib + seaborn helpers
import matplotlib.pyplot as plt
import seaborn as sns

def init():
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = (8, 5)
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['savefig.dpi'] = 120

def confusion(cm, labels):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('predicted')
    ax.set_ylabel('true')
    return ax

def feat_importance(model, names, top=15):
    import pandas as pd
    s = pd.Series(model.feature_importances_, index=names).sort_values()
    s.tail(top).plot.barh()
    return s

def roc(probs_dict, y_true):
    """plot multiple ROC curves on the same axes; probs_dict = {name: probs}."""
    from sklearn.metrics import roc_curve, auc
    fig, ax = plt.subplots()
    for name, p in probs_dict.items():
        f, t, _ = roc_curve(y_true, p)
        ax.plot(f, t, label=f'{name} AUC={auc(f,t):.3f}')
    ax.plot([0,1],[0,1], 'k--')
    ax.legend()
    return ax


def confusion_heatmap(y_true, y_pred, labels=None):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_xlabel('pred'); ax.set_ylabel('true')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha='center', va='center')
    return fig


def plot_hist_compare(a, b, bins=40, labels=('a', 'b')):
    import matplotlib.pyplot as plt
    plt.hist(a, bins=bins, alpha=0.5, label=labels[0])
    plt.hist(b, bins=bins, alpha=0.5, label=labels[1])
    plt.legend()


def confusion_heatmap(y_true, y_pred, labels=None):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_xlabel('pred'); ax.set_ylabel('true')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha='center', va='center')
    return fig


def plot_loss(history, title='loss'):
    import matplotlib.pyplot as plt
    plt.figure()
    if 'train' in history:
        plt.plot(history['train'], label='train')
    if 'val' in history:
        plt.plot(history['val'], label='val')
    plt.legend(); plt.title(title); plt.xlabel('step'); plt.ylabel('loss')
