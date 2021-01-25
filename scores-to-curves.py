import torch
import pandas as pd
import matplotlib.pyplot as plt
from pytorch_lightning.metrics import functional as plmF

def predict_to_score(predicted, target):
    assert predicted.shape == target.shape
    score = target - torch.sigmoid(predicted)
    return score.detach().cpu()

def scores_to_prob(scores):
    target = (scores >= 0.).float()
    prob = target - scores
    return prob, target

def plot(prob, target, output_name, show_plots=False, plot_pr=False, plot_roc=True):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    pr_auc, roc_auc = [], []
    for p, t in zip(prob, target):
        if not torch.all(t == 0):
            if plot_pr:
                precision, recall, _ = plmF.precision_recall_curve(p, t, pos_label=1)
                try:
                    pr_auc.append(plmF.classification.auc(recall, precision))
                except:
                    pass
                ax.plot(recall, precision, linewidth=0.2)
            if plot_roc:
                fpr, tpr, _ = plmF.roc(p, t, pos_label=1)
                try:
                    roc_auc.append(plmF.classification.auc(fpr, tpr))
                except:
                    pass
                ax.plot(fpr, tpr, linewidth=0.2)
    if show_plots:
        plt.show()
    else:
        plt.savefig(output_name)
    return pr_auc, roc_auc

def plot_progress(files, mean_over):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cmap = plt.get_cmap()
    for n, f in enumerate(files):
        color = cmap((n+1) / max(3, len(files)))
        table = pd.read_csv(f, sep='\t')
        x = table['iter']
        y = table['loss']
        stacked = [y[i:i - mean_over] for i in range(mean_over)]
        smoothed_y = [sum(items) / mean_over for items in zip(*stacked)]
        ax.plot(x, y, c=color, linewidth=0.5)
        start = mean_over // 2
        end = len(x) - mean_over + start
        ax.plot(x[start:end], smoothed_y, c=color, label=f)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.savefig('ft-deepsea-loss-plots.png')
    plt.show()

def main():
    # filename = './test-scores.pt'
    # scores = torch.load(filename)
    # prob, target = scores_to_prob(scores)
    # prob = prob.permute(1, 0)
    # target = target.permute(1, 0)
    # global_auc = plot(prob.flatten().unsqueeze(0),
    #             target.flatten().unsqueeze(0), 'global-auc-plot.png')
    # _, individual_roc = plot(prob, target, 'individual-auc-plot.png')
    # print(min(individual_roc), global_auc, max(individual_roc))

    filenames = ['From scratch loss', 'Pretrained loss']
    plot_progress(filenames, 20)

if __name__ == '__main__':
    main()