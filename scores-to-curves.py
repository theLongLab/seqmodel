import torch
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

def plot(prob, target, plot_pr=False, plot_roc=True):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    pr_auc, roc_auc = [], []
    for p, t in zip(prob, target):
        if not torch.all(t == 0):
            if plot_pr:
                precision, recall, _ = plmF.precision_recall_curve(p, t, pos_label=1)
                pr_auc.append(plmF.classification.auc(recall, precision))
                ax.plot(recall, precision, c='b')
            if plot_roc:
                fpr, tpr, _ = plmF.roc(p, t, pos_label=1)
                # roc_auc.append(plmF.classification.auc(fpr, tpr))
                ax.plot(fpr, tpr, c='g')
    plt.show()
    return pr_auc, roc_auc

def main():
    filename = 'outputs/56082675-test-scores-1k.pt'
    scores = torch.load(filename)
    prob, target = scores_to_prob(scores)
    prob = prob.permute(1, 0)
    target = target.permute(1, 0)
    global_auc = plot(prob.flatten().unsqueeze(0), target.flatten().unsqueeze(0))
    _, individual_roc = plot(prob, target)
    print(min(individual_roc), global_auc, max(individual_roc))

if __name__ == '__main__':
    main()