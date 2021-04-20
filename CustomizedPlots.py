import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix

def auc_plot_metrics(pred, true_labels):
    """Plots a ROC curve with the accuracy and the AUC"""
    acc = accuracy_score(true_labels, np.array(pred.flatten() >= .5, dtype='int'))
    fpr, tpr, thresholds = roc_curve(true_labels, pred)
    auc = roc_auc_score(true_labels, pred)

    fig, ax = plt.subplots(1, figsize=(8,8))
    ax.plot(fpr, tpr, color='red')
    ax.plot([0,1], [0,1], color='black', linestyle='--')
    ax.set_title(f"AUC: {auc}\nACC: {acc}");
    return fig


def plot_confusion_matrix(cm,target_names,title='Confusion matrix',cmap=None,normalize=True):
  accuracy = np.trace(cm) / float(np.sum(cm))
  misclass = 1 - accuracy

  if cmap is None:
      cmap = plt.get_cmap('Blues')

  plt.figure(figsize=(8, 6))
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()

  if target_names is not None:
      tick_marks = np.arange(len(target_names))
      plt.xticks(tick_marks, target_names, rotation=45)
      plt.yticks(tick_marks, target_names)

  if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

  thresh = cm.max() / 1.5 if normalize else cm.max() / 2
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      if normalize:
          plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
      else:
          plt.text(j, i, "{:,}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
  plt.show()


def f1_score_bar(f1_dl_hate, f1_dl_irony, f1_dl_offensive, f1_xl_hate, f1_xl_irony, f1_xl_offensive):
    n_groups = 3
    DistilBert = [f1_dl_hate * 100, f1_dl_irony * 100, f1_dl_offensive * 100]
    XLNet = [f1_xl_hate * 100, f1_xl_irony * 100, f1_xl_offensive * 100]

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(index, DistilBert, bar_width,
                     alpha=opacity,
                     color='b',
                     label='DistilBert')

    rects2 = plt.bar(index + bar_width, XLNet, bar_width,
                     alpha=opacity,
                     color='g',
                     label='XLnet')

    plt.ylabel('Scores')
    plt.title('F1 Scores (Macro)')
    plt.xticks(index + bar_width, ('Hate', 'Irony', 'Offensive'))
    plt.legend()

    plt.tight_layout()
    plt.show()

def auc_plot_metrics(hate_pred_dl,hate_pred_xl,hate_true_labels,irony_pred_dl,irony_pred_xl,irony_true_labels,offensive_pred_dl,offensive_pred_xl,offensive_true_labels):
    """Plots a ROC curve with the accuracy and the AUC"""
    fpr, tpr, thresholds = roc_curve(hate_true_labels, hate_pred_dl)
    fpr2, tpr2, thresholds2 = roc_curve(hate_true_labels, hate_pred_xl)
    auc_dl_hate = roc_auc_score(hate_true_labels, hate_pred_dl)
    auc_xl_hate = roc_auc_score(hate_true_labels, hate_pred_xl)

    fpr3, tpr3, thresholds3 = roc_curve(irony_true_labels, irony_pred_dl)
    fpr4, tpr4, thresholds4 = roc_curve(irony_true_labels, irony_pred_xl)
    auc_dl_irony = roc_auc_score(irony_true_labels, irony_pred_dl)
    auc_xl_irony = roc_auc_score(irony_true_labels, irony_pred_xl)

    fpr5, tpr5, thresholds5 = roc_curve(offensive_true_labels, offensive_pred_dl)
    fpr6, tpr6, thresholds6 = roc_curve(offensive_true_labels, offensive_pred_xl)
    auc_dl_offensive = roc_auc_score(offensive_true_labels, offensive_pred_dl)
    auc_xl_offensive = roc_auc_score(offensive_true_labels, offensive_pred_xl)

    fig, ax = plt.subplots(1,3)
    fig.set_figheight(5)
    fig.set_figwidth(18)

    ax[0].plot(fpr, tpr, color='red',label='DistilBert')
    ax[0].plot(fpr2, tpr2, color='blue',label='XLNet')
    ax[0].legend()
    ax[0].plot([0,1], [0,1], color='black', linestyle='--')
    ax[0].set_title(f"Hate ROC Cruve & AUC");
    ax[0].set_xlabel(f"DistilBert AUC: {auc_dl_hate}\n XLNet AUC: {auc_xl_hate}")

    ax[1].plot(fpr3, tpr3, color='red',label='DistilBert')
    ax[1].plot(fpr4, tpr4, color='blue',label='XLNet')
    ax[1].legend()
    ax[1].plot([0,1], [0,1], color='black', linestyle='--')
    ax[1].set_title(f"Irony ROC Cruve & AUC");
    ax[1].set_xlabel(f"DistilBert AUC: {auc_dl_irony}\n XLNet AUC: {auc_xl_irony}")

    ax[2].plot(fpr5, tpr5, color='red',label='DistilBert')
    ax[2].plot(fpr6, tpr6, color='blue',label='XLNet')
    ax[2].legend()
    ax[2].plot([0,1], [0,1], color='black', linestyle='--')
    ax[2].set_title(f"Irony ROC Cruve & AUC");
    ax[2].set_xlabel(f"DistilBert AUC: {auc_dl_offensive}\n XLNet AUC: {auc_xl_offensive}")

