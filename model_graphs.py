import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    classification_report
)
from sklearn.model_selection import learning_curve
import numpy as np

class ModelGraphs:
    def __init__(self, y_train, y_test, y_pred, y_pred_proba=None, save_dir='output', class_names=None):
        self.y_train = y_train
        self.y_test = y_test
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        self.save_dir = save_dir
        self.class_names = class_names if class_names is not None else np.unique(y_test)

        # Creare la cartella di salvataggio se non esiste
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def plot_confusion_matrix(self):
        cm = confusion_matrix(self.y_test, self.y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names, annot_kws={"size": 14})
        plt.title('Confusion Matrix', fontsize=16)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        save_path = os.path.join(self.save_dir, "confusion_matrix.png")
        plt.savefig(save_path)
        plt.close()

    def plot_roc_curve(self):
        if self.y_pred_proba is None:
            raise ValueError("y_pred_proba is required for ROC curve.")
        
        # Assumi che y_pred_proba abbia forma (n_samples, 2) e prendi la seconda colonna (probabilità della classe positiva)
        y_pred_proba_positive_class = self.y_pred_proba[:, 1]
        
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba_positive_class)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('Receiver Operating Characteristic', fontsize=16)
        plt.legend(loc="lower right", fontsize=12)
        save_path = os.path.join(self.save_dir, "roc_curve.png")
        plt.savefig(save_path)
        plt.close()

    def plot_precision_recall_curve(self):
        if self.y_pred_proba is None:
            raise ValueError("y_pred_proba is required for Precision-Recall curve.")
        
        # Assumi che y_pred_proba abbia forma (n_samples, 2) e prendi la seconda colonna (probabilità della classe positiva)
        y_pred_proba_positive_class = self.y_pred_proba[:, 1]
        
        precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba_positive_class)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.title('Precision-Recall Curve', fontsize=16)
        save_path = os.path.join(self.save_dir, "precision_recall_curve.png")
        plt.savefig(save_path)
        plt.close()

    def generate_classification_report(self):
        report = classification_report(self.y_test, self.y_pred, target_names=self.class_names)
        save_path = os.path.join(self.save_dir, "classification_report.txt")
        with open(save_path, "w") as f:
            f.write(report)

    def generate_all_reports(self):
        self.plot_confusion_matrix()
        if self.y_pred_proba is not None:
            self.plot_roc_curve()
            self.plot_precision_recall_curve()
        self.generate_classification_report()
