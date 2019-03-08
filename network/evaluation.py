from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
import os


class Evaluation:

    def __init__(self, experiment_directory, classes):
        self.classes = classes
        self.make_dir_if_non_existent(experiment_directory)
        self.experiment_directory = experiment_directory

    @staticmethod
    def make_dir_if_non_existent(dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def evaluate(self, predicted_scores, correct_labels, epoch, phase):
        assert predicted_scores.shape[0] == correct_labels.shape[0], \
            'Number of predicitions ({}) and labels ({}) do not match'.format(predicted_scores.shape[0],
                                                                              correct_labels.shape[0])
        precision = dict()
        recall = dict()
        thresholds = dict()
        average_precision = dict()

        for class_index, class_name in enumerate(self.classes):
            precision[class_name], recall[class_name], thresholds[class_name] = precision_recall_curve(
                correct_labels == class_index, predicted_scores[:, class_index])

            self.plot_prec_recall_vs_tresh(precision[class_name], recall[class_name],
                                           thresholds[class_name], class_name)
            self.make_dir_if_non_existent(os.path.join(self.experiment_directory, phase, class_name))
            plt.savefig(os.path.join(self.experiment_directory, phase, class_name,
                                     'prec_recall_{}_{}.png'.format(epoch, class_name)))
            plt.clf()

    @staticmethod
    def plot_prec_recall_vs_tresh(precisions, recalls, thresholds, class_name):
        plt.plot(thresholds, precisions[:-1], 'b:', label='precision')
        plt.plot(thresholds, recalls[:-1], 'r:', label='recall')
        plt.xlabel('Threshold')
        plt.legend(loc='upper left')
        plt.title('Precision and recall vs. threshold for {}'.format(class_name))
        plt.ylim([0, 1])
