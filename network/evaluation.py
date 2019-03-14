from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os
import numpy as np
from summarize import Summarize

class Evaluation:

    def __init__(self, experiment_directory, classes):
        self.classes = classes
        self.make_dir_if_non_existent(experiment_directory)
        self.experiment_directory = experiment_directory
        self.summarizer = None

    @staticmethod
    def make_dir_if_non_existent(dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def evaluate(self, predicted_scores, correct_labels, epoch, phase):
        if phase == 'val':
            self.make_dir_if_non_existent(os.path.join(self.experiment_directory, 'stats', str(epoch)))
            self.summarizer = Summarize(os.path.join(self.experiment_directory, 'stats', str(epoch)))
            self.summarizer.make_heading('Classification Summary - Epoch {}'.format(epoch), 1)

        mAP, precision, recall, average_precision, thresholds = self.make_curves(predicted_scores,
                                                                                 correct_labels, epoch, phase)

        return mAP, precision, recall, average_precision, thresholds

    def top_k(self, predicted_scores, correct_labels, k=10):
        correct_labels_oh = np.zeros_like(predicted_scores)
        correct_labels_oh[np.arange(correct_labels.shape[0]), correct_labels] = 1

        for cid in range(predicted_scores.shape[1]):
            sorted_indices = np.argsort((correct_labels_oh[:, cid] - 0.5) * predicted_scores[:, cid])

            best_top_k = sorted_indices[:k]
            worst_top_k = sorted_indices[k:]

    def make_curves(self, predicted_scores, correct_labels, epoch, phase):
        print('-'*30)
        print('Running evaluation for {} at epoch {}'.format(phase, epoch))
        assert predicted_scores.shape[0] == correct_labels.shape[0], \
            'Number of predicitions ({}) and labels ({}) do not match'.format(predicted_scores.shape[0],
                                                                              correct_labels.shape[0])
        precision = dict()
        recall = dict()
        thresholds = dict()
        average_precision = dict()
        f1 = dict()

        def get_f1score(p, r):
            p, r = np.array(p), np.array(r)
            return (p * r)*2/(p + r + 1e-6)

        # calculate metrics for different values of thresholds
        for class_index, class_name in enumerate(self.classes):
            precision[class_name], recall[class_name], thresholds[class_name] = precision_recall_curve(
                correct_labels == class_index, predicted_scores[:, class_index])
            f1[class_name] = get_f1score(precision[class_name], recall[class_name])

            average_precision[class_name] = average_precision_score(correct_labels == class_index,
                                                                    predicted_scores[:, class_index])

            if phase == 'val':
                self.plot_prec_recall_vs_tresh(precision[class_name], recall[class_name],
                                               thresholds[class_name], f1[class_name],
                                               class_name)
                self.make_dir_if_non_existent(os.path.join(self.experiment_directory, phase, class_name))
                save_fig_to = os.path.join(self.experiment_directory, phase, class_name,
                                           'prec_recall_{}_{}.png'.format(epoch, class_name))
                plt.savefig(save_fig_to)
                plt.clf()
                self.summarizer.make_heading('Precision Recall `{}` ({})'.format(class_name, phase), 3)
                self.summarizer.make_image(save_fig_to, 'Precision Recall {}'.format(class_name))
            print('Average precision for {} is {}'.format(class_name, average_precision[class_name]))

        mAP = sum([average_precision[class_name] for class_name in self.classes])/len(average_precision)
        print('Mean average precision is {}'.format(mAP))

        if phase == 'val':
            # make table with global metrics
            self.summarizer.make_heading('Global Metrics', 2)
            self.summarizer.make_text('Mean average precision is {}'.format(mAP))

            x_labels = [class_name for class_name in self.classes]
            y_labels = ['Average Precision (across thresholds)', 'Precision', 'Recall', 'f1-score']
            data = [[average_precision[class_name] for class_name in self.classes]]
            c_wise_prec = precision_score(correct_labels, np.argmax(predicted_scores, axis=1), average=None)
            data.append(c_wise_prec.tolist())
            c_wise_rec = recall_score(correct_labels, np.argmax(predicted_scores, axis=1), average=None)
            data.append(c_wise_rec.tolist())
            c_wise_f1 = f1_score(correct_labels, np.argmax(predicted_scores, axis=1), average=None)
            data.append(c_wise_f1.tolist())

            self.summarizer.make_table(data, x_labels, y_labels)

        return mAP, precision, recall, average_precision, thresholds

    @staticmethod
    def plot_prec_recall_vs_tresh(precisions, recalls, thresholds, f1_score, class_name):
        plt.plot(thresholds, precisions[:-1], 'b:', label='precision')
        plt.plot(thresholds, recalls[:-1], 'r:', label='recall')
        plt.plot(thresholds, f1_score[:-1], 'g:', label='f1-score')
        plt.xlabel('Threshold')
        plt.legend(loc='upper left')
        plt.title('Precision and recall vs. threshold for {}'.format(class_name))
        plt.ylim([0, 1])
