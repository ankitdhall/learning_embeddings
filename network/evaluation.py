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
                self.plot_prec_recall_vs_thresh(precision[class_name], recall[class_name],
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
    def plot_prec_recall_vs_thresh(precisions, recalls, thresholds, f1_score, class_name):
        plt.plot(thresholds, precisions[:-1], 'b:', label='precision')
        plt.plot(thresholds, recalls[:-1], 'r:', label='recall')
        plt.plot(thresholds, f1_score[:-1], 'g:', label='f1-score')
        plt.xlabel('Threshold')
        plt.legend(loc='upper left')
        plt.title('Precision and recall vs. threshold for {}'.format(class_name))
        plt.ylim([0, 1])


class MLEvaluation(Evaluation):

    def __init__(self, experiment_directory, labelmap, optimal_thresholds):
        Evaluation.__init__(self, experiment_directory, labelmap.classes)
        self.optimal_thresholds = optimal_thresholds
        self.labelmap = labelmap

    def get_optimal_thresholds(self):
        return self.optimal_thresholds

    def set_optimal_thresholds(self, best_f1_score):
        for class_ix, class_name in enumerate(self.labelmap.classes):
            self.optimal_thresholds[class_ix] = best_f1_score[class_name]['best_thresh']

    def make_curves(self, predicted_scores, correct_labels, epoch, phase):
        if phase == 'val':
            self.summarizer.make_heading('Data Distribution', 2)
            self.summarizer.make_table([[int(np.sum(correct_labels[:, class_ix]))
                                         for class_ix in range(self.labelmap.n_classes)]],
                                       x_labels=self.labelmap.classes)

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
        top_f1_score = dict()

        def get_f1score(p, r):
            p, r = np.array(p), np.array(r)
            return (p * r)*2/(p + r + 1e-6)

        # calculate metrics for different values of thresholds
        for class_index, class_name in enumerate(self.classes):
            precision[class_name], recall[class_name], thresholds[class_name] = precision_recall_curve(
                correct_labels[:, class_index], predicted_scores[:, class_index])
            f1[class_name] = get_f1score(precision[class_name], recall[class_name])

            average_precision[class_name] = average_precision_score(correct_labels[:, class_index],
                                                                    predicted_scores[:, class_index])

            if phase == 'val':
                best_f1_ix = np.argmax(f1[class_name])
                best_thresh = thresholds[class_name][best_f1_ix]
                top_f1_score[class_name] = {'best_thresh': best_thresh, 'f1_score@thresh': f1[class_name][best_f1_ix],
                                            'thresh_ix': best_f1_ix}

                self.plot_prec_recall_vs_thresh(precision[class_name], recall[class_name],
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

        level_begin_ix = 0
        for level_ix, level_name in enumerate(self.labelmap.level_names):
            mAP = sum([average_precision[class_name]
                       for class_name in self.classes[level_begin_ix:level_begin_ix+self.labelmap.levels[level_ix]]]) \
                  / self.labelmap.levels[level_ix]
            print('Mean average precision for {} is {}'.format(level_name, mAP))
            level_begin_ix += self.labelmap.levels[level_ix]

        if phase == 'val':
            # make table with global metrics
            self.summarizer.make_heading('Global Metrics', 2)
            self.summarizer.make_text('Mean average precision is {}'.format(mAP))

            y_labels = [class_name for class_name in self.classes]
            x_labels = ['Average Precision (across thresholds)', 'Precision@BF1', 'Recall@BF1', 'Best f1-score', 'Best thresh']
            data = []
            for class_index, class_name in enumerate(self.classes):
                per_class_metrics = [average_precision[class_name],
                                     precision[class_name][top_f1_score[class_name]['thresh_ix']],
                                     recall[class_name][top_f1_score[class_name]['thresh_ix']],
                                     top_f1_score[class_name]['f1_score@thresh'],
                                     top_f1_score[class_name]['best_thresh']]
                data.append(per_class_metrics)

            self.set_optimal_thresholds(top_f1_score)

            self.summarizer.make_table(data, x_labels, y_labels)

        return mAP, precision, recall, average_precision, thresholds
