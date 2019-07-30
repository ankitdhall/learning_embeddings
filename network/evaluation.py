from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import os
import numpy as np
from network.summarize import Summarize

import torch


class Evaluation:

    def __init__(self, experiment_directory, classes, generate_plots=False):
        self.classes = classes
        self.generate_plots = generate_plots
        self.make_dir_if_non_existent(experiment_directory)
        self.experiment_directory = experiment_directory
        self.summarizer = None

    def enable_plotting(self):
        self.generate_plots = True

    def disable_plotting(self):
        self.generate_plots = False

    @staticmethod
    def make_dir_if_non_existent(dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def evaluate(self, predicted_scores, correct_labels, epoch, phase, save_to_tensorboard, samples_split):
        if phase in ['val', 'test']:
            self.make_dir_if_non_existent(os.path.join(self.experiment_directory, 'stats',
                                                       ('best_' if not save_to_tensorboard else '') + phase + str(epoch)))
            self.summarizer = Summarize(os.path.join(self.experiment_directory, 'stats',
                                                     ('best_' if not save_to_tensorboard else '') + phase + str(epoch)))
            self.summarizer.make_heading('Classification Summary - Epoch {} {}'.format(epoch, phase), 1)

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
        print('-' * 30)
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
            return (p * r) * 2 / (p + r + 1e-6)

        # calculate metrics for different values of thresholds
        for class_index, class_name in enumerate(self.classes):
            precision[class_name], recall[class_name], thresholds[class_name] = precision_recall_curve(
                correct_labels == class_index, predicted_scores[:, class_index])
            f1[class_name] = get_f1score(precision[class_name], recall[class_name])

            average_precision[class_name] = average_precision_score(correct_labels == class_index,
                                                                    predicted_scores[:, class_index])

            if phase in ['val', 'test'] and self.generate_plots:
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

        mAP = sum([average_precision[class_name] for class_name in self.classes]) / len(average_precision)
        print('Mean average precision is {}'.format(mAP))

        if phase in ['val', 'test']:
            # make table with global metrics
            self.summarizer.make_heading('Class-wise Metrics', 2)
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


class Metrics:
    def __init__(self, predicted_labels, correct_labels):
        self.predicted_labels = predicted_labels
        self.correct_labels = correct_labels
        self.n_labels = correct_labels.shape[1]
        self.precision = dict()
        self.recall = dict()
        self.f1 = dict()
        self.accuracy = dict()

        self.cmat = dict()

        self.thresholds = dict()
        self.average_precision = dict()

        self.top_f1_score = dict()

        self.macro_scores = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        self.micro_scores = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        self.accuracy_score = 0.0

    def calculate_basic_metrics(self, list_of_indices):

        for label_ix in list_of_indices:

            self.precision[label_ix] = precision_score(self.correct_labels[:, label_ix],
                                                       self.predicted_labels[:, label_ix])
            self.recall[label_ix] = recall_score(self.correct_labels[:, label_ix],
                                                 self.predicted_labels[:, label_ix])
            self.f1[label_ix] = f1_score(self.correct_labels[:, label_ix],
                                         self.predicted_labels[:, label_ix])
            self.cmat[label_ix] = confusion_matrix(self.correct_labels[:, label_ix],
                                                   self.predicted_labels[:, label_ix])

            self.accuracy[label_ix] = self.predicted_labels[:, label_ix][np.where(self.correct_labels[:, label_ix] == 1)].mean()

        for metric in ['precision', 'recall', 'f1']:
            self.macro_scores[metric] = 1.0 * sum([getattr(self, metric)[label_ix]
                                                   for label_ix in list_of_indices]) / len(list_of_indices)
        combined_cmat = np.array([[0, 0], [0, 0]])
        temp = 0
        for label_ix in list_of_indices:
            temp += self.cmat[label_ix][0][0]
            combined_cmat += self.cmat[label_ix]

        self.micro_scores['precision'] = 1.0 * combined_cmat[1][1] / (combined_cmat[1][1] + combined_cmat[0][1])
        self.micro_scores['recall'] = 1.0 * combined_cmat[1][1] / (combined_cmat[1][1] + combined_cmat[1][0])
        self.micro_scores['f1'] = 2 * self.micro_scores['precision'] * self.micro_scores['recall'] / (
                self.micro_scores['precision'] + self.micro_scores['recall'])
        self.accuracy_score = self.predicted_labels[:, list_of_indices][np.where(self.correct_labels[:, list_of_indices] == 1)].mean()

        return {'macro': self.macro_scores, 'micro': self.micro_scores, 'precision': self.precision,
                'recall': self.recall, 'f1': self.f1, 'cmat': self.cmat, 'accuracy': self.accuracy,
                'accuracy_score': self.accuracy_score}


class MultiLabelEvaluation(Evaluation):

    def __init__(self, experiment_directory, labelmap, optimal_thresholds=None, generate_plots=False):
        Evaluation.__init__(self, experiment_directory, labelmap.classes, generate_plots)
        self.labelmap = labelmap
        self.predicted_labels = None
        if optimal_thresholds is None:
            self.optimal_thresholds = np.zeros(self.labelmap.n_classes)
        else:
            self.optimal_thresholds = optimal_thresholds

    def evaluate(self, predicted_scores, correct_labels, epoch, phase, save_to_tensorboard, samples_split):
        if phase in ['val', 'test']:
            self.make_dir_if_non_existent(os.path.join(self.experiment_directory, 'stats',
                                                       ('best_' if not save_to_tensorboard else '') + phase + str(epoch)))
            self.summarizer = Summarize(os.path.join(self.experiment_directory, 'stats',
                                                     ('best_' if not save_to_tensorboard else '') + phase + str(epoch)))
            self.summarizer.make_heading('Classification Summary - Epoch {} {}'.format(epoch, phase), 1)

        mAP, precision, recall, average_precision, thresholds = self.make_curves(predicted_scores,
                                                                                 correct_labels, epoch, phase)

        self.predicted_labels = predicted_scores >= np.tile(self.optimal_thresholds, (correct_labels.shape[0], 1))

        classes_predicted_per_sample = np.sum(self.predicted_labels, axis=1)
        print("Max: {}".format(np.max(classes_predicted_per_sample)))
        print("Min: {}".format(np.min(classes_predicted_per_sample)))
        print("Mean: {}".format(np.mean(classes_predicted_per_sample)))
        print("std: {}".format(np.std(classes_predicted_per_sample)))

        level_stop, level_start = [], []
        for level_id, level_len in enumerate(self.labelmap.levels):
            if level_id == 0:
                level_start.append(0)
                level_stop.append(level_len)
            else:
                level_start.append(level_stop[level_id - 1])
                level_stop.append(level_stop[level_id - 1] + level_len)

        level_wise_metrics = {}

        metrics_calculator = Metrics(self.predicted_labels, correct_labels)
        global_metrics = metrics_calculator.calculate_basic_metrics(list(range(0, self.labelmap.n_classes)))

        for level_id in range(len(level_start)):
            metrics_calculator = MetricsMultiLevel(self.predicted_labels, correct_labels)
            level_wise_metrics[self.labelmap.level_names[level_id]] = metrics_calculator.calculate_basic_metrics(
                list(range(level_start[level_id], level_stop[level_id]))
            )

        if phase in ['val', 'test']:
            # global metrics
            self.summarizer.make_heading('Global Metrics', 2)
            tabular_data = [[global_metrics[score_type][score] for score in ['precision', 'recall', 'f1']] for score_type in
                      ['macro', 'micro']]
            tabular_data[0].append(global_metrics['accuracy_score'])
            tabular_data[1].append(global_metrics['accuracy_score'])
            self.summarizer.make_table(
                data=tabular_data,
                x_labels=['Precision', 'Recall', 'F1', 'Accuracy'], y_labels=['Macro', 'Micro'])

            self.summarizer.make_heading('Class-wise Metrics', 2)
            self.summarizer.make_table(
                data=[[global_metrics['precision'][label_ix], global_metrics['recall'][label_ix],
                       global_metrics['f1'][label_ix], global_metrics['accuracy'][label_ix], samples_split['train'][label_ix], samples_split['val'][label_ix],
                       samples_split['test'][label_ix]] for label_ix in range(self.labelmap.n_classes)],
                x_labels=['Precision', 'Recall', 'F1', 'Acc', 'train freq', 'val freq', 'test freq'],
                y_labels=self.labelmap.classes)

            # level wise metrics
            for level_id, metrics_key in enumerate(level_wise_metrics):
                metrics = level_wise_metrics[metrics_key]
                tabular_data_lw = [[metrics[score_type][score] for score in ['precision', 'recall', 'f1']] for score_type in
                 ['macro', 'micro']]
                tabular_data_lw[0].append(metrics['accuracy_score'])
                tabular_data_lw[1].append(metrics['accuracy_score'])
                self.summarizer.make_heading('{} Metrics'.format(metrics_key), 2)
                self.summarizer.make_table(
                    data=tabular_data_lw,
                    x_labels=['Precision', 'Recall', 'F1', 'Accuracy'], y_labels=['Macro', 'Micro'])

                self.summarizer.make_heading('Class-wise Metrics', 2)
                self.summarizer.make_table(
                    data=[[global_metrics['precision'][label_ix], global_metrics['recall'][label_ix],
                           global_metrics['f1'][label_ix], global_metrics['accuracy'][label_ix],
                           global_metrics['cmat'][label_ix].ravel()[0], global_metrics['cmat'][label_ix].ravel()[1],
                           global_metrics['cmat'][label_ix].ravel()[2], global_metrics['cmat'][label_ix].ravel()[3],
                           int(samples_split['train'][label_ix]),
                           int(samples_split['val'][label_ix]), int(samples_split['test'][label_ix])]
                          for label_ix in range(level_start[level_id], level_stop[level_id])],
                    x_labels=['Precision', 'Recall', 'F1', 'Acc', 'tn', 'fp', 'fn', 'tp', 'train freq', 'val freq', 'test freq'],
                    y_labels=self.labelmap.classes[level_start[level_id]:level_stop[level_id]])

                if self.generate_plots:
                    score_vs_freq = [(global_metrics['f1'][label_ix], int(samples_split['train'][label_ix]))
                                     for label_ix in range(level_start[level_id], level_stop[level_id])]
                    self.make_score_vs_freq_hist(score_vs_freq,
                                                 os.path.join(self.experiment_directory, 'stats',
                                                              ('best_' if not save_to_tensorboard else '') + phase +
                                                              str(epoch)),
                                                 '{} {}'.format(self.labelmap.level_names[level_id], 'F1'))

        return global_metrics

    def make_score_vs_freq_hist(self, score_vs_freq, path_to_save, plot_title):
        x = np.array([sf[1] for sf in score_vs_freq])
        y = np.array([sf[0] for sf in score_vs_freq])

        nullfmt = NullFormatter()  # no labels

        # definitions for the axes
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        bottom_h = left_h = left + width + 0.05

        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom_h, width, 0.17]
        rect_histy = [left_h, bottom, 0.17, height]

        # start with a rectangular Figure
        plt.figure(1, figsize=(8, 8))

        axScatter = plt.axes(rect_scatter)
        axHistx = plt.axes(rect_histx)
        axHisty = plt.axes(rect_histy)

        # axHistx.get_xaxis().set_major_formatter(plt.NullFormatter())

        # the scatter plot:
        axScatter.scatter(x, y)
        axScatter.set_xscale('log')
        axScatter.set_xlabel('Training data size')
        axScatter.set_ylabel('Score')
        axScatter.set_ylim((0.0, 1.0))

        axHistx.set_xscale('log')
        axHistx.set_yscale('linear')

        axHisty.set_yscale('linear')
        axHisty.set_xscale('linear')

        # bins = np.arange(-lim, lim + binwidth, binwidth)
        _, bins = np.histogram(np.log10(x + 1), bins=50)
        axHistx.hist(x, bins=10 ** bins)
        axHisty.hist(y, bins=50, orientation='horizontal')

        axHistx.set_xlim(axScatter.get_xlim())
        axHisty.set_ylim(axScatter.get_ylim())

        # no labels
        # axHistx.xaxis.set_major_formatter(nullfmt)
        # axHisty.yaxis.set_major_formatter(nullfmt)

        save_fig_to = os.path.join(path_to_save, '{}_performance_vs_frequency.pdf'.format(plot_title))
        plt.savefig(save_fig_to, format='pdf')
        plt.clf()

    def get_optimal_thresholds(self):
        return self.optimal_thresholds

    def set_optimal_thresholds(self, best_f1_score):
        for class_ix, class_name in enumerate(self.labelmap.classes):
            self.optimal_thresholds[class_ix] = best_f1_score[class_name]['best_thresh']

    @staticmethod
    def get_f1score(p, r):
        p, r = np.array(p), np.array(r)
        return (p * r) * 2 / (p + r + 1e-6)

    def calculate_metrics(self, precision, recall, f1, average_precision, top_f1_score, thresholds, class_name, phase,
                          correct_labels, predicted_scores, epoch):

        precision[class_name], recall[class_name], thresholds[class_name] = precision_recall_curve(
            correct_labels, predicted_scores)
        f1[class_name] = self.get_f1score(precision[class_name], recall[class_name])

        average_precision[class_name] = average_precision_score(correct_labels, predicted_scores)

        if phase in ['val']:
            best_f1_ix = np.argmax(f1[class_name])
            best_thresh = thresholds[class_name][best_f1_ix]
            top_f1_score[class_name] = {'best_thresh': best_thresh, 'f1_score@thresh': f1[class_name][best_f1_ix],
                                        'thresh_ix': best_f1_ix}

            if self.generate_plots:
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

        return precision, recall, f1, average_precision, thresholds, top_f1_score

    def make_curves(self, predicted_scores, correct_labels, epoch, phase):
        if phase in ['val', 'test']:
            self.summarizer.make_heading('Data Distribution', 2)
            self.summarizer.make_table([[int(np.sum(correct_labels[:, class_ix]))
                                         for class_ix in range(self.labelmap.n_classes)]],
                                       x_labels=self.labelmap.classes)

        print('-' * 30)
        print('Running evaluation for {} at epoch {}'.format(phase, epoch))
        assert predicted_scores.shape[0] == correct_labels.shape[0], \
            'Number of predictions ({}) and labels ({}) do not match'.format(predicted_scores.shape[0],
                                                                             correct_labels.shape[0])

        precision = dict()
        recall = dict()
        thresholds = dict()
        average_precision = dict()
        f1 = dict()
        top_f1_score = dict()

        # calculate metrics for different values of thresholds
        for class_index, class_name in enumerate(self.classes):
            precision, recall, f1, average_precision, thresholds, top_f1_score = self.calculate_metrics(precision, recall, f1, average_precision, top_f1_score, thresholds, class_name, phase, correct_labels[:, class_index], predicted_scores[:, class_index], epoch)

        level_begin_ix = 0
        for level_ix, level_name in enumerate(self.labelmap.level_names):
            mAP = sum([average_precision[class_name]
                       for class_name in self.classes[level_begin_ix:level_begin_ix + self.labelmap.levels[level_ix]]]) \
                  / self.labelmap.levels[level_ix]
            level_begin_ix += self.labelmap.levels[level_ix]

        if phase in ['val']:
            self.make_table_with_metrics(mAP, precision, recall, top_f1_score, self.classes, correct_labels)
            self.set_optimal_thresholds(top_f1_score)

        return mAP, precision, recall, average_precision, thresholds

    def make_table_with_metrics(self, mAP, precision, recall, top_f1_score, class_names, correct_labels):
        # make table with global metrics
        self.summarizer.make_heading('Class-wise Metrics', 2)
        if mAP is not None:
            self.summarizer.make_text('Mean average precision is {}'.format(mAP))

        y_labels = [class_name for class_name in class_names]
        x_labels = ['Precision@BF1', 'Recall@BF1', 'Best f1-score', 'Best thresh', 'freq']
        data = []
        for class_index, class_name in enumerate(class_names):
            per_class_metrics = [precision[class_name][top_f1_score[class_name]['thresh_ix']],
                                 recall[class_name][top_f1_score[class_name]['thresh_ix']],
                                 top_f1_score[class_name]['f1_score@thresh'],
                                 top_f1_score[class_name]['best_thresh'],
                                 0 if correct_labels is None else int(np.sum(correct_labels[:, class_index]))]
            data.append(per_class_metrics)

        self.summarizer.make_table(data, x_labels, y_labels)


class MultiLabelEvaluationSingleThresh(MultiLabelEvaluation):

    def __init__(self, experiment_directory, labelmap, optimal_thresholds=None, generate_plots=False):
        MultiLabelEvaluation.__init__(self, experiment_directory, labelmap, optimal_thresholds, generate_plots)

    def make_curves(self, predicted_scores, correct_labels, epoch, phase):
        if phase in ['val', 'test']:
            self.summarizer.make_heading('Data Distribution', 2)
            self.summarizer.make_table([[int(np.sum(correct_labels[:, class_ix]))
                                         for class_ix in range(self.labelmap.n_classes)]],
                                       x_labels=self.labelmap.classes)

        print('-' * 30)
        print('Running evaluation for {} at epoch {}'.format(phase, epoch))
        assert predicted_scores.shape[0] == correct_labels.shape[0], \
            'Number of predictions ({}) and labels ({}) do not match'.format(predicted_scores.shape[0],
                                                                             correct_labels.shape[0])

        precision_c = dict()
        recall_c = dict()
        thresholds_c = dict()
        average_precision_c = dict()
        f1_c = dict()
        top_f1_score_c = dict()

        # calculate metrics for different values of thresholds
        precision_c, recall_c, f1_c, average_precision_c, thresholds_c, top_f1_score_c = \
            self.calculate_metrics(precision_c, recall_c, f1_c, average_precision_c, top_f1_score_c, thresholds_c,
                                   'all_classes', phase,
                                   correct_labels.flatten(), predicted_scores.flatten(), epoch)

        mAP_c = average_precision_c['all_classes']

        if phase in ['val']:
            self.make_table_with_metrics(mAP_c, precision_c, recall_c, top_f1_score_c, ['all_classes'], None)
            self.set_optimal_thresholds(top_f1_score_c)

        return mAP_c, precision_c, recall_c, average_precision_c, thresholds_c

    def set_optimal_thresholds(self, best_f1_score):
        for class_ix, class_name in enumerate(self.labelmap.classes):
            self.optimal_thresholds[class_ix] = best_f1_score['all_classes']['best_thresh']


class MetricsMultiLevel:
    def __init__(self, predicted_labels, correct_labels):
        self.predicted_labels = predicted_labels
        self.correct_labels = correct_labels
        self.n_labels = correct_labels.shape[1]
        self.precision = dict()
        self.recall = dict()
        self.f1 = dict()
        self.accuracy = dict()

        self.cmat = dict()

        self.thresholds = dict()
        self.average_precision = dict()

        self.top_f1_score = dict()

        self.macro_scores = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        self.micro_scores = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        self.accuracy_score = 0.0

    def calculate_basic_metrics(self, list_of_indices):

        for label_ix in list_of_indices:

            self.cmat[label_ix] = confusion_matrix(self.correct_labels[:, label_ix],
                                                   self.predicted_labels[:, label_ix])
            if self.cmat[label_ix].ravel().shape == (1, ):
                if np.all(np.array(self.predicted_labels[:, label_ix])):
                    self.cmat[label_ix] = np.array([[0, 0], [0, self.cmat[label_ix][0][0]]])
                else:
                    self.cmat[label_ix] = np.array([[self.cmat[label_ix][0][0], 0], [0, 0]])

            tn, fp, fn, tp = self.cmat[label_ix].ravel()
            if tp == 0 and fp == 0 and fn == 0:
                self.accuracy[label_ix] = self.predicted_labels[:, label_ix][np.where(self.correct_labels[:, label_ix] == 1)].mean() #(tp + tn) / (tp + tn + fp + fn)
                self.precision[label_ix] = 1.0
                self.recall[label_ix] = 1.0
                self.f1[label_ix] = 1.0
            elif tp == 0 and (fp > 0 or fn > 0):
                self.accuracy[label_ix] = self.predicted_labels[:, label_ix][np.where(self.correct_labels[:, label_ix] == 1)].mean()
                self.precision[label_ix] = 0.0
                self.recall[label_ix] = 0.0
                self.f1[label_ix] = 0.0
            else:
                self.accuracy[label_ix] = self.predicted_labels[:, label_ix][np.where(self.correct_labels[:, label_ix] == 1)].mean()
                self.precision[label_ix] = 1.0 * tp/(tp + fp)
                self.recall[label_ix] = 1.0 * tp/(tp + fn)
                self.f1[label_ix] = 2 * self.precision[label_ix] * self.recall[label_ix] /\
                                    (self.precision[label_ix] + self.recall[label_ix])

        for metric in ['precision', 'recall', 'f1']:
            self.macro_scores[metric] = 1.0 * sum([getattr(self, metric)[label_ix]
                                                   for label_ix in list_of_indices]) / len(list_of_indices)
        combined_cmat = np.array([[0, 0], [0, 0]])
        temp = 0
        for label_ix in list_of_indices:
            temp += self.cmat[label_ix][0][0]
            combined_cmat += self.cmat[label_ix]

        self.micro_scores['precision'] = 1.0 * combined_cmat[1][1] / (combined_cmat[1][1] + combined_cmat[0][1])
        self.micro_scores['recall'] = 1.0 * combined_cmat[1][1] / (combined_cmat[1][1] + combined_cmat[1][0])
        self.micro_scores['f1'] = 2 * self.micro_scores['precision'] * self.micro_scores['recall'] / (
                self.micro_scores['precision'] + self.micro_scores['recall'])
        self.accuracy_score = self.predicted_labels[:, list_of_indices][np.where(self.correct_labels[:, list_of_indices] == 1)].mean()

        return {'macro': self.macro_scores, 'micro': self.micro_scores, 'precision': self.precision,
                'recall': self.recall, 'f1': self.f1, 'cmat': self.cmat, 'accuracy': self.accuracy,
                'accuracy_score': self.accuracy_score}


class MultiLevelEvaluation(MultiLabelEvaluation):
    def __init__(self, experiment_directory, labelmap=None, generate_plots=False):
        MultiLabelEvaluation.__init__(self, experiment_directory, labelmap, None, generate_plots)
        self.labelmap = labelmap
        self.predicted_labels = None

    def evaluate(self, predicted_scores, correct_labels, epoch, phase, save_to_tensorboard, samples_split):
        if phase in ['val', 'test']:
            self.make_dir_if_non_existent(os.path.join(self.experiment_directory, 'stats',
                                                       ('best_' if not save_to_tensorboard else '') + phase + str(epoch)))
            self.summarizer = Summarize(os.path.join(self.experiment_directory, 'stats',
                                                     ('best_' if not save_to_tensorboard else '') + phase + str(epoch)))
            self.summarizer.make_heading('Classification Summary - Epoch {} {}'.format(epoch, phase), 1)

        predicted_scores = torch.from_numpy(predicted_scores)
        self.predicted_labels = np.zeros_like(predicted_scores)
        for level_id, level_name in enumerate(self.labelmap.level_names):
            start = sum([self.labelmap.levels[l_id] for l_id in range(level_id)])
            predicted_scores_part = predicted_scores[:, start:start+self.labelmap.levels[level_id]]
            # correct_labels_part = correct_labels[:, level_id]
            _, winning_indices = torch.max(predicted_scores_part, 1)

            self.predicted_labels[[row_ind for row_ind in range(winning_indices.shape[0])], winning_indices+start] = 1

        # mAP, precision, recall, average_precision, thresholds = self.make_curves(predicted_scores,
        #                                                                          correct_labels, epoch, phase)

        level_stop, level_start = [], []
        for level_id, level_len in enumerate(self.labelmap.levels):
            if level_id == 0:
                level_start.append(0)
                level_stop.append(level_len)
            else:
                level_start.append(level_stop[level_id-1])
                level_stop.append(level_stop[level_id-1] + level_len)

        level_wise_metrics = {}

        metrics_calculator = MetricsMultiLevel(self.predicted_labels, correct_labels)
        global_metrics = metrics_calculator.calculate_basic_metrics(list(range(0, self.labelmap.n_classes)))

        for level_id in range(len(level_start)):
            metrics_calculator = MetricsMultiLevel(self.predicted_labels, correct_labels)
            level_wise_metrics[self.labelmap.level_names[level_id]] = metrics_calculator.calculate_basic_metrics(
                list(range(level_start[level_id], level_stop[level_id]))
            )

        if phase in ['val', 'test']:
            # global metrics
            self.summarizer.make_heading('Global Metrics', 2)
            tabular_data = [[global_metrics[score_type][score] for score in ['precision', 'recall', 'f1']] for
                            score_type in ['macro', 'micro']]
            tabular_data[0].append(global_metrics['accuracy_score'])
            tabular_data[1].append(global_metrics['accuracy_score'])
            self.summarizer.make_table(
                data=tabular_data,
                x_labels=['Precision', 'Recall', 'F1', 'Accuracy'], y_labels=['Macro', 'Micro'])

            self.summarizer.make_heading('Class-wise Metrics', 2)
            self.summarizer.make_table(
                data=[[global_metrics['precision'][label_ix], global_metrics['recall'][label_ix],
                       global_metrics['f1'][label_ix], global_metrics['accuracy'][label_ix], int(samples_split['train'][label_ix]),
                       int(samples_split['val'][label_ix]), int(samples_split['test'][label_ix])]
                      for label_ix in range(self.labelmap.n_classes)],
                x_labels=['Precision', 'Recall', 'F1', 'Accuracy', 'train freq', 'val freq', 'test freq'],
                y_labels=self.labelmap.classes)

            # level wise metrics
            for level_id, metrics_key in enumerate(level_wise_metrics):
                metrics = level_wise_metrics[metrics_key]
                tabular_data_lw = [[metrics[score_type][score] for score in ['precision', 'recall', 'f1']] for
                                   score_type in
                                   ['macro', 'micro']]
                tabular_data_lw[0].append(metrics['accuracy_score'])
                tabular_data_lw[1].append(metrics['accuracy_score'])
                self.summarizer.make_heading('{} Metrics'.format(metrics_key), 2)
                self.summarizer.make_table(
                    data=tabular_data_lw,
                    x_labels=['Precision', 'Recall', 'F1', 'Accuracy'], y_labels=['Macro', 'Micro'])

                self.summarizer.make_heading('Class-wise Metrics', 2)
                self.summarizer.make_table(
                    data=[[global_metrics['precision'][label_ix], global_metrics['recall'][label_ix],
                           global_metrics['f1'][label_ix], global_metrics['accuracy'][label_ix],
                           global_metrics['cmat'][label_ix].ravel()[0], global_metrics['cmat'][label_ix].ravel()[1],
                           global_metrics['cmat'][label_ix].ravel()[2], global_metrics['cmat'][label_ix].ravel()[3],
                           int(samples_split['train'][label_ix]),
                           int(samples_split['val'][label_ix]), int(samples_split['test'][label_ix])]
                          for label_ix in range(level_start[level_id], level_stop[level_id])],
                    x_labels=['Precision', 'Recall', 'F1', 'Acc', 'tn', 'fp', 'fn', 'tp', 'train freq', 'val freq',
                              'test freq'],
                    y_labels=self.labelmap.classes[level_start[level_id]:level_stop[level_id]])

                if self.generate_plots:
                    score_vs_freq = [(global_metrics['f1'][label_ix], int(samples_split['train'][label_ix]))
                                     for label_ix in range(level_start[level_id], level_stop[level_id])]
                    self.make_score_vs_freq_hist(score_vs_freq,
                                                 os.path.join(self.experiment_directory, 'stats',
                                                              ('best_' if not save_to_tensorboard else '') + phase +
                                                              str(epoch)),
                                                 '{} {}'.format(self.labelmap.level_names[level_id], 'F1'))

        return global_metrics

    def set_optimal_thresholds(self, best_f1_score):
        pass
