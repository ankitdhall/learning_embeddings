from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib

matplotlib.use('Agg')
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
        if phase in ['val', 'test']:
            self.make_dir_if_non_existent(os.path.join(self.experiment_directory, 'stats', phase + str(epoch)))
            self.summarizer = Summarize(os.path.join(self.experiment_directory, 'stats', phase + str(epoch)))
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

            if phase in ['val', 'test']:
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

        self.cmat = dict()

        self.thresholds = dict()
        self.average_precision = dict()

        self.top_f1_score = dict()

        self.macro_scores = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        self.micro_scores = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    def calculate_basic_metrics(self):

        for label_ix in range(self.n_labels):
            self.precision[label_ix] = precision_score(self.correct_labels[:, label_ix],
                                                       self.predicted_labels[:, label_ix])
            self.recall[label_ix] = recall_score(self.correct_labels[:, label_ix],
                                                 self.predicted_labels[:, label_ix])
            self.f1[label_ix] = f1_score(self.correct_labels[:, label_ix],
                                         self.predicted_labels[:, label_ix])
            self.cmat[label_ix] = confusion_matrix(self.correct_labels[:, label_ix],
                                                   self.predicted_labels[:, label_ix])

        for metric in ['precision', 'recall', 'f1']:
            self.macro_scores[metric] = 1.0 * sum([getattr(self, metric)[label_ix]
                                                   for label_ix in range(self.n_labels)]) / self.n_labels
        combined_cmat = np.array([[0, 0], [0, 0]])
        temp = 0
        for label_ix in range(self.n_labels):
            temp += self.cmat[label_ix][0][0]
            combined_cmat += self.cmat[label_ix]

        self.micro_scores['precision'] = 1.0 * combined_cmat[1][1] / (combined_cmat[1][1] + combined_cmat[0][1])
        self.micro_scores['recall'] = 1.0 * combined_cmat[1][1] / (combined_cmat[1][1] + combined_cmat[1][0])
        self.micro_scores['f1'] = 2 * self.micro_scores['precision'] * self.micro_scores['recall'] / (
                self.micro_scores['precision'] + self.micro_scores['recall'])

        return {'macro': self.macro_scores, 'micro': self.micro_scores, 'precision': self.precision,
                'recall': self.recall, 'f1': self.f1, 'cmat': self.cmat}


class MLEvaluation(Evaluation):

    def __init__(self, experiment_directory, labelmap, optimal_thresholds=None):
        Evaluation.__init__(self, experiment_directory, labelmap.classes)
        self.labelmap = labelmap
        self.predicted_labels = None
        if optimal_thresholds is None:
            self.optimal_thresholds = np.zeros(self.labelmap.n_classes)
        else:
            self.optimal_thresholds = optimal_thresholds

    def evaluate(self, predicted_scores, correct_labels, epoch, phase):
        if phase in ['val', 'test']:
            self.make_dir_if_non_existent(os.path.join(self.experiment_directory, 'stats', phase + str(epoch)))
            self.summarizer = Summarize(os.path.join(self.experiment_directory, 'stats', phase + str(epoch)))
            self.summarizer.make_heading('Classification Summary - Epoch {} {}'.format(epoch, phase), 1)

        mAP, precision, recall, average_precision, thresholds = self.make_curves(predicted_scores,
                                                                                 correct_labels, epoch, phase)

        self.predicted_labels = predicted_scores >= np.tile(self.optimal_thresholds, (correct_labels.shape[0], 1))
        metrics_calculator = Metrics(self.predicted_labels, correct_labels)
        metrics = metrics_calculator.calculate_basic_metrics()

        if phase in ['val', 'test']:
            self.summarizer.make_heading('Global Metrics', 2)
            self.summarizer.make_table(
                data=[[metrics[score_type][score] for score in ['precision', 'recall', 'f1']] for score_type in
                      ['macro', 'micro']],
                x_labels=['Precision', 'Recall', 'F1'], y_labels=['Macro', 'Micro'])

        if phase == 'test':
            self.summarizer.make_heading('Class-wise Metrics', 2)
            self.summarizer.make_table(
                data=[[metrics[score][label_ix] for score in ['precision', 'recall', 'f1']] for label_ix in
                      range(self.labelmap.n_classes)],
                x_labels=['Precision', 'Recall', 'F1'], y_labels=self.labelmap.classes)

        return metrics['macro']['f1'], metrics['micro']['f1']

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
            self.make_table_with_metrics(mAP, precision, recall, top_f1_score, self.classes)
            self.set_optimal_thresholds(top_f1_score)

        return mAP, precision, recall, average_precision, thresholds

    def make_table_with_metrics(self, mAP, precision, recall, top_f1_score, class_names):
        # make table with global metrics
        self.summarizer.make_heading('Class-wise Metrics', 2)
        if mAP is not None:
            self.summarizer.make_text('Mean average precision is {}'.format(mAP))

        y_labels = [class_name for class_name in class_names]
        x_labels = ['Precision@BF1', 'Recall@BF1', 'Best f1-score', 'Best thresh']
        data = []
        for class_index, class_name in enumerate(class_names):
            per_class_metrics = [precision[class_name][top_f1_score[class_name]['thresh_ix']],
                                 recall[class_name][top_f1_score[class_name]['thresh_ix']],
                                 top_f1_score[class_name]['f1_score@thresh'],
                                 top_f1_score[class_name]['best_thresh']]
            data.append(per_class_metrics)

        self.summarizer.make_table(data, x_labels, y_labels)


class MLEvaluationSingleThresh(MLEvaluation):

    def __init__(self, experiment_directory, labelmap, optimal_thresholds=None):
        MLEvaluation.__init__(self, experiment_directory, labelmap, optimal_thresholds)

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
            self.make_table_with_metrics(mAP_c, precision_c, recall_c, top_f1_score_c, ['all_classes'])
            self.set_optimal_thresholds(top_f1_score_c)

        return mAP_c, precision_c, recall_c, average_precision_c, thresholds_c

    def set_optimal_thresholds(self, best_f1_score):
        for class_ix, class_name in enumerate(self.labelmap.classes):
            self.optimal_thresholds[class_ix] = best_f1_score['all_classes']['best_thresh']
