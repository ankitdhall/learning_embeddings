import os
import argparse
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class ConvertLog:
    """
    Export tensorboard logs as plots.
    """
    def __init__(self, path_to_log, path_to_save, experiment_list, metric_list, combine, val_only, legend_labels, title):
        """
        Constructor.
        :param path_to_log: <str> Location of the parent experiment directory
        :param path_to_save: <str> Parent directory to store the plots to
        :param experiment_list: <list <str> > List of experiments to convert
        :param metric_list: <list <str> > List of metrics to plot from tensorboard
        :param combine: <bool> If combine plots across different experiments
        :param val_only: <bool> If used, combines plots across experiments for val metrics only
        :param legend_labels: <list <str> > List of labels for the legend (one per experiment)
        :param title: <str> Title of the plot
        """
        self.path_to_log = path_to_log
        self.val_only = val_only
        self.metric_list = metric_list
        self.combine = combine
        self.title = title
        self.legend_labels = legend_labels
        self.path_to_save = os.path.join(path_to_save, 'plots')
        if not os.path.exists(self.path_to_save):
            os.makedirs(self.path_to_save)
        self.exp_list = experiment_list
        if self.exp_list is None:
            self.exp_list = os.listdir(self.path_to_log)
        print('== Will be converting the following experiments\n {}'.format(self.exp_list))
        if self.combine:
            print('== Will combine the plots')
        self.plot_window = None

    def convert_exp(self):
        """
        Convert the experiment
        :return: -
        """
        if self.combine:
            self.create_combined_train_val_plot()
        else:
            for exp in self.exp_list:
                tf_log = os.listdir(os.path.join(self.path_to_log, exp, 'tensorboard'))[0]
                tf_log = os.path.join(self.path_to_log, exp, 'tensorboard', tf_log)
                if not os.path.exists(os.path.join(self.path_to_save, exp)):
                    os.makedirs(os.path.join(self.path_to_save, exp))
                for metric in self.metric_list:
                    self.create_train_val_plot(metric, tf_log, exp)

    def create_combined_train_val_plot(self):
        if not os.path.exists(os.path.join(self.path_to_save, 'combined')):
            os.makedirs(os.path.join(self.path_to_save, 'combined'))
        for metric in self.metric_list:
            for exp_id, exp in enumerate(self.exp_list):
                tf_log = os.listdir(os.path.join(self.path_to_log, exp, 'tensorboard'))[0]
                tf_log = os.path.join(self.path_to_log, exp, 'tensorboard', tf_log)
                self.create_train_val_plot(metric, tf_log, exp, self.title,
                                           self.legend_labels[exp_id] if self.legend_labels else None)

            save_fig_to = os.path.join(self.path_to_save, 'combined', 'combined_train_val_{}.pdf'.format(metric))
            plt.savefig(save_fig_to, format='pdf')
            plt.clf()

    def create_train_val_plot(self, field, tf_log, exp, title=None, legend_label=None):
        """
        Create a plot with train and val.
        :param plt: Matplotlib plot
        :param field: <str> field to plot for train and val
        :param tf_log: <str> path to tf log
        :param exp: <str> name of experiment
        :param title: <str> plot title
        :param legend_label: <str> specify legend name
        :return:
        """
        train, val, epoch = [], [], []
        for e in tf.train.summary_iterator(tf_log):
            for v in e.summary.value:
                if v.tag == 'train_{}'.format(field):
                    train.append(v.simple_value)
                    epoch.append(e.step)
                if v.tag == 'val_{}'.format(field):
                    val.append(v.simple_value)

        # save as plot
        if not self.val_only:
            plt.plot(epoch, train, '--', label='train {} {}'.format(field, legend_label if legend_label else ''))
        plt.plot(epoch, val, '--', label='val {} {}'.format(field, legend_label if legend_label else ''))
        plt.xlabel('Epoch')
        plt.legend(loc='best')
        if title:
            plt.title(title)
        else:
            plt.title('train {0} vs. val {0}'.format(field))

        if not self.combine:
            save_fig_to = os.path.join(self.path_to_save, exp, '{}_train_val_{}.pdf'.format(exp, field))

            plt.savefig(save_fig_to, format='pdf')
            plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_logdir", help='Parent directory to experiment directory.', type=str,
                        default='/home/ankit/learning_embeddings/exp/multi_level_logs')
    parser.add_argument("--path_to_save", help='Parent directory to store plots.', type=str,
                        default='../')
    parser.add_argument("--experiment_list", help='Experiments to parse logs for.', nargs='*', default=None)
    parser.add_argument("--combine", help='If used, combines plots across experiments', action='store_true')
    parser.add_argument("--val_only", help='If used, combines plots across experiments for val metrics only', action='store_true')
    parser.add_argument("--legend_labels", help='List of labels for the legend', nargs='*', default=None)
    parser.add_argument("--title", help='List of labels for the legend', type=str, default=None)
    parser.add_argument("--metric_list", help='Metrics to plot.', nargs='*', default=['macro_f1', 'micro_f1', 'loss',
                                                                                      'micro_precision', 'micro_recall',
                                                                                      'macro_precision', 'macro_recall'])
    args = parser.parse_args()

    exp_list = ['ethec_full_resnet50_lr_0.01', 'ethec_single_thresh_full_resnet50_lr_0.01']
    clog = ConvertLog(path_to_log=args.path_to_logdir,
                      path_to_save=args.path_to_save,
                      experiment_list=args.experiment_list,
                      metric_list=args.metric_list,
                      combine=args.combine,
                      val_only=args.val_only,
                      legend_labels=args.legend_labels,
                      title=args.title)
    clog.convert_exp()
