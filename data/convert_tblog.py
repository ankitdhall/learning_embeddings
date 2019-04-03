import os
import argparse
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class ConvertLog:
    def __init__(self, path_to_log, path_to_save, experiment_list, metric_list):
        self.path_to_log = path_to_log
        self.metric_list = metric_list
        self.path_to_save = os.path.join(path_to_save, 'plots')
        if not os.path.exists(self.path_to_save):
            os.makedirs(self.path_to_save)
        self.exp_list = experiment_list
        if self.exp_list is None:
            self.exp_list = os.listdir(self.path_to_log)
        print('Will be converting the following experiments\n {}'.format(self.exp_list))

    def convert_exp(self):
        for exp in self.exp_list:
            tf_log = os.listdir(os.path.join(self.path_to_log, exp, 'tensorboard'))[0]
            tf_log = os.path.join(self.path_to_log, exp, 'tensorboard', tf_log)
            if not os.path.exists(os.path.join(self.path_to_save, exp)):
                os.makedirs(os.path.join(self.path_to_save, exp))
            for metric in self.metric_list:
                self.create_train_val_plot(metric, tf_log, exp)

    def create_train_val_plot(self, field, tf_log, exp):
        train, val, epoch = [], [], []
        for e in tf.train.summary_iterator(tf_log):
            for v in e.summary.value:
                if v.tag == 'train_{}'.format(field):
                    train.append(v.simple_value)
                    epoch.append(e.step)
                if v.tag == 'val_{}'.format(field):
                    val.append(v.simple_value)

        # save as plot
        plt.plot(epoch, train, 'b:', label='train_{}'.format(field))
        plt.plot(epoch, val, 'r:', label='val_{}'.format(field))
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        plt.title('train {0} vs. val {0}'.format(field))

        save_fig_to = os.path.join(self.path_to_save, exp, '{}_train_val_{}.pdf'.format(exp, field))

        plt.savefig(save_fig_to, format='pdf')
        plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_logdir", help='Parent directory to experiment directory.', type=str,
                        default='/home/ankit/learning_embeddings/exp/')
    parser.add_argument("--path_to_save", help='Parent directory to store plots.', type=str,
                        default='../')
    parser.add_argument("--experiment_list", help='Experiments to parse logs for.', nargs='*', default=None)
    parser.add_argument("--metric_list", help='Metrics to plot.', nargs='*', default=['macro_f1', 'micro_f1', 'loss',
                                                                                      'micro_precision', 'micro_recall',
                                                                                      'macro_precision', 'macro_recall'])
    args = parser.parse_args()

    exp_list = ['ethec_full_resnet50_lr_0.01', 'ethec_single_thresh_full_resnet50_lr_0.01']
    clog = ConvertLog(path_to_log=args.path_to_logdir,
                      path_to_save=args.path_to_save,
                      experiment_list=args.experiment_list,
                      metric_list=args.metric_list)
    clog.convert_exp()
