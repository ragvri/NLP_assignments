from keras.callbacks import Callback
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import classification_report
import numpy as np


class Metrics(Callback):

    def __init__(self, weights_file, verbose, logs_file, X_test=None, y_test=None, test_generator=None):
        self.verbose = verbose
        self.weights_file = weights_file
        self.logs_file = logs_file
        self.X_test = X_test
        self.y_test = y_test
        self.test_generator = test_generator

    def on_train_begin(self, logs={}):
        if self.verbose:
            print(f'Logs file is :{self.logs_file}')
        if self.verbose:
            print(f'weights file is :{self.weights_file}')
        self.max_f1 = 0

    def on_train_end(self, logs={}):
        self.model.load_weights(self.weights_file)
        val_predict = np.argmax(np.asarray(
            self.model.predict(self.validation_data[0])), axis=1)
        val_targ = np.argmax(self.validation_data[1], axis=1)

        _val_precision, _val_recall, _val_f1, _ = precision_recall_fscore_support(
            val_targ, val_predict, average='macro')
        _val_accuracy = accuracy_score(val_targ, val_predict)
        _val_accuracy, _val_f1, _val_precision, _val_recall = round(_val_accuracy, 3), round(
            _val_f1, 3), round(_val_precision, 3), round(_val_recall, 3)

        to_write = f'Finished training. Best values are:\nval_accuracy:{_val_accuracy} val_recall:{_val_recall} val_precision:{_val_precision} val_f1:{_val_f1}\nReport:\n {classification_report(val_targ, val_predict)}'

        if self.verbose:
            print(f'Training ended')
            print(to_write)
        with open(self.logs_file, 'a') as f:
            f.write(to_write)

    def on_epoch_end(self, epoch, logs={}):  # epoch is auto incrementing
        val_predict = np.argmax((np.asarray(self.model.predict(
            self.validation_data[0]))), axis=1)
        val_targ = np.argmax(self.validation_data[1], axis=1)

        _val_precision, _val_recall, _val_f1, _ = precision_recall_fscore_support(
            val_targ, val_predict, average='macro')
        _val_accuracy = accuracy_score(val_targ, val_predict)
        _val_accuracy, _val_f1, _val_precision, _val_recall = round(_val_accuracy, 3), round(
            _val_f1, 3), round(_val_precision, 3), round(_val_recall, 3)
        if _val_f1 > self.max_f1:
            self.max_f1 = _val_f1
            self.model.save_weights(self.weights_file)
        if self.verbose:
            print(f'Saved')
        if self.verbose:
            print(
                f'val_recall:{_val_recall} val_precision:{_val_precision} val_f1:{_val_f1} val_accuracy:{_val_accuracy}')
            print(f'{classification_report(val_targ,val_predict)}')
        return
