import logging
import sklearn.metrics as metrics


class Logger(object):
    def __init__(self,log_path):
        self.logger = logging.getLogger()
        self.formatter = logging.Formatter(fmt='[%(asctime)s]%(message)s',datefmt='%Y-%m-%d %H:%M:%S')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers=[]

        fh = logging.FileHandler(log_path, mode='w')
        fh.setLevel(logging.INFO)
        fh.setFormatter(self.formatter)
        self.logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(self.formatter)
        self.logger.addHandler(sh)

    def print(self, text):
        self.logger.info(text)


class Record(object):
    def __init__(self):
        self.accuracy  = 0.0
        self.macro_f1  = 0.0
        self.precision = []
        self.recall    = []
        self.f1_score  = []
    def update(self, accuracy, macro_f1, precision, recall, f1_score,**kwargs):
        self.accuracy  = accuracy
        self.macro_f1  = macro_f1
        self.precision = precision
        self.recall    = recall
        self.f1_score  = f1_score


def get_metrics(preds, labels):
    accuracy  = metrics.accuracy_score(labels, preds)
    macro_f1  = metrics.f1_score(labels, preds, average="macro")
    precision = metrics.precision_score(labels, preds, average=None)
    recall    = metrics.recall_score(labels, preds, average=None)
    f1_score  = metrics.f1_score(labels, preds, average=None)
    return {"accuracy":accuracy, "macro_f1":macro_f1, "precision":precision.tolist(),"recall":recall.tolist(),"f1_score":f1_score.tolist()}
    # return accuracy, macro_f1, precision.tolist(), recall.tolist(), f1_score.tolist()
