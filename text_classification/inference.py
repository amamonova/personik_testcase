from joblib import load
import logging
from sys import stdin

import fasttext.util

logger = logging.getLogger()


def read_model(path: str):
    clf = load(path)
    return clf


def load_fasttext(model_path: str):
    ft = fasttext.load_model(model_path)
    print('FastText is loaded.')
    logger.info('FastText is loaded.')
    return ft


def process(ft, clf):
    labels_dict = {0: 'VACATION-REQUEST', 1: 'SALARY-REQUEST',
                   2: 'SICK-LEAVE-REPORT', 3: 'OTHER'}
    for line in stdin:
        sent = ft.get_sentence_vector(line.strip())
        pred = clf.predict([sent])
        print(labels_dict[pred[0]])


if __name__ == '__main__':
    clf = read_model('fasttext_mlpclassifier.joblib')
    ft = load_fasttext('data/cc.ru.300.bin')
    process(ft, clf)
