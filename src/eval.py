from sklearn.metrics import accuracy_score, f1_score, average_precision_score
from sklearn.metrics import roc_auc_score as roc_auc
import numpy as np

from model import get_classifiers
from text_preprocessing import get_train_test_data
from log_to_neptune import visualize_data
from neptune.new.types import File

import neptune.new as neptune
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--data-dir', type=str, default='../data')
args = parser.parse_args()

def print_evaluation_scores(y_val, predicted, mode=None, run=None):
    acc = accuracy_score(y_val, predicted)
    f1 = f1_score(y_val, predicted, average='weighted')
    prec = average_precision_score(y_val, predicted, average='macro')
    print('Accuracy score: ', acc)
    print('F1 score: ', f1)
    print('Average precision score: ', prec)

    if mode and run:
        run["model/Accuracy "+ mode].log(acc)
        run["model/F1 "+ mode].log(f1)
        run["model/Precision "+ mode].log(prec)


def bag_of_words_tfidf_evaluation(run=None, visualize=None):

    solver = 'liblinear'
    random_seed = 42
    penalty = 'l1'

    if run:
        params = {"Algorithm": "Logistic Regression", "penalty": penalty, "solver": solver,
                  "seed": random_seed}
        run["parameters"] = params

    classifier_mybag, classifier_tfidf, y_train, y_val, mlb, tfidf_vectorizer, words_to_index, \
    dict_size = get_classifiers(args.data_dir)
    np.save("y_true.npy", y_val)
    X_train, _, X_val, _, X_test, X_train_mybag, X_val_mybag, X_test_mybag, X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vocab, _, words_to_index, dict_size = get_train_test_data(args.data_dir)
    y_val_predicted_labels_mybag = classifier_mybag.predict(X_val_mybag)
    y_val_predicted_scores_mybag = classifier_mybag.decision_function(X_val_mybag)
    np.save("y_pred.npy",y_val_predicted_labels_mybag)

    # TODO - remove after code development
    # y_true = np.load("y_true.npy")
    # y_val = np.load("y_pred.npy")

    y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
    y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)

    if visualize:
        # Compute the the various graphs for validation set
        visualize.compute_class_accuracy(y_val, y_val_predicted_labels_mybag, "mybag", mlb=mlb,
                                         name="Accuracy per class - Bag of words")
        visualize.compute_class_accuracy(y_val, y_val_predicted_labels_tfidf, "tfidf", mlb=mlb,
                                         name="Accuracy per class - TFIDF")
        # TODO - remove after code develpoment
        # visualize.compute_class_accuracy(y_true, y_val, mlb=mlb, name="Accuracy per class")
        # visualize.compute_class_accuracy(y_true, y_val, name="Accuracy per class")

    print("completed confusion_matrix_plots")

    print('Bag-of-words')
    print_evaluation_scores(y_val, y_val_predicted_labels_mybag, mode="bow", run=run)
    auc_bow = roc_auc(y_val, y_val_predicted_scores_mybag, multi_class='ovo')
    run["model/AUC Bag of words"].log(auc_bow)
    print("roc_auc: ", auc_bow)
    print('Tfidf')
    print_evaluation_scores(y_val, y_val_predicted_labels_tfidf, mode="tfidf", run=run)
    auc_tfidf = roc_auc(y_val, y_val_predicted_scores_tfidf, multi_class='ovo')
    run["model/AUC TFIDF"].log(auc_tfidf)
    print("roc_acu: ", auc_tfidf)

if __name__ == "__main__":
    run1 = neptune.init(
        project="kkrachtop/REMLA-project",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzODE5OTg5Yi03ZGY5LTRjOGQtOTMwNS1kMzg2NjdjNWNkNzQifQ==",
    )

    train_f = '/raw/train/train.tsv'
    validation_f = '/raw/eval/validation.tsv'
    test_f = '/raw/eval/test.tsv'
    visualize = visualize_data(train_f, validation_f, test_f, run=run1)

    bag_of_words_tfidf_evaluation(run=run1, visualize=visualize)

    visualize.find_different_labels()
    visualize.counts_histogram(visualize.train_counts)
    fig = visualize.labels_treemap(visualize.train_counts)