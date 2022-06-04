from ast import literal_eval
import pandas as pd
import matplotlib.pyplot as plt
import squarify
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from neptune.new.types import File

class visualize_data():
    def __init__(self, train_filename, val_filename, test_filename, run=None):
        self.data_dir = "../data"
        if run:
            self.run = run
        else:
            self.run = None

        # train, validation, test = self.read_data(train_filename, val_filename, test_filename)
        train = self.read_data(self.data_dir + train_filename)
        validation = self.read_data(self.data_dir + val_filename)
        test = pd.read_csv(self.data_dir + test_filename, sep='\t')

        # TODO - didn't load test yet
        # TODO - add formating (labels, title, legend) for all graphs
        self.train_split = self.split_multiple_tag_rows(train)
        self.train_counts = self.train_split.groupby(["tags"]).count().reset_index()

        self.validation_split = self.split_multiple_tag_rows(validation)
        self.test_counts = self.validation_split.groupby(["tags"]).count().reset_index()

    def read_data(self, filename):
        data = pd.read_csv(filename, sep='\t')
        data['tags'] = data['tags'].apply(literal_eval)
        return data

    # def read_data(self, train_f, val_f, test_f):
    #     train = pd.read_csv(self.data_dir + '/raw/train/train.tsv')
    #     validation = pd.read_csv(self.data_dir + '/raw/eval/validation.tsv')
    #     test = pd.read_csv(self.data_dir + '/raw/eval/test.tsv', sep='\t')
    #     return train, validation, test


    def split_multiple_tag_rows(self, data) -> pd.DataFrame:
        titlesList = []
        tagsList = []

        for index, row in data.iterrows():
            #     import pdb; pdb.set_trace()
            for tag in row["tags"]:
                titlesList.append(row["title"])
                tagsList.append(tag)
        return pd.DataFrame({"title": titlesList, "tags": tagsList})


    def counts_histogram(self, counts):
        fig = plt.figure(figsize=(7,4), dpi=100)
        sns.histplot(counts["title"].tolist())
        plt.grid()

        if self.run:
            self.run["dataset/histogram"] = File.as_html(fig)

        plt.show()

    def labels_treemap(self, counts):
        fig = plt.figure(figsize=(7,4), dpi=100)
        squarify.plot(sizes=counts['title'], label=counts['tags'], alpha=.8, figure=fig)
        plt.axis('off')

        if self.run:
            self.run["dataset/treemap"] = File.as_html(fig)

        plt.show()
        return fig

    def compute_confusion_matrix(self, y_true, y_pred, name):
        y_true_new = np.apply_along_axis(' '.join, 1, np.array(y_true,dtype=str))
        y_pred_new = np.apply_along_axis(' '.join, 1, np.array(y_pred,dtype=str))
        cm = confusion_matrix(y_true_new, y_pred_new)
        plt.savefig(name + ".png")
        plt.show()

    def compute_class_accuracy(self, y_true, y_pred, mode, mlb=None, name=None):
        y_pred_sq, y_true_sq = self.squeeze_labels(y_pred, y_true)

        y_true_final = []
        y_pred_final = []

        for i in range(len(y_pred_sq)):
            #     for j in range(len(y_pred[0])):
            common = list(set(y_pred_sq[i]).intersection(y_true_sq[i]))
            in_true = list(set(y_true_sq[i]) - set(y_pred_sq[i]))
            in_pred = list(set(y_pred_sq[i]) - set(y_true_sq[i]))

            y_true_final += common + in_true + [-1]*len(in_pred)
            y_pred_final += common + [-1]*len(in_true) + in_pred

        mat = confusion_matrix(y_true_final, y_pred_final)
        # TODO - Save accuracies so that you used them in the next run to check difference
        accuracies = mat.diagonal()/mat.sum(axis=1)
        x_labels = mlb.classes

        # TODO - remove after code development
        # np.save("accuracies.npy", accuracies)
        # np.save("x_labels.npy", x_labels)
        # x_labels = np.load("x_labels.npy")
        # accuracies = np.load("accuracies.npy")

        fig = plt.figure(figsize=(7,4))
        x = accuracies[1:]
        y = np.sum(y_true, axis=0)

        plt.scatter(x, y)
        plt.xticks(fontsize=18)
        plt.xticks(fontsize=18)
        if name:
            plt.title(name)

        for i, txt in enumerate(x_labels):
            plt.annotate(txt, (x[i], y[i]), fontsize=14)

        # fig = plt.figure(figsize=(20,5))
        # x = accuracies[1:]
        # x = np.arange(len(y))
        # plt.xticks(x, x_labels)
        # plt.xticks(rotation=45)
        # plt.plot(x, y)

        if self.run:
            self.run["model/accuracies" + mode]= File.as_html(fig)

        plt.show()

    def squeeze_labels(self, y_pred, y_true):
        y_pred_ids = np.where(y_pred != 0)
        y_true_ids = np.where(y_true != 0)
        y_pred_squeezed = []
        y_true_squeezed = []
        for i in range(len(y_true_ids[0])):
            y_pred_squeezed.append(list(y_pred_ids[1][y_pred_ids[0]==i]))
            y_true_squeezed.append(list(y_true_ids[1][y_true_ids[0]==i]))
        return y_pred_squeezed, y_true_squeezed

    def find_different_labels(self, y_train=None, mlb=None, prev_file=None):
        if prev_file:
            prev_labels = np.load(prev_file)
        else:
            prev_labels = np.load("x_labels_old.npy")
        if mlb:
            curr_labels = mlb.classes
        else:
            curr_labels = np.load("x_labels.npy")

        added_labels = list(set(curr_labels) - set(prev_labels))

        if self.run:
            np.savetxt("new_tags.csv", np.asarray(added_labels), fmt='%s')
            self.run["dataset/new_tags"].upload("new_tags.csv")

        print("Added labels are:")
        print(added_labels)



# train_f = '/raw/train/train.tsv'
# validation_f = '/raw/eval/validation.tsv'
# test_f = '/raw/eval/test.tsv'
# visualize = visualize_data(train_f, validation_f, test_f)
# visualize.counts_histogram(visualize.train_counts)
# fig = visualize.labels_treemap(visualize.train_counts)
# print(fig)

