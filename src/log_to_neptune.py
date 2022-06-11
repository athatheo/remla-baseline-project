import os
from ast import literal_eval
import pandas as pd
import matplotlib.pyplot as plt
import squarify
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from neptune.new.types import File
import plotly.graph_objects as go

class visualize_data():
    """
    Class that handles all visualizations for the Neptune dashboards
    """
    def __init__(self, train_filename, val_filename, test_filename, run=None):
        self.data_dir = "../data"
        self.hist_dir = "../data/raw/hist_files/"
        if run:
            self.run = run
        else:
            self.run = False

        # train, validation, test = self.read_data(train_filename, val_filename, test_filename)
        train = self.read_data(self.data_dir + train_filename)
        validation = self.read_data(self.data_dir + val_filename)
        test = pd.read_csv(self.data_dir + test_filename, sep='\t')

        # Split the data points with multiple tags
        self.train_split = self.split_multiple_tag_rows(train)
        self.train_counts = self.train_split.groupby(["tags"]).count().reset_index()
        self.train_counts.rename(columns={'title': 'occurrences'}, inplace=True)

        self.validation_split = self.split_multiple_tag_rows(validation)
        self.val_counts = self.validation_split.groupby(["tags"]).count().reset_index()
        self.val_counts.rename(columns={'title': 'occurrences'}, inplace=True)

        # Count number of classes and number of data points in the dataset
        self.train_len, self.train_classes_len = self.count_datapoints_classes(self.train_split,
                                                                               log=True,
                                                                               mode="train")
        self.val_len, self.val_classes_len = self.count_datapoints_classes(self.validation_split,
                                                                           log=True,
                                                                           mode="validation")

        self.test_len = self.count_test_datapoints(test,
                                                   log=True,
                                                   mode="test")

        # Get the top 5 classes with most and top 5 classes with least datapoints
        self.get_top_classes(self.train_counts, log=True, mode="train")
        self.get_top_classes(self.val_counts, log=True, mode="validation")

        # Find the new labels for this run
        self.find_different_labels(self.train_counts)
        # Visualize a treemap from training data
        self.labels_treemap(self.train_counts)

    def read_data(self, filename):
        data = pd.read_csv(filename, sep='\t')
        data['tags'] = data['tags'].apply(literal_eval)
        return data

    def count_test_datapoints(self, data, log=False, mode="test"):
        length = len(data.index)
        if log and mode and self.run:
            self.run["dataset/"+mode+" datapoints count"].log(length)
        return length

    def count_datapoints_classes(self, data, log=False, mode=None):
        length = len(data.index)
        classes = len(data.groupby(["tags"]).count().index)
        if log and mode and self.run:
            self.run["dataset/"+mode+" datapoints count"].log(length)
            self.run["dataset/"+mode+" classes count"].log(classes)
        return length, classes

    def get_top_classes(self, data, log=False, mode=None, num=5):
        data_sort = data.sort_values("occurrences")
        highs = data_sort[-num:].sort_values("occurrences", ascending=False)
        lows = data_sort[:num]
        if log and mode and self.run:
            highs_f = "tmp_highs.csv"
            lows_f = "tmp_lows.csv"
            highs.to_csv(highs_f, index=False)
            lows.to_csv(lows_f, index=False)
            self.run["dataset/" + mode + " top classes"].upload(highs_f)
            self.run["dataset/" + mode + " low classes"].upload(lows_f)

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
        # not used for now
        fig = plt.figure(figsize=(7,4), dpi=100)
        sns.histplot(counts["occurrences"].tolist())
        plt.grid()

        if self.run:
            self.run["dataset/histogram"] = File.as_html(fig)

    def labels_treemap(self, counts):
        fig = plt.figure(figsize=(13,8), dpi=100)
        squarify.plot(sizes=counts['occurrences'], label=counts['tags'], alpha=.8, figure=fig)
        plt.axis('off')

        if self.run:
            treemap_f = "tmp_treemap.png"
            plt.savefig(treemap_f)
            self.run["dataset/treemap"].upload(treemap_f)

    @ staticmethod
    def compute_confusion_matrix(y_true, y_pred):
        # Not used for now
        y_true_new = np.apply_along_axis(' '.join, 1, np.array(y_true,dtype=str))
        y_pred_new = np.apply_along_axis(' '.join, 1, np.array(y_pred,dtype=str))
        cm = confusion_matrix(y_true_new, y_pred_new)

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
        accuracies = mat.diagonal()/mat.sum(axis=1)
        x_labels = mlb.classes

        x = accuracies[1:]
        y = np.sum(y_true, axis=0)
        text = x_labels

        layout = dict(plot_bgcolor='white',
                      margin=dict(t=20, l=20, r=20, b=20),
                      xaxis=dict(
                                 title='Accuracy',
                                 # range=[0.9, 5.5],
                                 linecolor='#d9d9d9',
                                 showgrid=False,
                                 mirror=True),
                      yaxis=dict(title='Number of data points',
                                 # range=[95.5, 99.5],
                                 linecolor='#d9d9d9',
                                 showgrid=False,
                                 mirror=True))

        data = go.Scatter(x=x,
                          y=y,
                          text=text,
                          textposition='top right',
                          textfont=dict(color='#E58606'),
                          mode='markers+text',
                          marker=dict(color='#5D69B1', size=8),
                          #                   line=dict(color='#52BCA3', width=1, dash='dash'),
                          name='tags')

        fig = go.Figure(data=data, layout=layout)

        if self.run:
            self.run["model/accuracies" + mode] = File.as_html(fig)

    def squeeze_labels(self, y_pred, y_true):
        y_pred_ids = np.where(y_pred != 0)
        y_true_ids = np.where(y_true != 0)
        y_pred_squeezed = []
        y_true_squeezed = []
        for i in range(len(y_true_ids[0])):
            y_pred_squeezed.append(list(y_pred_ids[1][y_pred_ids[0]==i]))
            y_true_squeezed.append(list(y_true_ids[1][y_true_ids[0]==i]))
        return y_pred_squeezed, y_true_squeezed

    def find_different_labels(self, counts=None):
        prev_file = self.hist_dir + "x_labels_old.npy"
        if prev_file:
            prev_labels = np.load(prev_file)
        else:
            prev_labels = np.load(self.data_dir + "/x_labels_old.npy")
        curr_labels = counts["tags"].to_list()

        added_labels = list(set(curr_labels) - set(prev_labels))

        # Save the current labels as old ones
        np.save(prev_file, curr_labels)

        # Count the number of occurrences for each new tag
        new_tags = counts[counts["tags"].isin(added_labels)]

        if self.run:
            new_tags_f = self.hist_dir + "/tmp_new_tags.csv"
            new_tags.to_csv(new_tags_f, index=False)
            self.run["dataset/new_tags"].upload(new_tags_f)

        # Save the updated tags for the next run
        np.save(self.data_dir + "/x_labels_old.npy",curr_labels)

    def cleanup(self, extension="tmp_"):
        # Clean up all files starting with "tmp_"
        cwd = os.getcwd()
        files = os.listdir(cwd)

        for item in files:
            if item.startswith(extension):
                os.remove(os.path.join(cwd, item))

