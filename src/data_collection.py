from text_preprocessing import read_data

data = read_data("../data/raw/train/train.tsv")

from stackapi import StackAPI

SITE = StackAPI('stackoverflow')
questions = SITE.fetch('questions')

data = read_data("../data/raw/train/train.tsv")
tag_counts = {}
for tags in data['tags']:
    for tag in tags:
        if tag in tag_counts:
            tag_counts[tag] += 1
        else:
            tag_counts[tag] = 1
max_keyword = max(tag_counts, key=tag_counts.get)
file_object = open('../data/raw/train/train.tsv', 'a')
for item in questions['items']:
    if max_keyword not in item['tags']:
        file_object.write(item['title'] + '\t' + str(item['tags']) + '\n')
file_object.close()


