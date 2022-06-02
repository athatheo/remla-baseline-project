from text_preprocessing import read_data

data = read_data("data/raw/train/train.tsv")

from stackapi import StackAPI

SITE = StackAPI('stackoverflow')
questions = SITE.fetch('questions')

tag_counts = {}
for tags in data['tags']:
    for tag in tags:
        if tag in tag_counts:
            tag_counts[tag] += 1
        else:
            tag_counts[tag] = 1
max_keyword = max(tag_counts, key=tag_counts.get)
file_object = open('data/raw/train/train.tsv', 'a')
count_1 = 0
count_2 = 0
for item in questions['items']:
    try:
        if max_keyword not in item['tags'] and not data['title'].str.contains(item['title']).any():
            file_object.write(item['title'] + '\t' + str(item['tags']) + '\n')
    except:
        continue

file_object.close()


