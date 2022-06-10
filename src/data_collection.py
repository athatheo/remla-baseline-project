from text_preprocessing import read_data
from data_validation import validate
from stackapi import StackAPI
import re

TRAINSET_FILE = 'data/raw/train/train.tsv'

data = read_data(TRAINSET_FILE)

# pull 500 questions
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
file_object = open(TRAINSET_FILE, 'a')

newTagsCount = 0 # helper counter in order to just get the <maxNewTags> new tags
maxNewTags = 2
for item in questions['items']:
    try:
        if (max_keyword not in item['tags']
            and not data['title'].str.contains(item['title']).any()):
            filtered_tags = [tag for tag in item['tags'] if tag in tag_counts]
            if newTagsCount < maxNewTags:
                if filtered_tags != item['tags']:
                    new_tag = list(set(item['tags']).symmetric_difference(set(filtered_tags)))[0]
                    tag_counts[new_tag] = 1
                    newTagsCount += 1
            if filtered_tags:
                question = re.sub(r'\\u\w{4}', '', item['title'])
                file_object.write(question + '\t' + str(filtered_tags) + '\n')
    except:
        continue

file_object.close()

valid = validate(TRAINSET_FILE)
print(f'collected data are {"valid" if valid else "invalid"}')
