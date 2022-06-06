from text_preprocessing import read_data
from data_validation import validate
from stackapi import StackAPI


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

for item in questions['items']:
    try:
        if (max_keyword not in item['tags']
            and not data['title'].str.contains(item['title']).any()):
            filtered_tags = [tag for tag in item['tags'] if tag in tag_counts]
            if filtered_tags:
                file_object.write(item['title'] + '\t' + str(filtered_tags) + '\n')
    except:
        continue

file_object.close()

valid = validate(TRAINSET_FILE)
print(f'collected data are {"valid" if valid else "invalid"}')
