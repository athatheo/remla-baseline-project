# import enchant
import pandas as pd
from cerberus import Validator

df = pd.read_csv('../data/validation.tsv', sep='\t')
print(df.head())

# d = enchant.Dict("en_US")
# d.check(df[1][0])
types1 = [type(k) for k in df.to_dict().keys()]
print(types1)


schema = {
    'title': {'type': 'list', 'schema': {'type': ['string']}},
    'tags': {'required': False, 'type': 'list', 'schema': {'type': ['string'],
                                                           'regex': '(\\[)(\'.+\'\,\s)*(\'.+\')(\\])'}}}
v = Validator(schema)
t = df.to_dict('list')

print(v.validate(t))

'''
for t in df.to_dict()['title']:
        if isinstance(t, str):
                continue
        else:
                print(t)
'''
