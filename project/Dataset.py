import pandas as pd
from nltk.corpus import stopwords
import re
import langid

train_df = pd.read_csv('F:/Bachelor/project/dataset/train.csv', header=None, names=['label', 'title', 'text'])
test_df = pd.read_csv('F:/Bachelor/project/dataset/test.csv', header=None, names=['label', 'title', 'text'])

train_df['title'].fillna('', inplace=True)
test_df['title'].fillna('', inplace=True)

# train_len = 80000
# test_len = 20000

# augmented dataset
train_len = 160000
test_len = 40000

seed = 2

df = pd.concat([train_df.loc[train_df['label'] == 1].sample(train_len//2, random_state=seed),
                train_df.loc[train_df['label'] == 2].sample(train_len//2, random_state=seed),
                test_df.loc[test_df['label'] == 1].sample(test_len//2, random_state=seed),
                test_df.loc[test_df['label'] == 2].sample(test_len//2, random_state=seed)]).reset_index(drop=True)

df['text'] = df['title'] + '. ' + df['text']
df.drop('title', axis=1, inplace=True)

langs = [langid.classify(s)[0] for s in df['text']]
df = df[[l=='en' for l in langs]]

# delete unicode characters
stopwords = set(stopwords.words('english'))
def text_prepro(text):
  text = text.lower()
  text = ' '.join([word for word in text.split(' ') if word not in stopwords])
  text = text.encode('ascii', 'ignore').decode()
  text = re.sub(r'[\u4e00-\u9fa5]', ' ', text)
  text = re.sub(r'https*\S+', ' ', text)
  text = re.sub(r'@\S+', ' ', text)
  text = re.sub(r'#\S+', ' ', text)
  return text

df['text'] = df.text.apply(text_prepro)

print(df.head())
print(df.tail())

print(f'Label counts - training set:\n{df[:train_len].label.value_counts()}')
print(f'\nLabel counts - test set:\n{df[train_len:].label.value_counts()}')

train_data = df[:train_len]
test_data = df[train_len:]

# train_data.to_csv('F:/Bachelor/project/dataset/processed-train.csv')
# test_data.to_csv('F:/Bachelor/project/dataset/processed-test.csv')

train_data.to_csv('F:/Bachelor/project/dataset/augmented-processed-train.csv')
test_data.to_csv('F:/Bachelor/project/dataset/augmented-processed-test.csv')