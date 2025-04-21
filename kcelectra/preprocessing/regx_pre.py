import re
import emoji
from soynlp.normalizer import repeat_normalize
from konlpy.tag import Mecab ## linux 에서만
from konlpy.tag import Okt
from zipf_heaps import plot_heaps, plot_zipf
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import requests

url = "https://gist.githubusercontent.com/spikeekips/40eea22ef4a89f629abd87eed535ac6a/raw/4f7a635040442a995568270ac8156448f2d1f0cb/stopwords-ko.txt"

response = requests.get(url)
if response.status_code == 200:
    stopwords = set(response.text.splitlines())
else:
    stopwords = set()
    print("불용어 리스트를 가져오지 못했습니다.")


file_path = "merged_comments.xlsx"
df = pd.read_excel(file_path)

abs_path = os.path.abspath(file_path)
print(f"🔍 Trying to load file at: {abs_path}")
print(f"📂 File exists? {os.path.exists(abs_path)}")

## 정규표현식 전처리
pattern = re.compile('[^ ㄱ-ㅣ가-힣]+')
url_pattern = re.compile(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

def clean(x):
    x = pattern.sub(' ', x)
    x = emoji.replace_emoji(x, replace='')
    x = url_pattern.sub('', x)
    x = x.strip()
    x = repeat_normalize(x, num_repeats=2)
    return x

df_cleaned = df["a"].dropna().astype(str).apply(clean)
df_cleaned = df_cleaned[df_cleaned.str.strip() != ''].to_frame(name='a').reset_index(drop=True)

mecab = Mecab()

## 의미있는 단어 추출
## NN, VV
def extract_meaningful_words(text):
    pos_tags = mecab.pos(text)
    meaningful = [
        word for word, tag in pos_tags
        if tag.startswith(('NNG', 'NNP', 'VV', 'VA')) and word not in stopwords
    ]
    return meaningful

## Bigram
def generate_ngrams(words, n=2):
    return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

## 의미 있는 단어 추출 및 Bigram 생성
comments = df_cleaned['a'].tolist()
tokenized_comments = [extract_meaningful_words(comment) for comment in comments]

all_words = [word for tokens in tokenized_comments for word in tokens]
bigrams = [generate_ngrams(tokens, n=2) for tokens in tokenized_comments]
flat_bigrams = [item for sublist in bigrams for item in sublist]

## 빈도 계산
word_counts = Counter(all_words).most_common(50)
bigram_counts = Counter(flat_bigrams).most_common(50)

words, counts = zip(*word_counts)
plt.figure(figsize=(16, 6))
sns.barplot(x=list(words), y=list(counts), palette='coolwarm')
plt.xticks(rotation=45, ha='right')
plt.title('의미 있는 단어 출현 빈도 Top 50', fontsize = 18)
plt.xlabel('단어', fontsize = 14)
plt.ylabel('빈도수', fontsize = 14)
plt.tight_layout()
plt.show()

bigrams, bigram_freqs = zip(*bigram_counts)
plt.figure(figsize=(16, 6))
sns.barplot(x=list(bigrams), y=list(bigram_freqs), palette='crest')
plt.xticks(rotation=45, ha='right')
plt.title('의미 있는 Bigram 출현 빈도 Top 50',fontsize = 18)
plt.xlabel('Bigram',fontsize = 14)
plt.ylabel('빈도수',fontsize = 14)
plt.tight_layout()
plt.show()

plot_zipf(all_words)
plot_heaps(tokenized_comments)
