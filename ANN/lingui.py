import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# 1. Load the data
df = pd.read_csv("https://github.com/jbesomi/texthero/raw/master/dataset/bbcsport.csv")

# 2. Initialize Vectorizer (ngram_range=(2, 2) means only bigrams)
vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words='english')

# 3. Fit and transform the text data
matrix = vectorizer.fit_transform(df['text'])

# 4. Get the feature names (the n-grams)
# Note: This returns the unique bigrams found in the corpus
ngrams = vectorizer.get_feature_names_out()

print(ngrams[:10])