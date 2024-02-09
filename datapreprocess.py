import nltk
import numpy as np
from nltk.stem.porter import *
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('wordnet')
stemmer = PorterStemmer()
def preprocess(tweets):

    def remove_pattern(input_txt, pattern):
        r = re.findall(pattern, input_txt)
        for i in r:
            input_txt = re.sub(i, '', input_txt)
        return input_txt

    tweets['t1'] = np.vectorize(remove_pattern)(tweets['t1'], "@[\w]*")

    #Removing Punctuations, Numbers, and Special Characters
    tweets['t1'] = tweets['t1'].str.replace("[^a-zA-Z#]", " ")

    #Removing Short Words
    tweets['t1'] = tweets['t1'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

    #Tokenization
    tokenized_tweet = tweets['t1'].apply(lambda x: x.split())

    #stemming
    tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])

    for i in range(len(tokenized_tweet)):
        tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

    tweets['t1'] = tokenized_tweet
    print(tweets.head)

    bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
    bow = bow_vectorizer.fit_transform(tweets['t1'])
    return bow,tweets