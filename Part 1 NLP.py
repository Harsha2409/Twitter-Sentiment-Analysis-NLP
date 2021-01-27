# -*- coding: utf-8 -*-
"""
Created on Mon May 08 16:59:52 2020

Team: Harsha Mangnani and Anthony Bedi

@author: hkmangnani@gmail.com
"""

################## IMPORT PACKAGES ###########################

#import required packages

#!pip install gensim


#import required packages

#!pip install gensim

#!pip install plotly

#!pip install GetOldTweets3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import pyLDAvis
import pyLDAvis.gensim
import gc
import time
import datetime
import warnings
import re
import plotly.express as px
import plotly.graph_objects as go

import GetOldTweets3 as got
from datetime import datetime, timedelta

from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.corpus import stopwords
from nltk.corpus import wordnet

import gensim
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from scipy import sparse

#settings
warnings.filterwarnings("ignore")
lem = WordNetLemmatizer()
tokenizer=ToktokTokenizer()

from textblob import TextBlob
#settings
warnings.filterwarnings("ignore")
lem = WordNetLemmatizer()
tokenizer=ToktokTokenizer()

from textblob import TextBlob


#############################################################################
###################### DATA COLLECTION ####################################
############################################################################

import os
os.system("python ./twitter_search2.py covid-19 -c 5000")
os.system("python ./twitter_search2.py coronavirus -c 5000")
os.system("python ./twitter_search2.py coronavirus -c 5000")

df1 = pd.read_csv('result_covid-19.csv', index_col=0)
df2 = pd.read_csv('result_pandemic.csv', index_col=0)
df3 = pd.read_csv('result_coronavirus.csv', index_col=0)

df = [df1, df2, df3]
df = pd.concat(df)

df.to_csv("tweets1.csv")

df = pd.read_csv('tweets1.csv', index_col=0)

df.reset_index(inplace=True)  #created date will now be a column, not index


##############################################################################
################### DATA CLEANING ##########################################
##############################################################################

####################################### Date and Time Segregation 

# devide timestamp into date and time

df['Date'] = pd.to_datetime(df['usersince'])
df['usersince_date'] = df['Date'].apply( lambda x: x.strftime("%Y-%m-%d"))
df['usersince_time'] = df['Date'].apply( lambda x: x.strftime("%H-%M-%S"))
df.drop(['Date'],axis = 1, inplace =True)

df['Date2'] = pd.to_datetime(df['created'])
df['created_date'] = df['Date2'].apply( lambda x: x.strftime("%Y-%m-%d"))
df['created_time'] = df['Date2'].apply( lambda x: x.strftime("%H-%M-%S"))
df.drop(['Date2'],axis = 1, inplace =True)


###################################### Remove unwanted patterns from tweets:

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt

######################################  Removing Twitter Handles (@user)

df['tidy_tweet'] = np.vectorize(remove_pattern)(df['text'], "@[\w]*")



############################### Removing Punctuations, Numbers and Special Characters

#Here we will replace everything except characters and hashtags with spaces.
#The regular expression “[^a-zA-Z#1-9]” means anything except alphabets and ‘#'.'

df['tidy_tweet'] = df['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")
  
  
################################### Removing Short Words: less than 3 alphabets. Eg: hi, oh, hmm 

df['tidy_tweet'] = df['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))



####################################### Lower Casing 

df['tidy_tweet']  = df['tidy_tweet'].str.lower()


############################################# Removing STOPWORDS

#Importing stopwords from nltk library

from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))


##Function to remove the stopwords
def stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

# Applying the stopwords to 'text_punct' and store into 'text_stop'
df["tidy_tweet"] = df["tidy_tweet"].apply(stopwords)

df["tidy_tweet"].head()


################################## Removing Most Common Words (optional) (outlier)

#We can also remove commonly occurring words from our text data First
#let’s check the 15 most frequently occurring words in our text data.

# Checking the first 10 most frequent words
from collections import Counter
cnt = Counter()
for text in df["tidy_tweet"].values:
    for word in text.split():
        cnt[word] += 1
        
freq_words = cnt.most_common(15)
freq_words

#did not do the removal
'''
# Removing the frequent words
freq = set([w for (w, wc) in freq_words])

# function to remove the frequent words

def freqwords(text):
    return " ".join([word for word in str(text).split() if word not 
in freq])

# Passing the function freqwords
df["tidy_tweet"] = df["tidy_tweet"].apply(freqwords)
'''


############################################ Removing rare words

# Removal of 15 rare words

freq_less = pd.Series(' '.join(df['tidy_tweet']).split()).value_counts()[-15:] # 15 rare words
freq_less

#removed rare words
df['tidy_tweet'] = df['tidy_tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in freq_less))


######################################### Removing emoticons

#did not do this

# Function to remove emoticons

'''
#!pip install emot

import emot
from emot.emo_unicode import UNICODE_EMO, EMOTICONS

# Function for removing emoticons

def remove_emoticons(text):
    emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in EMOTICONS) + u')')
    return emoticon_pattern.sub(r'', text)

remove_emoticons("Hello :-)")

    # applying remove_emoticons to 'tidy_text'
df['tidy_tweet'] = df['tidy_tweet'].apply(remove_emoticons)
'''
############################################# Converting Emojis to Texts

from emot.emo_unicode import UNICODE_EMO, EMOTICONS
#!pip install emoji
import emoji

# Converting emojis to words
def convert_emojis(text):
    for emot in UNICODE_EMO:
        text = emoji.demojize(text)
        #text = text.replace(emot, "_".join(UNICODE_EMO[emot].replace(",","").replace(":","").split()))
        return text
    
#example
text1 = "Hilarious 😂"
print(convert_emojis(text1))

#Applying the remove emoji function
df['tidy_tweet'] = df['tidy_tweet'].apply(convert_emojis)

#################################################### Removing URLs

# Function for url's

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

# Examples
text = "This is my website, https://www.abc.com"
remove_urls(text)

#Passing the function
df['tidy_tweet'] = df['tidy_tweet'].apply(remove_urls)


################################ BEFORE AND AFTER CLEANING ###############
#Before and After Preprocessing
df[['text', 'tidy_tweet']].head()



################################################ Tokenization

#Creating function for tokenization
def tokenization(text):
    text = re.split('\W+', text)
    return text


# Passing the function to 'tidy_tweet' and store into'token_tweet'
df['token_tweet'] = df['tidy_tweet'].apply(lambda x: tokenization(x)


#Before and after Tokeinzation
df[['tidy_tweet','token_tweet']].head()

#################################################### Stemming

#Stemming

from nltk.stem.porter import * 
stemmer = PorterStemmer() 

stem_tweet = df['token_tweet'].apply(lambda x: [stemmer.stem(i) for i in x]) 
                
#Now let’s stitch these tokens back together. It can easily be done using nltk’s MosesDetokenizer function.

for i in range(len(stem_tweet)):
    stem_tweet[i] = ' '.join(stem_tweet[i])    
    
df['stem_tweet'] = stem_tweet

#Before and after Stemming
df[['token_tweet', 'stem_tweet']].sample(5)



####################################################### Lemmatization

#Lemmatizing

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Pos tag, used Noun, Verb, Adjective and Adverb
wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV} 

# Function for lemmatization using POS tag
def lemmatize_words(text):
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])

# Passing the function to 'text_rare' and store in 'text_lemma'
df["lemm_tweet"] = df["tidy_tweet"].apply(lemmatize_words)

#Before and After Lemmatization
df[['tidy_tweet', 'lemm_tweet']].sample(5)


############################# Adding word count for Anthony's Hypotheis ##########

import re
re.compile('<title>(.*)</title>')


#Sentense count in each comment:

    #  '\n' can be used to count the number of sentences in each comment
    
df['count_sent']=df["text"].apply(lambda x: len(re.findall("\n",str(x)))+1)

#Word count in each comment:
df['count_word']=df["text"].apply(lambda x: len(str(x).split()))

#Unique word count
df['count_unique_word']=df["text"].apply(lambda x: len(set(str(x).split())))


df.shape

##############################################################################
############################ SENTIMENT ANALYSIS USING TEXTBLOB ################
###############################################################################

#Collecting 100 tweets for each day from Feb 11, 2020 to May 08, 2020 (excluded)

posPercent = []
negPercent = []
numTweets = []
dates = []

#set start date, strip time because the 'created_at' is datetime object. We only need date.


start_date = datetime.strptime('2020-02-11', '%Y-%m-%d')  

while start_date != datetime.strptime('2020-05-08', '%Y-%m-%d'): #runs till date May 08, 2020
    
    dates.append(start_date) #append dates (used to plot)
    
    #search for keywords. strip time. Max tweets 100.
    #tried for 150-500 tweets, was taking a lot of time and had to interrupt kernel so stuck with 100.
    #using Kafka mightbe a work around, but I am still learning it. Hence, stuck with 100
    
    tweetCriteria = got.manager.TweetCriteria().setQuerySearch("corona virus pandemic covid").setSince(start_date.strftime('%Y-%m-%d')).setUntil((start_date + timedelta(days=1)).strftime('%Y-%m-%d')).setMaxTweets(100)
    tweets = got.manager.TweetManager.getTweets(tweetCriteria)
    
    #incremented days by 1. SO for sure, we get tweets for each day related to our keywords.
    
    ptweet = 0
    ntweet = 0
    for tweet in tweets:
        
        #remove unnecessary patterns
        
        text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)", " ", tweet.text).split())
        
        #count number of positive and negative tweets.
        
        ptweet += (TextBlob(text).sentiment.polarity > 0.2)
        ntweet += (TextBlob(text).sentiment.polarity < -0.2)
        
    #calculate percent of positive and negative tweets
    
    posPercent.append(100*ptweet/len(tweets))
    negPercent.append(100*ntweet/len(tweets))
    numTweets.append(len(tweets))
    
    start_date += timedelta(days=1)


###################### Plot time series plot of polarity

data = pd.DataFrame(list(zip(dates, numTweets, posPercent, negPercent)), columns=['Date','Num Tweets', '% Pos', '% Neg'])
fig = go.Figure()

fig.add_trace(go.Scatter(x=data.Date, y=data['% Pos'],
                    mode='lines',
                    name='% Pos'))
fig.add_trace(go.Scatter(x=data.Date, y=data['% Neg'],
                    mode='lines',
                    name='% Neg'))
fig.update_layout(title='Twitter Sentiment Analysis Coronavirus Response',
                   xaxis_title='Date',
                   yaxis_title='Percentage')
fig.show()


################################## Sentiment of each tweeet in original data set 

subj_list= []
polarity_list = []

for index, row in df.iterrows():
    tweet = row["tidy_tweet"]
    text = TextBlob(tweet) #convert tweet to TextBlob object
    
    #sentiment analysis
    subjectivity = text.sentiment.subjectivity # 0 to 1
    polarity = text.sentiment.polarity # -1 to 1
    
    #print(tweet, subjectivity, polarity)
    
    subj_list.append(subjectivity)
    polarity_list.append(polarity)
    
df['Subjectivity'] = subj_list
df['Polarity'] = polarity_list


############################################## Word Cloud for all tweets

from wordcloud import WordCloud

all_words = ''.join([text for text in df['tidy_tweet']])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=200).generate(all_words)

plt.figure(figsize=(10, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.title("Frequent Words in Tweets", fontsize=20)
plt.axis('off')
plt.show()

############################################## Word Cloud for Positive tweets

pos_words =' '.join([text for text in df['tidy_tweet'][df['Polarity'] > 0.4]]) 
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=200).generate(pos_words)
plt.figure(figsize=(10, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.title("Frequent Words in Positive Tweets", fontsize=20)
plt.axis('off')
plt.show()


####################################################### Word Cloud for Negative tweets

neg_words =' '.join([text for text in df['tidy_tweet'][df['Polarity'] < -0.4]]) 
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=200).generate(neg_words)
plt.figure(figsize=(10, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.title("Frequent Words in Negative Tweets", fontsize=20)
plt.axis('off')
plt.show()



####################################### 'Trends' --> Impact of Hashtag

# function to collect hashtags

def hashtag_extract(x):
    hashtags = []
    
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)
    return hashtags

# extracting hashtags from positive tweets 
tw_pos = hashtag_extract(df['tidy_tweet'][df['Polarity'] > 0.4])
ht_pos = hashtag_extract(df['hashtag'][df['Polarity'] > 0.4])

# extracting hashtags from negative tweets
tw_neg = hashtag_extract(df['tidy_tweet'][df['Polarity'] < -0.4]) 
ht_neg = hashtag_extract(df['hashtag'][df['Polarity'] < -0.4])

# combining trends list
pos = sum(ht_pos,tw_pos)
neg = sum(ht_neg,tw_neg)

#unnesting list
pos = sum(pos, [])
neg = sum(neg, [])

#Example
print(pos[0:5])
print(neg[0:5])

####################################### Plot top 'n' positive hashtags/trends

import seaborn as sns

# plot positive hashtags

a = nltk.FreqDist(pos)
d = pd.DataFrame({'Hashtag': list(a.keys()),'Count': list(a.values())}) 

# selecting top 10 most frequent hashtags

d = d.nlargest(columns="Count", n = 10)
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.title("Top 10 hashtags/trends in Positive Tweets", fontsize=20)
plt.xticks(rotation=30)
plt.show()
plt.tight_layout()


######################################### Plot top 'n' negative hashtag/tweets

#plot negative hashtags

a = nltk.FreqDist(neg)
d = pd.DataFrame({'Hashtag': list(a.keys()),'Count': list(a.values())}) 

# selecting top 10 most frequent hashtags

d = d.nlargest(columns="Count", n = 10)
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.title("Top 10 hashtags/trends in Negative Tweets", fontsize=20)
plt.xticks(rotation=30)
plt.show()
plt.tight_layout()



#############################################################################
##################### SENTIMENT ANALYSIS USING VADER ######################
############################################################################

#!pip install vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sent_analyser = SentimentIntensityAnalyzer()

#function for calculating sentiment
def calculate_sentiment_analyser(text):    
    return sent_analyser.polarity_scores(text)

#Example1
c = (" I love Mango so much that I am starting to hate it")
calculate_sentiment_analyser(c)

#Example 2
c = (" I love Mango so much that I am starting to hate it!!!")
calculate_sentiment_analyser(c)

#Example 3
c = (" I love Mango so much that I am starting to hate it!! <3")
calculate_sentiment_analyser(c)

#################Vader considers punctuations and emojis
############# will keep punctuations and emojis intact

#remove user handles
df['vader_tweet'] = np.vectorize(remove_pattern)(df['text'], "@[\w]*")
df.head(2)

# remove special characters, kept punctuation like , and ! intact
df['vader_tweet'] = df['vader_tweet'].str.replace("[^a-zA-Z#!,$]", " ")
df.head(2)

#remove short words
df['vader_tweet'] = df['vader_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
df.head(2)

#lower case
df['vader_tweet'] = df['vader_tweet'].str.lower()
df.head(2)

#remove stopwords
df["vader_tweet"] = df["vader_tweet"].apply(stopwords)
df.head(2)

#remove rare words
df['vader_tweet'] = df['vader_tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in freq_less))
df.head(2)

#remove URls
df['vader_tweet'] = df['vader_tweet'].apply(remove_urls)
df.head(2)

##################### apply sentiment analyser function on cleaned tweets

df['sentiment_analyser']=df['vader_tweet'].apply(calculate_sentiment_analyser)

s = pd.DataFrame(index = range(0,len(df)),columns= ['compound_score','compound_score_sentiment'])

for i in range(0,len(df)):
    
  s['compound_score'][i] = df['sentiment_analyser'][i]['compound']
  
  if (df['sentiment_analyser'][i]['compound'] <= -0.05):
    s['compound_score_sentiment'][i] = 'Negative'   
    
  if (df['sentiment_analyser'][i]['compound'] >= 0.05):
    s['compound_score_sentiment'][i] = 'Positive'
    
  if ((df['sentiment_analyser'][i]['compound'] >= -0.05) & (df['sentiment_analyser'][i]['compound'] <= 0.05)):
    s['compound_score_sentiment'][i] = 'Neutral'
    
df['compound_score'] = s['compound_score']
df['compound_score_sentiment'] = s['compound_score_sentiment']

##########################comparison between textblob and vader
df[['vader_tweet','Polarity', 'compound_score', 'compound_score_sentiment']].sample(10)


################################## Plot number of tweets in each class
df['compound_score_sentiment'].value_counts().plot(kind='bar', title='Number of Tweets')

df.compound_score_sentiment.value_counts()


##############################################################################
################################ Feature Extraction ########################
#############################################################################


########################### Bag of Words

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import gensim


bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
bow = bow_vectorizer.fit_transform(df['tidy_tweet'])
bow.shape


############################ TF-IDF
#TF-IDF vectorizer

tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(df['tidy_tweet'])
tfidf.shape


##################################### Word Embeddings

#Word2Vec features

tokenized_tweet = df['token_tweet'] 


model_w2v = gensim.models.Word2Vec(
            tokenized_tweet,
            size=200, # desired no. of features/independent variables
            window=5, # context window size
            min_count=2,
            sg = 1, # 1 for skip-gram model
            hs = 0,
            negative = 10, # for negative sampling
            workers= 2, # no.of cores
            seed = 34) 

model_w2v.train(tokenized_tweet, total_examples= len(df['token_tweet']), epochs=20)

#Example:

#We will specify a word and the model will pull out the most similar words from the corpus.

model_w2v.wv.most_similar(positive="trump")

model_w2v.wv.most_similar(positive="pandemic")


####### Preparing vectors of tweets

#function to create vectors
def word_vector(tokens, size):
    '''
    This function will create a vector for each tweet
    by taking the average of the vectors of the words present in the tweet.
    '''
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += model_w2v[word].reshape((1, size))
            count += 1.
        except KeyError: # handling the case where the token is not in vocabulary
            continue

    if count != 0:
        vec /= count
    return vec

#applying the function
    #Preparing word2vec feature set…

wordvec_arrays = np.zeros((len(tokenized_tweet), 200)) 

for i in range(len(tokenized_tweet)):
    wordvec_arrays[i,:] = word_vector(tokenized_tweet[i], 200)
    wordvec_df = pd.DataFrame(wordvec_arrays)

wordvec_df.shape    

#Now we have 200 new features, whereas in Bag of Words and TF-IDF we had 1000 features.


##########################################
# Create a copy of the DataFrame to work 
df.to_csv('tweets2.csv')

##############################################################################
############################## BUILDING CLASSIFICATION MODELS ############

############################ Train Test Split ##############################

# below line causes shuffling of indices, to avoid using train_test_split later
df = df.reindex(np.random.permutation(df.index))

df_new = df[['tidy_tweet', 'compound_score_sentiment']]

####################################### 2:1 for Train and Test

def shuffle(df, test_proportion):
    ratio = int(len(df)/test_proportion)
    train = df[ratio:][:]
    test =  df[:ratio][:]
    
    return train,test


train,test= shuffle(df_new,3)


###################################### length of tweets

#Distribution of tweets

length_train = train['tidy_tweet'].str.len()
length_test = test['tidy_tweet'].str.len()
plt.hist(length_train, bins=20, label="train_tweets") 
plt.hist(length_test, bins=20, label="test_tweets") 
plt.legend() 
plt.title("Distribution of lebgth of tweets in train and test set ")
plt.show()


############################ Modelling on Bag of Words ###############

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import gensim
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features = 1000, stop_words='english')

bow_features = bow_vectorizer.fit_transform(df['tidy_tweet']).toarray()
labels = df_new.compound_score_sentiment
bow_features.shape

########### modelling

models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0)
]

CV = 5

cv_bow = pd.DataFrame(index=range(CV * len(models)))
entries = []

for model in models:
    
    model_name = model.__class__.__name__
    f1 = cross_val_score(model, bow_features, labels, scoring='f1_weighted', cv=CV)
    
    for fold_idx, f1 in enumerate(f1):
        entries.append((model_name, fold_idx, f1))
    
cv_bow = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'f1'])


cv_bow.sample(5)

######## Plot
plt.figure(figsize=(7,5))
sns.boxplot(x='model_name', y='f1', data=cv_bow)
sns.stripplot(x='model_name', y='f1', data=cv_bow, size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.xticks(rotation = 20)
plt.ylabel("F1 scores")
plt.show()

print("F1 scores of different models on Bag of Words (BoW) Features")
bow_df = cv_bow.groupby('model_name').f1.median()
bow_df


######################### Modelling on TF-IDF Features ##################

#for modelling on TF-IDF

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2',
                        encoding='latin-1', ngram_range=(1, 2), stop_words='english')

features = tfidf.fit_transform(df_new.tidy_tweet).toarray()
labels = df_new.compound_score_sentiment
features.shape

from sklearn.model_selection import cross_val_score

################## modelling

models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0)
]

CV = 5

cv_tfidf = pd.DataFrame(index=range(CV * len(models)))
entries = []

for model in models:
    
    model_name = model.__class__.__name__
    f1 = cross_val_score(model, features, labels, scoring='f1_weighted', cv=CV)
    #accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
    
    for fold_idx, f1 in enumerate(f1):
        entries.append((model_name, fold_idx, f1))
    
cv_tfidf = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'f1'])

cv_tfidf.sample(5)

#Plot

plt.figure(figsize=(7,5))
sns.boxplot(x='model_name', y='f1', data=cv_tfidf)
sns.stripplot(x='model_name', y='f1', data=cv_tfidf, size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.xticks(rotation = 30)
plt.ylabel("F1 scores")
plt.show()

print("F1 scores of different models on TF-1DF Features")
tfidf_df = cv_tfidf.groupby('model_name').f1.median()
tfidf_df


############################# Modelling on Word2Vec Features ##################

#Preparing word2vec feature set for modelling

wordvec_arrays = np.zeros((len(tokenized_tweet), 200)) 

for i in range(len(tokenized_tweet)):
    wordvec_arrays[i,:] = word_vector(tokenized_tweet[i], 200)
    wordvec_features = wordvec_arrays

labels = df_new.compound_score_sentiment
wordvec_features.shape

#on Word2Vec

models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
    
]

CV = 5

cv_w2v = pd.DataFrame(index=range(CV * len(models)))
entries = []

for model in models:
    
    model_name = model.__class__.__name__
    f1 = cross_val_score(model, wordvec_features, labels, scoring='f1_weighted', cv=CV)
    
    for fold_idx, f1 in enumerate(f1):
        entries.append((model_name, fold_idx, f1))
    
cv_w2v = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'f1'])

cv_w2v.sample(3)

cv_w2v.dropna()

#  Plot

plt.figure(figsize=(7,5))
sns.boxplot(x='model_name', y='f1', data=cv_w2v)
sns.stripplot(x='model_name', y='f1', data=cv_w2v, size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.xticks(rotation = 30)
plt.ylabel("F1 scores")
plt.show()

print("F1 scores of different models on Word2Vec Features")
w2v_df = cv_w2v.groupby('model_name').f1.median()
w2v_df

############# merging

result = pd.merge(bow_df, tfidf_df, on='model_name')
result = pd.merge(result, w2v_df, on = 'model_name')
result.columns = ['bow', 'tfidf', 'w2v']
result


####################### CLASSIFICATION LINEAR SVC #################

#Model SVC on TF-IDF prediction

model = LinearSVC()

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features,
                                                                                 labels,
                                                                                 df.index,
                                                                                 test_size=0.33,
                                                                                 random_state=0)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

########### EVALUATION


from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average = 'weighted')
recall = recall_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average = 'weighted')

cfr = classification_report(y_test, y_pred)

print("Accuracy : ", accuracy)
print("F1 : ", f1)
print("Recall : ", recall)
print("Precision : ", precision)
print("Classification Report : ")
print(cfr)

#################### CONFUSION MATRIX

#confusion matrix

cm = metrics.confusion_matrix(y_test, y_pred)

df_cm = pd.DataFrame(cm, columns=np.unique(y_test), index = np.unique(y_test))

df_cm.index.name = 'Actual'

df_cm.columns.name = 'Predicted'

plt.figure(figsize = (10,7))

sns.set(font_scale=1.4)#for label size

sns.heatmap(df_cm, cmap="Blues", fmt = 'd', annot=True,annot_kws={"size": 16})# font size



###############################################################################
##################### TOPIC MODELLING - LDA ####################################
###############################################################################

#LDA Topic Modelling

#BIGRAMS

bigram = gensim.models.Phrases(text)

def clean(word_list):
    """
    Function to clean the pre-processed word lists 
    #ADDED BIGRAMS for LDA
    
    Following transformations will be done
    1) Stop words removal from the nltk stopword list
    2) Bigram collation (Finding common bigrams and grouping them together using gensim.models.phrases)
    3) Lemmatization (Converting word to its root form : babies --> baby ; children --> child)
    """
    #remove stop words
    clean_words = [w for w in word_list if not w in eng_stopwords]
    
    #collect bigrams
    clean_words = bigram[clean_words]
    
    #Lemmatize
    clean_words=[lem.lemmatize(word, "v") for word in clean_words]
    return(clean_words)    


all_text = text.apply(lambda x:clean(x))
 
dictionary = Dictionary(all_text)
print("There are",len(dictionary),"number of words in the final dictionary")


print(dictionary.doc2bow(all_text.iloc[1]))
print("Wordlist from the sentence:",all_text.iloc[1])

#to check
print("Wordlist from the dictionary lookup:", 
      dictionary[21],dictionary[22],dictionary[23],dictionary[24],dictionary[25],dictionary[26],dictionary[27])




# BUILDING CORPUS

corpus = [dictionary.doc2bow(text) for text in all_text]


#Aplying LDA 

ldamodel = LdaModel(corpus=corpus, num_topics=15, id2word=dictionary)

#### visualizing lDA

pyLDAvis.enable_notebook()   #ran in jupyter notebook

pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)





################################ END #########################################

