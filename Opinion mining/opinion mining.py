import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import LancasterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import re
import pickle

df = pd.read_csv('comments.csv',
                 delimiter=',', encoding='ISO-8859-1')
df.columns = ['Sentiment','id','text']
print(df.head())
df = df[['Sentiment','text']]

print(df.Sentiment.value_counts())
df['Sentiment'] = df['Sentiment'].replace({4:1})
print(df.head())

## remove stopwords and punctuation marks
stuff_to_be_removed = list(stopwords.words('english'))+list(punctuation)
stemmer = LancasterStemmer()
corpus = df['text'].tolist()
print(len(corpus))
print(corpus[0])

final_corpus = []
final_corpus_joined = []
for i in df.index:
    text = re.sub('[^a-zA-Z]', ' ', df['text'][i])
    #Convert to lowercase
    text = text.lower()
    #remove tags
    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    ##Convert to list from string
    text = text.split()
    #Lemmatisation
    lem = WordNetLemmatizer()
    text = [lem.lemmatize(word) for word in text 
            if not word in stuff_to_be_removed] 
    text1 = " ".join(text)
    final_corpus.append(text)
    final_corpus_joined.append(text1)

data_cleaned = pd.DataFrame()
data_cleaned["text"] = final_corpus_joined
data_cleaned["Sentiment"] = df["Sentiment"].values
print(data_cleaned.head())

data_eda = pd.DataFrame()
data_eda['text'] = final_corpus
data_eda['Sentiment'] = df["Sentiment"].values
print(data_eda.head())

# Storing positive data seperately
positive = data_eda[data_eda['Sentiment'] == 1]
positive_list = positive['text'].tolist()
# Storing negative data seperately
negative = data_eda[data_eda['Sentiment'] == 0]
negative_list = negative['text'].tolist()

positive_all = " ".join([word for sent in positive_list for word in sent ])
negative_all = " ".join([word for sent in negative_list for word in sent ])

## Word CLoud Positive data
from wordcloud import WordCloud
WordCloud()
wordcloud = WordCloud(width=2000,
                      height=1000,
                      background_color='#F2EDD7FF',
                      max_words = 100).generate(positive_all)

plt.figure(figsize=(20,30))
plt.imshow(wordcloud)
plt.title("Positive")
plt.show()

## Word CLoud Negative data
from wordcloud import WordCloud
WordCloud()
wordcloud = WordCloud(width=2000,
                      height=1000,
                      background_color='#F2EDD7FF',
                      max_words = 100).generate(negative_all)

plt.figure(figsize=(20,30))
plt.imshow(wordcloud)
plt.title("negative")
plt.show()

def get_count(data):
    dic = {}
    for i in data:
        for j in i:
            if j not in dic:
                dic[j]=1
            else:
                dic[j]+=1             
    return(dic)

count_corpus = get_count(positive_list)

count_corpus = pd.DataFrame({"word":count_corpus.keys(),"count":count_corpus.values()})
count_corpus = count_corpus.sort_values(by = "count", ascending = False)

plt.figure(figsize = (15,10))
sns.barplot(x = count_corpus["word"][:20], y = count_corpus["count"][:20])
plt.title('one words in positive data')
plt.show()


count_corpus = get_count(negative_list)

count_corpus = pd.DataFrame({"word":count_corpus.keys(),"count":count_corpus.values()})
count_corpus = count_corpus.sort_values(by = "count", ascending = False)

plt.figure(figsize = (15,10))
sns.barplot(x = count_corpus["word"][:20], y = count_corpus["count"][:20])
plt.title('one words in negative data')
plt.show()

#TFIDF for sentiment analysis + ML Algo

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
tfidf = TfidfVectorizer()
vector = tfidf.fit_transform(data_cleaned['text'])
print(vector)
y = data_cleaned['Sentiment']
X_train, X_test, y_train, y_test = train_test_split(vector, 
                                                    y, 
                                                    test_size=0.33, 
                                                    random_state=42,
                                                    stratify = y)
print(type(X_test))
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score,classification_report
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

print("training accuracy = ",round(accuracy_score(y_train,y_train_pred),2)*100)
ConfusionMatrixDisplay.from_predictions(y_train,y_train_pred,normalize = 'all')
print(classification_report(y_train,y_train_pred))
plt.show()

print("testing accuracy = ",round(accuracy_score(y_test,y_test_pred),2)*100)
ConfusionMatrixDisplay.from_predictions(y_test,y_test_pred,normalize = 'all')
print(classification_report(y_test,y_test_pred))
plt.show()

vec=tfidf.transform([data_cleaned['text'][0]])
print(vec)
print(lr.predict(vec))
####################################################   SAVE MODEL   ###############################################
# filename="opinion mining.sav"
# pickle.dump(lr, open(filename, 'wb'))
# filename="tfidf.sav"
# pickle.dump(tfidf, open(filename, 'wb'))