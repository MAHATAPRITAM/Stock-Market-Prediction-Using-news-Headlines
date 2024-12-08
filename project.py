"""
Title: Stock market prediction using news headlines
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from textblob import TextBlob
from xgboost import XGBClassifier

def analize_sentiment(tweet):
    
    analysis = TextBlob((str(tweet)))     #defining the function which will find the plority of a sentence
    return analysis.polarity 

news= pd.read_csv('Combined_News_DJIA.csv')

train_news = news[news['Date'] < '2014-07-15']   # SPLITTING THE DATASET INTO TRAINING AND TESTING
test_news = news[news['Date'] > '2014-07-14']

train_news_list = []
for row in range(0,len(train_news.index)): # CONVERT THE TRAINNG DATASET OF 27 COLUMNS INTO ONE ELEMENT IN THE LIST FOR EACH DAY
    train_news_list.append(' '.join(str(k) for k in train_news.iloc[row,2:27]))
    
vectorize= CountVectorizer(min_df=0.01, max_df=0.8) # DEFINING THE VECTOR FUNCTION, SPECIFYING THR MIN AND MAX WORD FREQUENCY FILTER
news_vector = vectorize.fit_transform(train_news_list) # TRANSFORMING THE TRAINING DATASET INTO WORD FREQUENCY TRANFORMATION
print( "THE TABLE OF FREQUENCY WORD DISTRIBUTION" , news_vector.shape)

lr=LogisticRegression()
model = lr.fit(news_vector, train_news["Label"])

test_news_list = []
for row in range(0,len(test_news.index)):
    test_news_list.append(' '.join(str(x) for x in test_news.iloc[row,2:27]))# CONVERT THE TESTING DATASET OF 27 COLUMNS INTO ONE ELEMENT IN THE LIST FOR EACH DAY

test_vector = vectorize.transform(test_news_list) # TRANSFORMING THE TESTING DATASET INTO WORD FREQUENCY TRANFORMATION

predictions = model.predict(test_vector)

pd.crosstab(test_news["Label"], predictions, rownames=["Actual"], colnames=["Predicted"])

accuracy1=accuracy_score(test_news['Label'], predictions)
print("the baseline model accuracy", accuracy1)

# words = vectorize.get_feature_names_out()
# coefficients = model.coef_.tolist()[0]
# coeffdf = pd.DataFrame({'Word' : words,'Coefficient' : coefficients})  # WORD DISTRIBUTION OF THE MODEL

# coeffdf = coeffdf.sort_values(['Coefficient', 'Word'], ascending=[0, 1])
# print("Top ten words according to the baseline model",coeffdf.head(10))
# print("Last ten words according to the baseline model",coeffdf.tail(10))

# words = vectorize.get_feature_names_out()  # Corrected method name
# coefficients = model.coef_.tolist()[0]
# coeffdf = pd.DataFrame({'Word' : words,'Coefficient' : coefficients})  # WORD DISTRIBUTION OF THE MODEL

# coeffdf = coeffdf.sort_values(['Coefficient', 'Word'], ascending=[0, 1])
# print("Top ten words according to the baseline model",coeffdf.head(10))
# print("Last ten words according to the baseline model",coeffdf.tail(10))

 # Define nmodel for logistic regression with bigram and TF-IDF

nvectorize = TfidfVectorizer(min_df=0.05, max_df=0.85, ngram_range=(2,2))  # Define nvectorize for bigram
news_nvector = nvectorize.fit_transform(train_news_list)

nmodel = lr.fit(news_nvector, train_news["Label"])

nwords = nvectorize.get_feature_names_out()  # Corrected method name
ncoefficients = nmodel.coef_.tolist()[0]
ncoeffdf = pd.DataFrame({'Word' : nwords, 
                        'Coefficient' : ncoefficients})
ncoeffdf = ncoeffdf.sort_values(['Coefficient', 'Word'], ascending=[0, 1])
ncoeffdf.head(10)
ncoeffdf.tail(10)


#bi-gram 

nvectorize = TfidfVectorizer(min_df=0.05, max_df=0.85,ngram_range=(2,2)) # DEFINING THE TFID TRANSFORMATION FUNCTION
news_nvector = nvectorize.fit_transform(train_news_list)

print(" TFID TRANSFOMATION DATAFRAME SHAPE",news_nvector.shape)



test_news_list = []
for row in range(0,len(test_news.index)):
    test_news_list.append(' '.join(str(x) for x in test_news.iloc[row,2:27])) # CONVERT THE TESTING DATASET OF 27 COLUMNS INTO ONE ELEMENT IN THE LIST FOR EACH DAY
ntest_vector = nvectorize.transform(test_news_list)
npredictions = nmodel.predict(ntest_vector)

pd.crosstab(test_news["Label"], npredictions, rownames=["Actual"], colnames=["Predicted"])

accuracy2=accuracy_score(test_news['Label'], npredictions)
print(" Logistics Regression with Bigram and TFID",accuracy2)

nwords = nvectorize.get_feature_names_out()
ncoefficients = nmodel.coef_.tolist()[0]
ncoeffdf = pd.DataFrame({'Word' : nwords, 
                        'Coefficient' : ncoefficients})
ncoeffdf = ncoeffdf.sort_values(['Coefficient', 'Word'], ascending=[0, 1])
ncoeffdf.head(10)
ncoeffdf.tail(10)

#random forest - bigram

nvectorize = TfidfVectorizer(min_df=0.01, max_df=0.95,ngram_range=(2,2))
news_nvector = nvectorize.fit_transform(train_news_list)

rfmodel = RandomForestClassifier(random_state=55)  #DEFINNG THE RANDOM FOREST MODEL
rfmodel = rfmodel.fit(news_nvector, train_news["Label"])
test_news_list = []
for row in range(0,len(test_news.index)):
    test_news_list.append(' '.join(str(x) for x in test_news.iloc[row,2:27]))
ntest_vector = nvectorize.transform(test_news_list)

rfpredictions = rfmodel.predict(ntest_vector)
accuracyrf = accuracy_score(test_news['Label'], rfpredictions)
print("Random forest with tfid and bigram", accuracyrf)

#Naive Bayes

nvectorize = TfidfVectorizer(min_df=0.05, max_df=0.8,ngram_range=(2,2)) #DEFINING THE NAIVE BAYS MODEL
news_nvector = nvectorize.fit_transform(train_news_list)

nbmodel = MultinomialNB(alpha=0.5)
nbmodel = nbmodel.fit(news_nvector, train_news["Label"])

test_news_list = []
for row in range(0,len(test_news.index)):
    test_news_list.append(' '.join(str(x) for x in test_news.iloc[row,2:27])) # CONVERT THE TESTING DATASET OF 27 COLUMNS INTO ONE ELEMENT IN THE LIST FOR EACH DAY
ntest_vector = nvectorize.transform(test_news_list)

nbpredictions = nbmodel.predict(ntest_vector)
nbaccuracy=accuracy_score(test_news['Label'], nbpredictions)
print("Naive Bayes accuracy: ",nbaccuracy)

#Gradient Boosting Classifier

gbmodel = GradientBoostingClassifier(random_state=52)  # DEFINING THE GARDIANT BOOSTING MODEL
gbmodel = gbmodel.fit(news_nvector, train_news["Label"])
test_news_list = []
for row in range(0,len(test_news.index)):
    test_news_list.append(' '.join(str(x) for x in test_news.iloc[row,2:27]))
ntest_vector = nvectorize.transform(test_news_list)

gbpredictions = gbmodel.predict(ntest_vector.toarray())
gbaccuracy = accuracy_score(test_news['Label'], gbpredictions)

from sklearn.metrics import confusion_matrix
print(" CONFUSION MATRIX OF THE GRADIANT BOOSTING ", confusion_matrix(test_news['Label'], gbpredictions))


print("Gradient Boosting accuracy: ",gbaccuracy)

#trigram

n3vectorize = TfidfVectorizer(min_df=0.0004, max_df=0.115,ngram_range=(3,3)) # DEFINING THE TFID , TRIGRAM MODEL
news_n3vector = n3vectorize.fit_transform(train_news_list)

print(news_n3vector.shape)

n3model = lr.fit(news_n3vector, train_news["Label"])

test_news_list = []
for row in range(0,len(test_news.index)):
    test_news_list.append(' '.join(str(x) for x in test_news.iloc[row,2:27])) # CONVERT THE TESTING DATASET OF 27 COLUMNS INTO ONE ELEMENT IN THE LIST FOR EACH DAY
n3test_vector = n3vectorize.transform(test_news_list)
n3predictions = n3model.predict(n3test_vector)

pd.crosstab(test_news["Label"], n3predictions, rownames=["Actual"], colnames=["Predicted"])

accuracy3=accuracy_score(test_news['Label'], n3predictions)
print("TRIGARAM ACCURACY", accuracy3)

n3words = n3vectorize.get_feature_names_out()
n3coefficients = n3model.coef_.tolist()[0]
n3coeffdf = pd.DataFrame({'Word' : n3words, 
                        'Coefficient' : n3coefficients})
n3coeffdf = n3coeffdf.sort_values(['Coefficient', 'Word'], ascending=[0, 1])
print("trigram top ten word distibution", n3coeffdf.head(10))
print("trigram last ten word distibution", n3coeffdf.tail(10))    # trigram model word distribution 

### sentiment analysis

train_sentiment=train_news
test_sentiment = test_news
train_sentiment =train_sentiment.drop(['Date', 'Label'], axis=1)
for column in train_sentiment:
    train_sentiment[column]=train_sentiment[column].apply(analize_sentiment)  #converting the train headlines into polarity scores
train_sentiment = train_sentiment+10  # removing negative co:efficient from the datset for better performance

test_sentiment =test_sentiment.drop(['Date', 'Label'], axis=1)
for column in test_sentiment:
    test_sentiment[column]=test_sentiment[column].apply(analize_sentiment) # converting the test headlines into ploarity 
test_sentiment=test_sentiment+10 # removing negative co:efficient from the datset for better performance

XGB_model= XGBClassifier()  # training the polarity score datset with DIJA 
gradiant=XGB_model.fit(train_sentiment, train_news['Label'])
y_pred= gradiant.predict(test_sentiment)


from sklearn.metrics import confusion_matrix
print(confusion_matrix(test_news['Label'], y_pred))
from sklearn.metrics import accuracy_score
print("Sentiment Accuracy",accuracy_score(test_news['Label'], y_pred))
from sklearn.metrics import f1_score
print("f1_score__",f1_score(test_news['Label'], y_pred, average='weighted'))

######################END####################

