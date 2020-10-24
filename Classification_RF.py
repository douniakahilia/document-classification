# In[1]:


#so as to minimize typing strokes.
import nltk
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import sklearn.metrics
from pandas import DataFrame,Series
from sklearn.feature_extraction.text import TfidfVectorizer
from time import time
import matplotlib.pyplot as plt


# In[2]:


import os
os.chdir(os.getcwd()) 


# In[3]:


df=pd.read_csv('data.csv')#opening the desired csv file within the zipped file and loading 
#it as desired data frame.


# In[4]:


df.head()


# In[5]:


print("The data-set has %d rows and %d columns"%(df.shape[0],df.shape[1])) #using the shape attribute of the dataframe object.
#where the first element shows the number of rows and the second element shows the number of columns.


# In[6]:


df.tail(10) #to display last ten rows:


# In[7]:
##Hunting Missing Values:

#data-sets usually have missing values in them for a variety of reasons. In Numpy, missing values are represented as NaN and 
#using the following routine, we can quickly check if there is any column in the loaded dataframe that has the missing values.
#in the event of finding missing values, we can drill down further and can adjust out subsequent strategies accordingly:


#from __future__ import print_function #my current version of python doesn't have the functionality that I intend to use in the
#following lines of codes so thus importing fresh and new print function "from the future"
print (df.columns) 

#to calculate number of missing values in each column. True values are coerced as 1 and False as 0 and thus I used that
#fact in "sum" function to calculate how many missing values are there in each column:
for col_name in df.columns:
    print (col_name,end=": ")
    print (sum(df[col_name].isnull()))


# In[8]:

## Handling duplicate values in the dataframe:
    
sum(df.duplicated()) #5 rows are duplicate


# In[9]:


df.ix[df.duplicated(keep='first'),]


# In[14]:


#creating a new dataframe with the duplicate rows removed from the original dataframe. We can also use drop_duplicate(inplace=True)
#parameter as well.
jobs_df=df.drop_duplicates()


# In[15]:


#now verifying whether there are still duplicate values in our dataframe or not:
sum(jobs_df.duplicated()) 
#also the size of our dataframe has also reduced as:
jobs_df.shape
#previously we had  19997 rows and now with 5 duplicate rows removed, we have 19992 rows left in our dataframe:


# In[16]:


category_counter={x:0 for x in set(jobs_df['label'])}


# In[17]:


for each_cat in jobs_df['label']:
    category_counter[each_cat]+=1


# In[18]:


print(category_counter)


# In[19]:


corpus=jobs_df.document


# In[20]:


all_words=[w.split() for w in corpus]
all_flat_words=[ewords for words in all_words for ewords in words]
from nltk.corpus import stopwords
all_flat_words_ns=[w for w in all_flat_words if w not in stopwords.words("english")]
#removing all the stop words from the corpus
set_nf=set(all_flat_words_ns)
#removing all duplicates


# In[21]:


print("Number of unique vocabulary words in the document column of the dataframe: %d"%len(set_nf))


# In[22]:


porter=nltk.PorterStemmer()
for each_row in jobs_df.itertuples():
    m1=map(lambda x:  str(x.encode('utf-8')),(each_row[1]+' '+each_row[2]).lower().split())
    #for each row,converts them to lower case. Also converts them to unicode because NLTK's porter stemmer expects unicode data.
    m2=map(lambda x: porter.stem(x),m1)
    #Using Porter Stemmer in NLTK, stemming is performed on the str created in previous step.
    jobs_df.loc[each_row[0],'document_New']=' '.join(m2)
#a derived column is created and the pre-processed string is stored in that column for each row.
#here's a sneak-peek of the dataset with newly created column that contains our processed text. Its still in one column and 
#in subsequent steps, I will create a document term matrix using TFIDF mechanism to create features for classifiers:


# In[23]:


jobs_df.head()


# In[24]:


from sklearn.feature_extraction.text import CountVectorizer
# countvectorizer Convert a collection of text documents to a matrix of token counts
count_vect = CountVectorizer()
data_counts = count_vect.fit_transform(jobs_df.document_New)
data_counts.shape
#[n_samples, n_features]


# In[25]:


# TF-IDF
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
data_tfidf = tfidf_transformer.fit_transform(data_counts)
data_tfidf.shape
 #TF:Total words, in each document.


# In[26]:


training_time_container={'b_naive_bayes':0,'mn_naive_bayes':0,'random_forest':0,'linear_svm':0}
prediction_time_container={'b_naive_bayes':0,'mn_naive_bayes':0,'random_forest':0,'linear_svm':0}
accuracy_container={'b_naive_bayes':0,'mn_naive_bayes':0,'random_forest':0,'linear_svm':0}


# In[27]:


from sklearn.model_selection import train_test_split
variables = data_tfidf
#considering the TFIDF features as independent variables to be input to the classifier.
labels = jobs_df.label
#considering the category values as the class labels for the classifier.
variables_train, variables_test,  labels_train, labels_test  =   train_test_split(variables, labels, test_size=.3)
#splitting the data into random training and test sets for both independent variables and labels.


# In[28]:


#analyzing the shape of the training and test data-set:
print('Shape of Training Data: '+str(variables_train.shape))
print('Shape of Test Data: '+str(variables_test.shape))


# In[48]:

from sklearn.ensemble import RandomForestClassifier

rf_classifier=RandomForestClassifier(n_estimators=100)
t0=time()
rf_classifier=rf_classifier.fit(variables_train,labels_train)
training_time_container['random_forest']=time()-t0
print("Training Time: %fs"%training_time_container['random_forest'])
t0=time()
rf_predictions=rf_classifier.predict(variables_test)
prediction_time_container['random_forest']=time()-t0
print("Prediction Time: %fs"%prediction_time_container['random_forest'])
accuracy_container['random_forest']=sklearn.metrics.accuracy_score(labels_test, rf_predictions)
print ("Accuracy Score of Random Forests Classifier: ")
print(accuracy_container['random_forest'])

# In[1]:




# In[90]:


def text_Classification_RF(raw_docx):
    content = raw_docx
    content=[content]
    print(content)
    X_new_count=count_vect.transform(content)
    X_new_tfidf= tfidf_transformer.fit_transform(X_new_count)
    predicted=rf_classifier.predict(X_new_tfidf)
    for x in predicted:
      print(x)
  
    return x 
