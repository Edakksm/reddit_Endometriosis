import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem import PorterStemmer


train_df = pd.read_csv('../data/train_Jan7.csv')
test_df = pd.read_csv('../data/test_Jan7.csv')

x_train, y_train = train_df[['Message', 'Status']].values.T
y_train = y_train.astype('int')
x_train = x_train.astype('str')


# Same for test
x_test, y_test = test_df[['Message', 'Status']].values.T
y_test = y_test.astype('int')
x_test = x_test.astype('str')

vect = CountVectorizer(min_df=2, ngram_range=(2,2))
X_train = vect.fit(x_train).transform(x_train)
print(X_train[1].toarray())
X_test = vect.transform(x_test)

tfidf = TfidfTransformer(use_idf=True, norm='l2',smooth_idf=True)
tfidf.transform(X_train).toarray()

porter = PorterStemmer()

print('Len of vocabulary is {0}'.format(len(vect.vocabulary_)))
print(len(vect.get_feature_names()))

param_grid = {'C':[0.001,0.01,0.1,1,10]}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, y_train)
lr = grid.best_estimator_
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
one_pred = [y for y in y_pred if y == 1]
one_actual = [y for y in y_test if y == 1]
print(len(one_pred))
print(len(one_actual))
#print('Score is {0}'.format(lr.score(X_test, y_test)))
#for i in range(len(y_pred)):
 #   print('Actual {0}:Pred {1}'.format(y_test[i], y_pred[i]))

