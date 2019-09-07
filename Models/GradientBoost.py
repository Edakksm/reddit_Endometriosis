import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

train_df = pd.read_csv('../data/train_Jan7.csv')
test_df = pd.read_csv('../data/test_Jan7.csv')

x_train, y_train = train_df[['Message', 'Status']].values.T
y_train = y_train.astype('int')
x_train = x_train.astype('str')

print(x_train.shape)

# Same for test
x_test, y_test = test_df[['Message', 'Status']].values.T
y_test = y_test.astype('int')
x_test = x_test.astype('str')

print(x_test.shape)
vect = CountVectorizer(min_df=2, ngram_range=(2,2))
X_train = vect.fit(x_train).transform(x_train)
print(X_train[1].toarray())
X_test = vect.transform(x_test)

print('Len of vocabulary is {0}'.format(len(vect.vocabulary_)))
print(len(vect.get_feature_names()))

param_grid = {'n_estimators':[200,100, 50],'max_depth':[5,6,7,8],'min_samples_leaf':[10,50,100],'max_features':['sqrt','log2']}
grid = GridSearchCV(GradientBoostingClassifier(), param_grid, cv=5)
grid.fit(X_train, y_train)
clf = grid.best_estimator_
print(grid.best_params_)
clf.fit(X_train,y_train)

# For test set
y_test_pred = clf.predict(X_test)
recall = recall_score(y_true=(y_test), y_pred=y_test_pred)
precision = precision_score(y_true=(y_test), y_pred=y_test_pred)
print('Recall of test set: {0}]'.format(recall))
print('Precision of test set :{0}'.format(precision))

# For training set
y_train_pred = clf.predict(X_train)
recall = recall_score(y_true=(y_train), y_pred=y_train_pred)
precision = precision_score(y_true=(y_train), y_pred=y_train_pred)
print('Recall of train set: {0}]'.format(recall))
print('Precision of train set :{0}'.format(precision))

for i in range(len(y_test_pred)):
    if y_test_pred[i] == 1:
        print(x_test[i])
        print('Actual {0}:Pred {1}'.format(y_test[i], y_test_pred[i]))

