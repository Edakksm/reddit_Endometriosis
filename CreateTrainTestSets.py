import os
import pandas as pd
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords

# For each post in positive, get the data. Then check in neg for posts on the same day
# For each post in poitive, select the txt that has almost the same length as positive
import collections
def createDataFrame(path):
    labels = {'pos4':1, 'neg4':0}
    #df = pd.DataFrame()
    reddit = []
    raw_data = collections.defaultdict(dict)
    try:
        for s in ('pos4','neg4'):
            p = os.path.join(path, s)
            for file in os.listdir(p):
                with open(os.path.join(p,file),'r',encoding='utf-8') as ifile:
                    lines = ifile.readlines()
                lines = [line.rstrip('\n') for line in lines]
                msg_len = 0
                msg = ' '
                reddit = set()
                for line in lines:
                    txt = line.split('|')
                    username = file[0:-4]
                    reddit.add(txt[0])
                    try:
                        if int(txt[2]) > 0:
                            msg_len = msg_len + int(txt[2])
                            msg = msg + txt[3]
                          #  raw_data[username]['Reddit'] = txt[0]
                          #  raw_data[username]['CreatedDate'] = txt[1]
                           # raw_data[username]['MessageLength'] = txt[2]
                            #raw_data[username]['Message'] = txt[3]
                            #raw_data[username]['Status'] = labels[s]
                    except Exception as ex:
                        print(ex)
                raw_data[username]['Reddit'] = reddit
              #  raw_data[username]['CreatedDate'] = txt[1]
                raw_data[username]['MessageLength'] = msg_len
                raw_data[username]['Message'] = msg
                raw_data[username]['Status'] = labels[s]
                #raw_data[username]['NoOfMsg'] = int(raw_data[username]['NoOfMsg']) + 1
    except Exception as ex:
        print(ex)
    df = pd.DataFrame.from_dict(raw_data, orient='index')
    return df


df = createDataFrame(os.getcwd())
df.to_csv('train_Mar19.csv')
#X_train = df.loc[:10, ['username', 'post']].values
#y_train = df.loc[:10,'status'].values
#X_test = df.loc[11:, ['username','post']].values
#y_test = df.loc[11:,'status'].values


#posts = df.post.str.cat(sep = ' ')
#tokens = word_tokenize(posts)
#vocabulary = set(tokens)
#print(vocabulary)
#stopwords = set(stopwords.words('english'))
#tokens = [w for w in tokens if w not in stopwords]
#frequency_dist = nltk.FreqDist(tokens)
#print(sorted(frequency_dist,key=frequency_dist.__getitem__, reverse=True))


#print(X_train.shape)
#print(X_test.shape)

print('comp;eted')
