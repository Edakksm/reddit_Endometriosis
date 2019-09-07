from Init import Init
from collections import defaultdict
from collections import OrderedDict
from datetime import datetime
from datetime import date
from dateutil.relativedelta import relativedelta
import urllib.request
import json
import itertools
import heapq
from heapq import heappush, heappop
import collections
import arrow
import time
from dateutil import parser
import pandas as pd

import csv

def getEndoUsers():
    init = Init()
    sub_reddits = init.subReddit.split(',') # Subreddits would be endo/adenomyosis/endometriosis
    users = defaultdict(date)
    for sub_red in sub_reddits:
        users = getEndoSubredditUser(users, sub_red)

    return users


def getEndoSubredditUser(users, sub_red):
    start_date = datetime.now()
    init_date = parser.parse('Apr 01 2017 12:00AM')
    user_data = getEndoBatchUsers2(sub_red, start_date, init_date)
    for batch in user_data:
        for items in batch:
            author = items['author']
            created_date = datetime.utcfromtimestamp(items['created_utc'])
            if author not in users:
                users[author] = created_date
            else:
                if created_date < users[author]:
                    users[author] = created_date
   # print(len(users))
    return users


def getEndoBatchUsers2(sub_red, start_date, init_date):
    users = []
   # start_date = datetime.now()
    start_date_epoch = time.mktime(start_date.timetuple())
    end_date = datetime.now() - relativedelta(days=int(10))
    end_date_epoch = time.mktime(end_date.timetuple())
  #  init_date = parser.parse(init.initDate)
    while end_date > init_date:
         api_comment_url = 'https://api.pushshift.io/reddit/search/comment/?subreddit=' + sub_red + '&before='+ str(int(start_date_epoch)) + '&after=' + str(int(end_date_epoch)) + '&size=5000&sort=desc'
         url = urllib.request.urlopen(api_comment_url)
         user_data = json.loads(url.read().decode())
         users.append(user_data['data'])
         api_submission_url = 'https://api.pushshift.io/reddit/search/submission/?subreddit=' + sub_red + '&before='+ str(int(start_date_epoch)) + '&after=' + str(int(end_date_epoch)) + '&size=5000&sort=desc'
         url = urllib.request.urlopen(api_submission_url)
         user_data = json.loads(url.read().decode())
         users.append(user_data['data'])
         start_date = end_date
         start_date_epoch = end_date_epoch
         end_date = start_date - relativedelta(days=int(10))
         end_date_epoch = time.mktime(end_date.timetuple())
    return users

def getRedditsOfEndoUsers(users):
    import csv
    users_non_endo_submissions = defaultdict(list)
    try:
        for user_name, first_submission_date in users.items():
            count = 0
            url = urllib.request.urlopen("https://api.pushshift.io/reddit/comment/search?metadata=true&before=0d&limit=1000&sort=desc&author=" + user_name)
            user_data = json.loads(url.read().decode())
            comment_num = user_data["metadata"]["results_returned"]
            total = user_data["metadata"]["total_results"]
            user_endo_comment_createdDate = users[user_name]['date']
            three_months = datetime.strptime(user_endo_comment_createdDate, '%m/%d/%Y %H:%M') - relativedelta(months=int(3))
            six_months = datetime.strptime(user_endo_comment_createdDate, '%m/%d/%Y %H:%M') - relativedelta(months=int(9))
            users_non_endo_submissions[user_name] = defaultdict(int)
            for j in range(len(user_data['data'])):
                count = count + 1
                comment = user_data["data"][j]
                created_date = datetime.utcfromtimestamp(comment['created_utc'])
                body = comment['body']
                subreddit = comment['subreddit']
                if created_date < three_months and created_date > six_months:
                        users_non_endo_submissions[user_name][subreddit] += 1
            print(user_name)
            print(count)

    except Exception as e:
        print(e)

    return users_non_endo_submissions

   # return sort_non_subreddits(users_non_endo_submissions, len(users_non_endo_submissions))

def sort_non_subreddits(non_subreddits, n):
    if non_subreddits is not None and len(non_subreddits)>0:
        try:
            heap = [(-value, key) for key,value in non_subreddits.items()]
            most_common_subreddits = heapq.nsmallest(n, heap)
            most_common_subreddits = [(key, -value) for value,key in most_common_subreddits]
            return most_common_subreddits
        except Exception as e:
            init.logger.writeError(e.message)
            return non_subreddits

def createPositiveNegativeCases(endo_users, non_subreddits):
    for reddit in non_subreddits:
         pos,positive_users, negative_users = process(endo_users, pos,positive_users, negative_users,reddit)

def WriteStatistics_NonEndoReddits(nonEndoSubReddits, endo_user_count):
    import  csv
    non_reddit = collections.defaultdict(dict)
    s = ''
    with open('statistics_2018.csv', 'w', newline='') as f:
        filewriter = csv.writer(f)
        for subreddit, count in nonEndoSubReddits:
            filewriter.writerow([subreddit,count,(count/endo_user_count) * 100])

def process():
    # Read the endo users file,
    # Read the non-endo subreddits file
    # Create positive and negative cases
        # positive - users found in both files
        # negative - users found in non-endo sub but not an endo user
    endo_users = pd.read_csv('endoUsers.csv')


def process1(users, pos,positive_users, negative_users, n_subRed):
        sub = init.ConnectToReddit()
        cnt = 0
        endo = sub.subreddit(n_subRed).hot(limit=None)
        for hot_msg in endo:
            if not hot_msg.stickied:
                created_date = datetime.utcfromtimestamp(hot_msg.created)
                hot_msg.comments.replace_more(limit=5) #
                comments = hot_msg.comments.list()
                for cmt in comments:
                   cnt = cnt + 1
                   if hasattr(cmt, 'author') and hasattr(cmt.author, 'name'):
                     start_time = datetime.now()
                     if cmt.author.name not in positive_users and cmt.author.name not in negative_users:
                         if cmt.author.name in users:
                             user_endo_comment_createdDate = users[cmt.author.name]
                             three_months = user_endo_comment_createdDate - relativedelta(months=int(init.start_duration))
                             six_months = user_endo_comment_createdDate - relativedelta(months=int(12))
                             if created_date.date() < three_months and created_date.date() > six_months:
                                 positive_users[cmt.author.name] = 1
                             if created_date.date() < user_endo_comment_createdDate:
                                 pos[cmt.author.name] = 1
                          #   dates[cmt.author.name]['3months'] = three_months
                           #  dates[cmt.author.name]['createdDate'] = created_date.date()
                            # dates[cmt.author.name]['6months'] = six_months
                             #with open('dates2.csv','a') as f:
                              #   f.write(str(dates[cmt.author.name]))
                               #  f.write('\n')
                         else:
                             negative_users[cmt.author.name] = 1
                         end_time = datetime.now()
                         s = (end_time - start_time).seconds
                 #    print(s)
        print('len of pos user is {0}'.format(len(pos)))
        return  pos,positive_users, negative_users

init = Init()
#endo_users = getEndoUsers()
#newusers = {k:v for (k, v) in endo_users.items() if v.year == 2018}
#with open('endoUsers_2018.csv', 'w', newline='') as f:
 #   filewriter = csv.writer(f)
  #  for k, v in newusers.items():
   #     filewriter.writerow([k, v])

endo_users = pd.read_csv('endoUsers_2018.csv').set_index('user_name').T.to_dict()
#endo_users.set_index('user_name').T.to_dict('list')
#endo_users.set_index('user_name').T
subreddits = getRedditsOfEndoUsers(endo_users)
non_subreddits = defaultdict(int)

if subreddits is not None and len(subreddits) > 0:
    try:
        for i,j in subreddits.items():
            for k,v in j.items():
                non_subreddits[k] += 1
     #   init.fileWriter.writeData(users_non_endo_submissions)
    except Exception as e:
        init.logger.writeError(e.message)
non_endo_subreddits = [(reddit,count) for reddit, count in non_subreddits.items() if reddit.lower() not in ['endo','endometriosis']]
non_endo_subreddits = sort_non_subreddits(non_subreddits, len(non_subreddits))
WriteStatistics_NonEndoReddits(non_endo_subreddits,len(endo_users))
#non_endo_subreddit_users = getNonEndoSubredditUsers()

#positive_users, negative_users = CreatePositiveNegativeUsers(endo_users, non_subreddits)
