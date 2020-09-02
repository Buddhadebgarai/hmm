import pandas as pd
import numpy as np
import os
from utils import state_calculator

current_path = os.getcwd()
filename = "sachin_rt_tweets.csv"
df2 = pd.read_csv(f"{current_path}\\data\\state\\{filename}").dropna()
df=df2[2:20]
print("Generating state matrix...")

LIKE_LOW_RANGE = 100
LIKE_MID_RANGE = 1000

RETWEET_LOW_RANGE = 0.75
RETWEET_MID_RANGE = 1.75

TWEET_LOW_RANGE = 0.7
TWEET_MID_RANGE = 0.25

like = df["sd_like"]
retweet = df["sd_retweet"]
tweet = df["sd_tweet"]

like_state = state_calculator(like, LIKE_LOW_RANGE, LIKE_MID_RANGE)
retweet_state = state_calculator(retweet, RETWEET_LOW_RANGE, RETWEET_MID_RANGE)
tweet_state = state_calculator(tweet, TWEET_LOW_RANGE, TWEET_MID_RANGE)

# d = {
#     'month(days)': df["month(days)"],
#     'like_state': like_state,
#     'tweet_state': tweet_state,
#     'retweet_state': retweet_state
# }
# df2 = pd.DataFrame(data=d)

STATE = ['LLL', 'LLM', 'LLH', 'LML', 'LMM', 'LMH', 'LHL', 'LHM', 'LHH', 'MLL',
         'MLM', 'MLH', 'MML', 'MMM', 'MMH', 'MHL', 'MHM', 'MHH', 'HLL',
         'HLM', 'HLH', 'HML', 'HMM', 'HMH', 'HHL', 'HHM', 'HHH']

comparison_arr = []
for i in range(0, len(like_state)):
    s = f"{like_state[i]}{retweet_state[i]}{tweet_state[i]}"
    comparison_arr.append(s)

state_arr = []
for i in comparison_arr:
    if i in STATE:
        state_arr.append(STATE.index(i))

# d3 = {
#     "month(days)": df["month(days)"],
#     "state": state_arr,
# }
# df3 = pd.DataFrame(data=d3)
# df3.to_csv(f"{current_path}\\output\\state\\{filename}")
# print("State output created.")


trans_dict = {}
for i in range(1, len(state_arr) - 1):
    s = state_arr[i], state_arr[i + 1]
    if s in trans_dict:
        trans_dict[s] += 1
    else:
        trans_dict[s] = 1
# print(trans_dict)


a = np.zeros((27, 27), np.int64)
for key in trans_dict.keys():
    a[key] = trans_dict[key]

row_count = 27
state_trans_matrix = a.copy() + 1
sum_of_rows = np.sum(a, axis=1) + 27

STATE_MATRIX = np.zeros((27, 27))

for i in range(row_count):
    STATE_MATRIX[i] = state_trans_matrix[i] / sum_of_rows[i]

print("State matrix generated.")
