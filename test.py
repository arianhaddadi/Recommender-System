import text2emotion as te
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json


# df = pd.DataFrame(columns=["username", "item", "rating", "helpfulness", "review"])
#
# with open("data/reviews.json") as file:
#     for line in file:
#         line = json.loads(line)
#         df = df.append({
#             "username": line["reviewerID"],
#             "item": line["asin"],
#             "rating": line["overall"],
#             "helpfulness": 2 * line["helpful"][0] - line["helpful"][1],
#             "review_emotions": te.get_emotion(line["reviewText"])
#         }, ignore_index=True)
#
# df.to_csv("data/amazon_reviews.csv", index=False)

# from sklearn.datasets import load_boston
# import pandas as pd
# import xgboost as xgb
# from sklearn.metrics import mean_squared_error
# import numpy as np
# from sklearn.model_selection import train_test_split
#
# boston = load_boston()
# data = pd.DataFrame(boston.data)
# data.columns = boston.feature_names
# data['PRICE'] = boston.target
# X, y = data.iloc[:, :-1], data.iloc[:, -1]
# data_dmatrix = xgb.DMatrix(data=X, label=y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
#
# xg_reg = xgb.XGBRegressor(objective='reg:linear', colsample_bytree=0.3, learning_rate=0.1,
#                           max_depth=5, alpha=10, n_estimators=10)
# xg_reg.fit(X_train, y_train)
# preds = xg_reg.predict(X_test)
#
# rmse = np.sqrt(mean_squared_error(y_test, preds))

#
# params = {"objective": "reg:linear", 'colsample_bytree': 0.3, 'learning_rate': 0.1,
#           'max_depth': 5, 'alpha': 10}
#
# cv_results = xgb.cv(dtrain=data_dmatrix, params=None, nfold=3,
#                     num_boost_round=50, early_stopping_rounds=10, metrics="rmse", as_pandas=True, seed=123)
# print(cv_results.head())


# df1 = pd.read_csv("data/polarity.csv")
# df2 = pd.read_csv("data/amazon_reviews.csv")
# sen_min, sen_max = df1["oss"].min(), df1["oss"].max()
# # df1["oss"].apply(lambda x: (x + sen_min) / (sen_max - sen_min))
# df2["sentiment"] = df1["oss"].apply(lambda x: 5 * (x + abs(sen_min)) / (sen_max - sen_min))
#
# df2.to_csv("data/amazon_reviews.csv")

# df1 = pd.read_csv("data/amazon_reviews.csv")
# df2 = pd.DataFrame()
# sen_min, sen_max = df1["helpfulness"].min(), df1["helpfulness"].max()
# # df1["oss"].apply(lambda x: (x + sen_min) / (sen_max - sen_min))
# df1["helpfulness"] = df1["helpfulness"].apply(lambda x: 5 * x)
# df1.to_csv("data/amazon_reviews.csv", index=False)
# df2 = pd.read_csv("data/amazon_reviews.csv").drop(["Unnamed: 0"], axis=1)
# df2.to_csv("data/amazon_reviews.csv", index=None)
# exit()

# df = pd.DataFrame(columns=["a", "b"])
# df["a"] = [2, 3]
# df["b"] = [4, 5]
#
# f1 = df.iloc[0]
# f2 = df["a"]
# f3 = df.loc[0]
# f4 = df.loc[df["a"] == 2]
# pass



# all_reviews = pd.read_csv("data/amazon_reviews.csv")
#
# def getReviewsForIteration(reviews, i, K):
#     reviews_len = len(reviews)
#     base_start = int(i * reviews_len / K)
#     base_end = int((i + 1) * reviews_len / K)
#     iteration_reviews = reviews.iloc[:base_start]
#     iteration_reviews = iteration_reviews.append(reviews.iloc[base_end:])
#     return iteration_reviews
#
# K = 5
# for i in range(K):
#     df1 = getReviewsForIteration(all_reviews, i, K)
#     df2 = all_reviews.iloc[int(i * len(all_reviews) / K): int((i + 1) * len(all_reviews) / K)]
#
#     print(f"len all = {len(all_reviews)}, len df1 = {len(df1)}, len df2 = {len(df2)}")
#     # print(f"indexes df1 = {list(df1.index)[0]}")
#     pass

# df1 = pd.read_csv("data/amazon_reviews.csv")
# df2 = pd.read_csv("data/xgb_dataset_personality.csv")
# df3 = pd.DataFrame(columns=["username", "item", "helpfulness", "deviation", 'Happy', 'Angry', 'Surprise', 'Sad', 'Fear', "rating"])
#
# for i in range(len(df1)):
#     d1 = df1.iloc[i]
#     d2 = df2.iloc[i]
#     # js = d1["review_emotions"].replace("'", '"')
#     emot = json.loads(d1["review_emotions"].replace("'", '"'))
#
#     df3 = df3.append({
#         "username": d1["username"],
#         "item": d1["item"],
#         "helpfulness": d1["helpfulness"],
#         "deviation": d2["deviation"],
#         'Happy': emot["Happy"],
#         'Angry': emot["Angry"],
#         'Surprise': emot["Surprise"],
#         'Sad': emot["Sad"],
#         'Fear': emot["Fear"],
#         "rating": d2["rating"]
#         }, ignore_index=True)
# df3.to_csv("data/xgb_dataset_emotion.csv", index=None)
# df = pd.read_csv("data/amazon_reviews.csv")
# users = df["username"].unique()
# minimum = math.inf
# countmin = 0
# for user in users:
#     leng = len(df.loc[df["username"] == user])
#     if leng < minimum:
#         minimum = leng
#         countmin = 0
#     elif leng == minimum:
#         countmin += 1


    # user_review = review_df.loc[review_df["username"] == user]
    # user_review = user_review.loc[user_review["item"] == item]

    # neighbour_review = review_df.loc[review_df["username"] == neighbour]
    # neighbour_review = neighbour_review.loc[neighbour_review["item"] == item]
#     user_pers = personality_df.loc[personality_df["username"] == user].iloc[0]
#     neigh_pers = personality_df.loc[personality_df["username"] == neighbour].iloc[0]
#
#     out_df = out_df.append({
#         "Extraversion": user_pers["Extraversion"],
#         "Neuroticism": user_pers["Neuroticism"],
#         "Agreeableness": user_pers["Agreeableness"],
#         "Conscientiousness": user_pers["Conscientiousness"],
#         "Opennes": user_pers["Opennes"],
#         "neigh_Extraversion": neigh_pers["Extraversion"],
#         "neigh_Neuroticism": neigh_pers["Neuroticism"],
#         "neigh_Agreeableness": neigh_pers["Agreeableness"],
#         "neigh_Conscientiousness": neigh_pers["Conscientiousness"],
#         "neigh_Opennes": neigh_pers["Opennes"]
#     }, ignore_index=True)
#
# report_df["Extraversion"] = out_df["Extraversion"]
# report_df["Neuroticism"] = out_df["Neuroticism"]
# report_df["Agreeableness"] = out_df["Agreeableness"]
# report_df["Conscientiousness"] = out_df["Conscientiousness"]
# report_df["Opennes"] = out_df["Opennes"]
# report_df["neighbour_Extraversion"] = out_df["neigh_Extraversion"]
# report_df["neighbour_Neuroticism"] = out_df["neigh_Neuroticism"]
# report_df["neighbour_Agreeableness"] = out_df["neigh_Agreeableness"]
# report_df["neighbour_Conscientiousness"] = out_df["neigh_Conscientiousness"]
# report_df["neighbour_Opennes"] = out_df["neigh_Opennes"]
#
# report_df.to_csv("data/amazon_report_no_neighbours.csv", index=False)



# df = pd.DataFrame(columns=["username", "num_of_ratings", "Extraversion", "Neuroticism", "Agreeableness", "Conscientiousness", "Opennes"])

# reviews = pd.read_csv("data/amazon_reviews.csv")
# personality = pd.read_csv("data/Amazon_Normalized_pers_scores.csv")

# users = reviews["username"].unique()

# for user in users:
#     user_pers = personality.loc[personality["username"] == user].iloc[0]
#     user_reviews_num = len(reviews.loc[reviews["username"] == user])

#     df = df.append({
#         "username": user,
#         "num_of_ratings": user_reviews_num,
#         "Extraversion": user_pers["Extraversion"],
#         "Neuroticism": user_pers["Neuroticism"],
#         "Agreeableness": user_pers["Agreeableness"],
#         "Conscientiousness": user_pers["Conscientiousness"],
#         "Opennes": user_pers["Opennes"]
#     }, ignore_index=True)

# df.to_csv("charts.csv", index=False)


# df = pd.read_csv("charts.csv")
# for pers in ["Extraversion", "Neuroticism", "Agreeableness", "Conscientiousness", "Opennes"]:

#     df_sorted = df.sort_values(pers)
#     values = df_sorted[pers].unique()

#     x_values = values
#     y_values = list()

#     for value in values:
#         df_sorted_value = df_sorted.loc[df_sorted[pers] == value]
#         average = df_sorted_value["num_of_ratings"].mean()

#         y_values.append(average)

#     plt.plot(x_values, y_values)
#     plt.title(pers)
#     plt.xlabel(f'{pers} values')
#     plt.ylabel("Num of Ratings")
#     plt.show()


# for pers in ["Extraversion", "Neuroticism", "Agreeableness", "Conscientiousness", "Opennes"]:
#     corr = df[[pers, "num_of_ratings"]].corr()
#     print(f'personality = {pers}, corr={corr[pers].iloc[1]}')

# with open("data/amazon_train_set_dict.json") as file:
#     dic = json.loads(file.readline())

# for user in dic:
#     print(dic[user]["ratings_average"])

# df = pd.read_excel("data/reviews.xlsx")
# df = df.loc[df["type"] == "Hotels"][["username", "text", "helpfulness", "rating", "taObject"]].dropna()
# df.to_csv("data/reviews.csv")


# reviews = pd.read_csv("data/reviews.csv")

# reviews_text = pd.DataFrame(columns=["review_emotions"])

# counter = 0
# for row in reviews.itertuples(index=False):
#     text = row[1]
#     print(counter)
#     counter += 1
#     reviews_text = reviews_text.append({
#         "review_emotions": te.get_emotion(text)
#     }, ignore_index=True)

# reviews["review_emotions"] = reviews_text["review_emotions"]
# reviews = reviews.drop("text", axis=1)
# reviews.to_csv("data/reviews2.csv")

# reviews = pd.read_csv("data/reviews.csv")
# users = reviews["username"].unique()
# # chart_info = list()
# reviews_map = dict()
# reviews_num_map = dict()

# for user in users:
#     user_reviews = reviews.loc[reviews["username"] == user]
#     chart_info.append(/[len(user_reviews), user])
#     num_of_reviews = len(user_reviews)
#     if num_of_reviews not in reviews_map:
#         reviews_map[num_of_reviews] = list()
#         reviews_num_map[num_of_reviews] = 0
#     reviews_map[num_of_reviews].append(user)
#     reviews_num_map[num_of_reviews] += 1

# keys = list(reviews_map.keys())
# keys.sort()
# reviews_map_temp = dict()
# for num in keys:
#     reviews_map_temp[num] = reviews_num_map[num] * num
# with open("reviews_map.json", "w") as file:
#     file.write(json.dumps(reviews_map))
# with open("reviews_num_map.json", "w") as file:
#     file.write(json.dumps(reviews_map_temp))

# users, num_reviews = list(), list()

# for num, user in chart_info:
#     users.append(user)
#     num_reviews.append(num)

# print(num_reviews[0], num_reviews[len(num_reviews)-1])
# plt.figure(figsize=(20, 10))
# plt.plot(users, num_reviews)
# plt.title("Num of Reviews Per User")
# plt.xlabel("Users")
# plt.ylabel("Num of Reviews")
# plt.savefig("1.png")

# with open("reviews_num.txt") as file:
#     revs_list = file.readline()

# revs_list = eval(revs_list)
# num_rev = len(revs_list)
# ranges = list()
# for i in range(5):
#     range_rev_list = revs_list[int(i * num_rev / 5): int((i + 1) * num_rev / 5)]
#     ranges.append(range_rev_list)

# with open("reviews_num_ranges.txt", "w") as file:
#     for elem in ranges:
#         file.write(str(elem) + "\n")
# with open("reviews_num_ranges.txt") as file:
#     lines = file.readlines()
# with open("reviews_map.json") as file:
#     reviews_map = json.loads(file.readline())
# reviews = pd.read_csv("data/amazon_reviews.csv")
# reviews_ranges = list()

# for line in lines:
#     range_list = eval(line)
#     reviews_range = pd.DataFrame()
#     for num in range(range_list[0], range_list[1] + 1):
#         if str(num) not in reviews_map:
#             continue
#         users_with_range = reviews_map[str(num)]
#         for user in users_with_range:
#             reviews_range = reviews_range.append(reviews.loc[reviews["username"] == user], ignore_index=True)
#     reviews_ranges.append(reviews_range)

# for i, elem in enumerate(reviews_ranges):
#     elem.to_csv(f"amazon_reviews_{i}.csv", index=False)


# df = pd.read_csv("data/reviews.csv")
# users = df["username"].unique()
# hotels = df["item"].unique()
# num_of_reviews = len(df)

# reviews_map = dict()
# hotels_map = dict()
# min_rev_num, max_rev_num = math.inf, -math.inf
# min_hotel_num, max_hotel_num = math.inf, -math.inf

# for user in users:
#     user_reviews = df.loc[df["username"] == user]
#     user_reviews_length = len(user_reviews)
#     min_rev_num = min(min_rev_num, user_reviews_length)
#     max_rev_num = max(max_rev_num, user_reviews_length)


# for hotel in hotels:
#     hotel_reviews = df.loc[df["item"] == hotel]
#     hotel_rev_num = len(hotel_reviews)
#     min_hotel_num = min(min_hotel_num, hotel_rev_num)
#     max_hotel_num = max(max_hotel_num, hotel_rev_num)

# print(len(users), len(hotels), num_of_reviews, min_hotel_num, max_hotel_num, min_rev_num, max_rev_num)

# df_out = pd.DataFrame(columns=["username", "item", "rating", "helpfulness"])

# df = pd.read_excel("data/reviews.xlsx")
# df = df.loc[(df["type"] == "Hotels") | (df["type"] == "Restaurants")]

# for row in df.itertuples(index=False):
#     username = row[1]
#     rating = row[6]
#     helpfulness = row[7]
#     item = row[9]
#     df_out = df_out.append({
#         "username": username,
#         "item": item,
#         "rating": rating,
#         "helpfulness": helpfulness
#     }, ignore_index=True)

# df_out.to_csv("data/reviews2.csv", index=False)

# df = pd.read_csv("data/reviews.csv")
# blacklist = list()
# users = df["username"].unique()
# for user in users:
#     user_reviews = df.loc[df["username"] == user]
#     if len(user_reviews) < 5:
#         blacklist.append(user)

# print(len(df))

# for user in blacklist:
#     user_reviews = df.loc[df["username"] == user]
#     # for row in user_reviews.itertuples():
#     df.drop(user_reviews.index.values, inplace=True)

# df.to_csv("data/reviews.csv", index=False)

# df = pd.read_csv("data/reviews.csv")
# df_full = pd.read_excel("data/reviews.xlsx")
# texts = list()

# for row in df.itertuples():
#     try:
#         text = df_full.loc[(str(df_full["username"]) == row[1]) & (df_full["taObject"] == row[2])]["text"].iloc[0]
#     except Exception as e:
#         pass
#     texts.append(text)
#
# df["review_text"] = texts

# df.to_csv("data/reviews.csv", index=False)

# reviews = pd.read_csv("data/reviews.csv")
# polarity = pd.read_csv("data/polarity.csv")
# sentiments = polarity["oss"]
# min_sen, max_sen = min(sentiments), max(sentiments)
# sentiments = np.array(sentiments)
# sentiments = ((sentiments - min_sen) / max_sen) * 5
# sentiments = list()
# for row in reviews.itertuples(index=False):
#     username = row[0]
#     item = row[1]
#
#     sent = polarity.loc[(polarity["username"] == username) & (polarity["taObject"] == item)]
#     if len(sent) != 0:
#         sentiments.append(sent["oss"].iloc[0])
#     else:
#         sentiments.append(-63)
#
# min_sen, max_sen = min(sentiments), max(sentiments)
# sentiments = np.array(sentiments)
# sentiments = ((sentiments - min_sen) / max_sen) * 5
#
# reviews["sentiment"] = sentiments
# reviews.fillna(0, inplace=True)
# reviews.to_csv("data/reviews.csv", index=False)

# df = pd.read_excel("data/reviews.xlsx")
# df = df.loc[df["type"] == "Restaurants"][["username", "helpfulness", "taObject", "rating", "text"]]
# print(len(df))
# df.to_csv("temp.csv", index=False)
# K = 5
# for i in range(K):
#     df_temp = df.iloc[i * int(len(df) / K) : (i + 1) * int(len(df) / K)]
#     df_temp.to_csv(f"temp_{i}.csv", index=False)

# df = pd.read_csv("data/reviews.csv")
# rest_emot = pd.read_csv("data/reviews_emotions_rest.csv")
# hotel_emot = pd.read_csv("data/reviews_emotions_hotel.csv")
# all_reviews = pd.read_excel("data/reviews.xlsx")
# emotions = list()
# count = 0
# for row in df.itertuples(index=False):
#     username = row[0]
#     item = row[1]
#
#     rest_review = rest_emot.loc[(rest_emot["username"] == username) & (rest_emot["item"] == item)]
#     hotel_review = hotel_emot.loc[(hotel_emot["username"] == username) & (hotel_emot["item"] == item)]
#     if len(rest_review) > 0:
#         emotions.append(rest_review.iloc[0]["emotions"])
#     elif len(hotel_review) > 0:
#         emotions.append(hotel_review.iloc[0]["emotions"])
#     else:
#         rev = all_reviews.loc[(all_reviews["username"] == username) & (all_reviews["taObject"] == item)]
#         if len(rev) == 0:
#             count += 1
#             print(item)
#             emotions.append(None)
#         else:
#             emotions.append(te.get_emotion(str(rev.iloc[0]["text"])))
# df["emotions"] = emotions
# df.to_csv("data/reviews_temp.csv", index=False)
# print(count)

df = pd.read_csv("data/reviews_temp.csv")
df = df.loc[pd.isna(df["emotions"])]
print(df["username"])
print(len(df))

# stri = """"""
# print(te.get_emotion(stri))





















