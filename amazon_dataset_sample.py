import json
import math
import random
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error


class Address:
    PERSONALITY_SCORES = "data/Amazon_Normalized_pers_scores.csv"
    PERSONALITY_DIFFERENCES_EUCLIDEAN = "data/amazon_personality_differences_euclidean.csv"
    PERSONALITY_DIFFERENCES_PEARSON = "data/amazon_personality_differences_pearson.csv"
    DEVIATIONS = "data/amazon_deviations.csv"
    HELPFULNESS = "data/amazon_helpfulness.csv"
    TRUST_SCORE = "data/amazon_trust_scores.csv"
    ALL_REVIEWS = "data/amazon_reviews.csv"
    NEIGHBOURS_DICT = "data/amazon_neighbours_dict.json"
    EMOTIONS = "data/amazon_emotions.csv"
    TRAIN_SET = "data/amazon_reviews_train_set.csv"
    TRAIN_SET_DICT = "data/amazon_train_set_dict.json"
    TEST_SET = "data/amazon_reviews_test_set.csv"
    TEST_SET_DICT = "data/amazon_test_set_dict.json"
    ITEM_BIASES = "data/amazon_item_biases.json"
    USER_BIASES = "data/amazon_user_biases.json"
    XGB_DATASET_PERSONALITY = "data/xgb_dataset_personality.csv"
    XGB_DATASET_EMOTION = "data/xgb_dataset_emotion.csv"
    XGB_RESULTS = "data/xgb_results.csv"

def getPersonalityTraits():
    return ["Extraversion", "Neuroticism", "Agreeableness", "Conscientiousness", "Opennes", "Extraversion", "Opennes"]


def personality():
    personalities = pd.read_csv(Address.PERSONALITY_SCORES)

    users = list(personalities["username"])
    personality_traits = getPersonalityTraits()
    personality_matrix = pd.DataFrame(columns=users)
    already_calculated_users = set()

    for user in users:
        personality_series = list()
        for destination in users:
            if user == destination:
                personality_series.append(1)
            elif destination in already_calculated_users:
                personality_series.append(personality_matrix.loc[destination][user])
            else:
                user_emotions = personalities.loc[personalities["username"] == user].iloc[0]
                user_emotions_mean = user_emotions[personality_traits].mean()
                destination_emotions = personalities.loc[personalities["username"] == destination].iloc[0]
                destination_emotions_mean = destination_emotions[personality_traits].mean()

                # numerator_sum, denominator_sum_user, denominator_sum_destination = 0, 0, 0
                diff = 0
                for personality_trait in personality_traits:
                    # user_emotion, destination_emotion = user_emotions[personality_trait], destination_emotions[personality_trait]

                    # numerator_sum += (user_emotion - user_emotions_mean) * (destination_emotion - destination_emotions_mean)
                    # denominator_sum_user += (user_emotion - user_emotions_mean) ** 2
                    # denominator_sum_destination += (destination_emotion - destination_emotions_mean) ** 2

                    diff += math.pow(user_emotions[personality_trait] - destination_emotions[personality_trait], 2)

                # correlation = (numerator_sum / math.sqrt(denominator_sum_user * denominator_sum_destination) + 1) / 2
                # personality_series.append(correlation)
                personality_series.append(1 - math.sqrt(diff))

        already_calculated_users.add(user)
        series = pd.Series(data=personality_series, index=users, name=user)
        personality_matrix = personality_matrix.append(series)
    personality_matrix.to_csv(Address.PERSONALITY_DIFFERENCES_EUCLIDEAN)


def calHelpfulness(skewness, K):
    return K / (1 + math.e ** -(K * skewness))


def calEmotionDiff(user_reviewed_hotels, destination_reviewed_hotels, common_hotels):
    sum = 0
    emotions = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise']
    for hotel in common_hotels:
        user_review_emotions = user_reviewed_hotels[hotel]["emotion"]
        destination_review_emotions = destination_reviewed_hotels[hotel]["emotion"]

        diff = 0
        for emotion in emotions:
            diff += math.pow(user_review_emotions[emotion] - destination_review_emotions[emotion], 2)

        sum += math.sqrt(diff)
    return 1 - (sum / len(common_hotels))


def factors(reviews, helpfulness_k, command):
    users = reviews.keys()

    if command["H"]:
        helpfulness_matrix = pd.DataFrame(columns=users)
    if command["D"]:
        deviation_matrix = pd.DataFrame(columns=users)
    if command["E"]:
        emotion_matrix = pd.DataFrame(columns=users)

    already_calculated_users = set()

    for user in users:
        if command["H"]:
            helpfulness_series = list()
        if command["D"]:
            deviation_series = list()
        if command["E"]:
            emotion_series = list()

        for destination in users:
            if user == destination:
                if command["H"]:
                    helpfulness_series.append(1)
                if command["D"]:
                    deviation_series.append(1)
                if command["E"]:
                    emotion_series.append(1)

            elif destination in already_calculated_users:
                if command["H"]:
                    helpfulness_series.append(helpfulness_matrix.loc[destination][user])
                if command["D"]:
                    deviation_series.append(deviation_matrix.loc[destination][user])
                if command["E"]:
                    emotion_series.append(emotion_matrix.loc[destination][user])

            else:
                user_reviews = reviews[user]
                destination_reviews = reviews[destination]

                if command["H"]:
                    user_helpfulness = calHelpfulness(user_reviews["skewness"], helpfulness_k)
                    destination_helpfulness = calHelpfulness(destination_reviews["skewness"], helpfulness_k)

                user_reviewed_hotels = user_reviews["reviews"]
                destination_reviewed_hotels = destination_reviews["reviews"]

                num_of_common_reviews = 0
                sum_of_deviation = 0
                common_hotels = list()

                for hotel in user_reviewed_hotels:
                    if hotel in destination_reviewed_hotels:
                        common_hotels.append(hotel)
                        num_of_common_reviews += 1
                        sum_of_deviation += (1 - abs(user_reviewed_hotels[hotel]["deviation"] - destination_reviewed_hotels[hotel]["deviation"]))

                if num_of_common_reviews == 0:
                    if command["H"]:
                        helpfulness_series.append(0)
                    if command["D"]:
                        deviation_series.append(0)
                    if command["E"]:
                        emotion_series.append(0)

                else:
                    if command["H"]:
                        helpfulness_series.append((1 - abs(user_helpfulness - destination_helpfulness)) / num_of_common_reviews)
                    if command["D"]:
                        deviation_series.append(sum_of_deviation / num_of_common_reviews)
                    if command["E"]:
                        emotion_series.append(calEmotionDiff(user_reviewed_hotels, destination_reviewed_hotels, common_hotels))

        already_calculated_users.add(user)

        if command["H"]:
            helpfulness_series = pd.Series(data=helpfulness_series, index=users, name=user)
            helpfulness_matrix = helpfulness_matrix.append(helpfulness_series)
        if command["D"]:
            deviation_series = pd.Series(data=deviation_series, index=users, name=user)
            deviation_matrix = deviation_matrix.append(deviation_series)
        if command["E"]:
            emotion_series = pd.Series(data=emotion_series, index=users, name=user)
            emotion_matrix = emotion_matrix.append(emotion_series)

    if command["H"]:
        helpfulness_matrix.to_csv(Address.HELPFULNESS)
    if command["D"]:
        deviation_matrix.to_csv(Address.DEVIATIONS)
    if command["E"]:
        emotion_matrix.to_csv(Address.EMOTIONS)


def calDeviation(user_rating, average_rating, N):
    if user_rating == 0:
        return 0
    return (-1 / N) * abs(user_rating - average_rating)


def generateReviewsDict(reviews, generate_new_one=True):
    if not generate_new_one:
        with open(Address.TRAIN_SET_DICT) as file:
            return json.loads(file.readline())
    else:
        users_reviews = dict()
        average_rating = reviews["rating"].mean()
        users = reviews["username"].unique()

        current_time = time.time()
        for user in users:
            user_reviews = reviews.loc[reviews["username"] == user]
            reviewed_hotels = user_reviews["item"]

            sum_of_helpfulness = user_reviews["helpfulness"].sum()
            sum_of_ratings = user_reviews["rating"].sum()
            sum_of_sentiments = user_reviews["sentiment"].sum()

            reviews_dict = dict()
            for hotel in reviewed_hotels:
                user_review = user_reviews.loc[user_reviews["item"] == hotel]
                user_rating = user_review["rating"].iloc[0]
                review_time = user_review["reviewTime"].iloc[0]
                user_review_emotions_dict = user_review["review_emotions"].apply(eval).iloc[0]

                reviews_dict[hotel] = dict()
                reviews_dict[hotel]["deviation"] = calDeviation(user_rating, average_rating, N=4)
                reviews_dict[hotel]["rating"] = user_rating
                reviews_dict[hotel]["emotion"] = user_review_emotions_dict
                reviews_dict[hotel]["coefficient"] = (current_time - review_time) / current_time * 5

            users_reviews[user] = dict()
            users_reviews[user]["reviews"] = reviews_dict
            users_reviews[user]["skewness"] = sum_of_helpfulness / len(reviewed_hotels)
            users_reviews[user]["sentiment_average"] = sum_of_sentiments / len(reviewed_hotels)
            users_reviews[user]["ratings_average"] = sum_of_ratings / len(reviewed_hotels)


        with open(Address.TRAIN_SET_DICT, "w") as file:
            file.write(json.dumps(users_reviews))

        return users_reviews


def calVicinity(reviews, user, candidate):
    user_reviews = reviews[user]["reviews"]
    candidate_reviews = reviews[candidate]["reviews"]

    num_of_common_reviews = 0
    for review in user_reviews:
        if review in candidate_reviews:
            num_of_common_reviews += 1
    return num_of_common_reviews


def getNeighbours(reviews, K, calculate=True):
    if not calculate:
        with open(Address.NEIGHBOURS_DICT) as file:
            return json.loads(file.readline())
    else:
        users_neighbours = dict()
        users = reviews.keys()
        for user in users:
            neighbours_dict = dict()
            for candidate in users:
                if candidate != user:
                    num_of_common_reviews = calVicinity(reviews, user, candidate)
                    if num_of_common_reviews in neighbours_dict:
                        neighbours_dict[num_of_common_reviews].append(candidate)
                    else:
                        neighbours_dict[num_of_common_reviews] = [candidate]
            keys = list(neighbours_dict.keys())
            keys.sort(reverse=True)
            neighbours = list()
            for key in keys:
                if len(neighbours) + len(neighbours_dict[key]) < K:
                    neighbours.extend(neighbours_dict[key])
                else:
                    neighbours.extend(neighbours_dict[key][:K - len(neighbours)])
            users_neighbours[user] = neighbours

        with open(Address.NEIGHBOURS_DICT, "w") as file:
            file.write(json.dumps(users_neighbours))

        return users_neighbours


def trustScore(reviews, command):
    if command["H"]:
        helpfulness_values = pd.read_csv(Address.HELPFULNESS)
        helpfulness_values.columns.values[0] = "users"
        helpfulness_values = helpfulness_values.set_index("users")
        # users = list(helpfulness_values.index.values)

    if command["D"]:
        deviation_values = pd.read_csv(Address.DEVIATIONS)
        deviation_values.columns.values[0] = "users"
        deviation_values = deviation_values.set_index("users")
        # users = list(deviation_values.index.values)

    if command["E"]:
        emotion_values = pd.read_csv(Address.EMOTIONS)
        emotion_values.columns.values[0] = "users"
        emotion_values = emotion_values.set_index("users")
        # users = list(emotion_values.index.values)

    if command["P"]:
        personality_diffs = pd.read_csv(Address.PERSONALITY_DIFFERENCES_EUCLIDEAN)
        personality_diffs.columns.values[0] = "users"
        personality_diffs = personality_diffs.set_index("users")
        # users = list(personality_diffs.index.values)

    users = reviews.keys()
    trust_scores_matrix = pd.DataFrame(columns=users)
    already_calculated_users = set()

    for user in users:
        trust_scores_series = list()
        for destination in users:
            if user == destination:
                trust_scores_series.append(1)
            elif destination in already_calculated_users:
                trust_scores_series.append(trust_scores_matrix.loc[destination][user])
            else:
                values_to_average = list()
                if command["H"]:
                    values_to_average.append(helpfulness_values[destination][user])
                if command["D"]:
                    values_to_average.append(deviation_values[destination][user])
                if command["E"]:
                    values_to_average.append(emotion_values[destination][user])
                if command["P"]:
                    values_to_average.append(personality_diffs[destination][user])
                if command["S"]:
                    values_to_average.append(abs(reviews[user]["sentiment_average"] - reviews[destination]["sentiment_average"]))

                trust_scores_series.append(sum(values_to_average) / len(values_to_average))

        already_calculated_users.add(user)
        series = pd.Series(data=trust_scores_series, index=users, name=user)
        trust_scores_matrix = trust_scores_matrix.append(series)
    trust_scores_matrix.to_csv(Address.TRUST_SCORE)


def predict(test_set_dict, reviews, neighbours, biases, *,
            threshold=4, with_coeff=False, predictions_dict=None, biases_cache=None):
    mae, rmse = 0, 0
    relevant, recommended, relevant_recommended = 0, 0, 0

    trust_scores = pd.read_csv(Address.TRUST_SCORE)
    trust_scores.columns.values[0] = "username"
    trust_scores = trust_scores.set_index("username")

    user_biases = biases[0]
    user_biases_cache = biases_cache[0] if (biases_cache is not None) else None
    item_biases = biases[1]
    item_biases_cache = biases_cache[1] if (biases_cache is not None) else None
    users = test_set_dict.keys()
    for user in users:
        user_neighbours = neighbours[user]

        items_to_predict = test_set_dict[user]

        for item in items_to_predict:
            numerator = 0
            denominator = 0
            neighbours_that_reviewed_item = set()

            for neighbour in user_neighbours:
                if item in reviews[neighbour]["reviews"]:
                    neighbours_that_reviewed_item.add(neighbour)
                    neighbour_review = reviews[neighbour]["reviews"][item]
                    trust_score = trust_scores[user][neighbour]
                    bmi = user_biases[neighbour] + reviews[neighbour]["ratings_average"] + item_biases[item]
                    # bmi = reviews[neighbour]["ratings_average"]

                    numerator += trust_score * (neighbour_review["rating"] - bmi)
                    denominator += abs(trust_score)

            bui = user_biases[user] + reviews[user]["ratings_average"] + item_biases[item]
            prediction = bui + ((numerator / denominator) if denominator != 0 else 0)
            real = items_to_predict[item]

            if denominator != 0:
                if user_biases_cache is not None:
                    for neighbour in neighbours_that_reviewed_item:
                        if neighbour not in user_biases_cache:
                            user_biases_cache[neighbour] = list()
                        user_biases_cache[neighbour].append({
                            "denominator": denominator,
                            "trust_score": trust_scores[user][neighbour],
                            "prediction": prediction,
                            "real_rating": real
                        })

                if item_biases_cache is not None:
                    for neighbour in neighbours_that_reviewed_item:
                        if item not in item_biases_cache:
                            item_biases_cache[item] = list()
                        item_biases_cache[item].append({
                            "denominator": denominator,
                            "trust_score": trust_scores[user][neighbour],
                            "prediction": prediction,
                            "real_rating": real
                        })

            if predictions_dict is not None:
                if user not in predictions_dict:
                    predictions_dict[user] = dict()
                predictions_dict[user][item] = prediction

            # if prediction > 20:
            #     temp = item_biases[item]
            #     temp2 = user_biases[user]
            #     temp3 = reviews[user]["ratings_average"]
            #
            #     print(prediction, real)

            relevant += real >= threshold
            recommended += prediction >= threshold
            relevant_recommended += real >= threshold and prediction >= threshold

            mae += abs(prediction - real) / len(test_set_dict)
            rmse += ((prediction - real) ** 2) / len(test_set_dict)

    return {
        "mae": mae,
        "rmse": math.sqrt(rmse),
        "precision": relevant_recommended / recommended,
        "recall": relevant_recommended / relevant,
    }

def generateTestSetDict(test_set, generate_new_one=True):
    if not generate_new_one:
        with open(Address.TEST_SET_DICT) as file:
            return json.loads(file.readline())
    else:
        test_set_dict = dict()
        users = test_set["username"].unique()

        for user in users:
            test_set_dict[user] = dict()

            user_reviews = test_set.loc[test_set["username"] == user]

            for row in user_reviews.itertuples(index=False):
                item = row[1]
                rating = row[2]

                test_set_dict[user][item] = rating

        with open(Address.TEST_SET_DICT, "w") as file:
            file.write(json.dumps(test_set_dict))

        return test_set_dict

def updateBiases(biases, biases_cache, predictions_dict, test_set_dict, lambda_value, learning_rate):
    user_biases, item_biases = biases
    user_biases_cache, item_biases_cache = biases_cache

    for user in test_set_dict:
        derivative_as_neighbour = 0
        if user in user_biases_cache:
            user_caches = user_biases_cache[user]
            for cache in user_caches:
                # temp = 2 / 5 * (cache["prediction"] - cache["real_rating"]) * (-cache["trust_score"] / cache["denominator"])
                # if temp > 2:
                #     print(10)
                derivative_as_neighbour += 2 * (cache["prediction"] - cache["real_rating"]) * (-cache["trust_score"] / cache["denominator"])

        derivative_as_rater = 0
        user_predicted_items = test_set_dict[user]
        for item in user_predicted_items:
            derivative_as_rater += 2 * (predictions_dict[user][item] - user_predicted_items[item])

        derivative = derivative_as_neighbour + derivative_as_rater + 2 * lambda_value * user_biases[user]
        user_biases[user] -= learning_rate * derivative / 25

    for item in item_biases:
        derivative_as_neighbour_item = 0
        if item in item_biases_cache:
            item_caches = item_biases_cache[item]
            for cache in item_caches:
                derivative_as_neighbour_item += 2 * (cache["prediction"] - cache["real_rating"]) * (-cache["trust_score"] / cache["denominator"])

        derivative_as_user_item = 0
        for user in test_set_dict:
            if item in test_set_dict[user]:
                derivative_as_user_item += 2 * (predictions_dict[user][item] - test_set_dict[user][item])

        derivative = derivative_as_neighbour_item + derivative_as_user_item + 2 * lambda_value * item_biases[item]
        item_biases[item] -= learning_rate * derivative / 25


def train_biases(test_set_dict, reviews, neighbours, biases, num_of_epochs, lambda_value, learning_rate):
    predict_func_args = [test_set_dict, reviews, neighbours, biases]
    predictions_dict = dict()
    biases_cache = [dict(), dict()]
    predict(*predict_func_args, predictions_dict=predictions_dict, biases_cache=biases_cache)

    for i in range(num_of_epochs):
        print(f'epoch {i+1}')
        updateBiases(biases, biases_cache, predictions_dict, test_set_dict, lambda_value, learning_rate)
        biases_cache = [dict(), dict()]
        predictions_dict = dict()
        predict(*predict_func_args, predictions_dict=predictions_dict, biases_cache=biases_cache)


def getBiases(test_set_dict, generate_new_ones):
    biases = list()
    if not generate_new_ones:
        with open(Address.USER_BIASES) as file:
            biases.append(json.loads(file.readline()))
        with open(Address.ITEM_BIASES) as file:
            biases.append(json.loads(file.readline()))
    else:
        item_biases = dict()
        user_biases = dict()

        for user in test_set_dict:
            user_biases[user] = 0
            for item in test_set_dict[user]:
                item_biases[item] = 0


        biases.append(user_biases)
        biases.append(item_biases)

        with open(Address.USER_BIASES, "w") as file:
            file.write(json.dumps(user_biases))
        with open(Address.ITEM_BIASES, "w") as file:
            file.write(json.dumps(item_biases))

    return biases


def calFeaturesAndPredict(train_set, test_set, command):
    reviews = generateReviewsDict(train_set, generate_new_one=True)

    neighbours = getNeighbours(reviews, K=20, calculate=True)

    factors(reviews, helpfulness_k=1, command=command)

    trustScore(reviews, command=command)

    test_set_dict = generateTestSetDict(test_set, generate_new_one=True)

    # exit()

    biases = getBiases(test_set_dict, generate_new_ones=True)

    train_biases(test_set_dict, reviews, neighbours, biases, num_of_epochs=20, lambda_value=0.1, learning_rate=0.1)

    return predict(test_set_dict, reviews, neighbours, biases)


def splitTrainTest(all_reviews, numOfReviewsPerUserTestSet):
    users = all_reviews["username"].unique()
    samples = set()
    for user in users:
        user_reviews = all_reviews.loc[all_reviews["username"] == user].sort_values(["reviewTime"], ascending=False)
        choices = list(user_reviews.index.values)[:numOfReviewsPerUserTestSet]
        # choices = random.sample(list(user_reviews.index.values), k=numOfReviewsPerUserTestSet)
        assert len(choices) == numOfReviewsPerUserTestSet
        for choice in choices:
            samples.add(choice)

    test_set = all_reviews.loc[samples]
    train_set = all_reviews.loc[set(all_reviews.index.values) - samples]

    train_set.to_csv(Address.TRAIN_SET, index=False)
    test_set.to_csv(Address.TEST_SET, index=False)



def getCommands():
    commands = [{
        "D": True,
        "H": False,
        "S": True,
        "E": False,
        "P": True,
        "Coeff": False,
    # }, {
    #     "D": False,
    #     "H": False,
    #     "S": False,
    #     "E": True,
    #     "P": False,
    #     "Coeff": False,
    }]
    return commands


def run():
    # for i in range(5):
        # all_reviews = pd.read_csv(f"amazon_reviews_{i}.csv")
        # print(f'range {i}')
        all_reviews = pd.read_csv(Address.ALL_REVIEWS)

        commands = getCommands()
        for command in commands:

            splitTrainTest(all_reviews, numOfReviewsPerUserTestSet=1)

            train_set = pd.read_csv(Address.TRAIN_SET)
            test_set = pd.read_csv(Address.TEST_SET)

            print(f"Data Check: Train Set Length = {len(train_set)}, Test Set Length = {len(test_set)}, Total Length = {len(all_reviews)}")

            accuracyParams = calFeaturesAndPredict(train_set, test_set, command)

            print_result(command=command, accuracyParams=accuracyParams)


def print_result(command, accuracyParams):
    factors = list(command.keys())[:-1]
    factorsToFactorsMap = {
        "P": "Personality",
        "D": "Deviation",
        "H": "Helpfulness",
        "E": "Emotion",
        "S": "Sentiment",
    }
    result = ""
    for factor in factors:
        if command[factor]:
            result += factorsToFactorsMap[factor] + "-"

    result += f', RMSE = {accuracyParams["rmse"]}, MAE={accuracyParams["mae"]}, '
    result += f'Precision={accuracyParams["precision"]}, Recall={accuracyParams["recall"]}'

    print(result)

def prepareDatasetXGB():
    reviews = generateReviewsDict(pd.read_csv(Address.ALL_REVIEWS))
    personality_scores = pd.read_csv(Address.PERSONALITY_SCORES)
    df = pd.DataFrame(columns=["username", "deviation", "helpfulness",
                               "Extraversion", "Neuroticism", "Agreeableness",
                               "Conscientiousness", "Opennes", "rating"])

    for user in reviews:
        user_reviews = reviews[user]["reviews"]
        for hotel in user_reviews:
            review = user_reviews[hotel]
            user_personality = personality_scores.loc[personality_scores["username"] == user]

            df = df.append({
                "username": user,
                "deviation": review["deviation"],
                "helpfulness": review["helpfulness"],
                "Extraversion": user_personality["Extraversion"].iloc[0],
                "Neuroticism": user_personality["Neuroticism"].iloc[0],
                "Agreeableness": user_personality["Agreeableness"].iloc[0],
                "Conscientiousness": user_personality["Conscientiousness"].iloc[0],
                "Opennes": user_personality["Opennes"].iloc[0],
                "rating": review["rating"]
            }, ignore_index=True)
    df.to_csv(Address.XGB_DATASET_PERSONALITY)

def doXGBTraining(dataset, K):
    x, y = dataset.iloc[:, 1:-1], dataset.iloc[:, -1]
    rmse = 0

    kf = KFold(n_splits=K, shuffle=True)
    for train_index, test_index in kf.split(x):
        xg_reg = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=1,
                                  max_depth=10, reg_alpha=10, n_estimators=10)
        x_train, x_test, y_train, y_test = x.iloc[train_index], x.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
        xg_reg.fit(x_train, y_train)
        preds = xg_reg.predict(x_test)

        rmse += np.sqrt(mean_squared_error(y_test, preds.round()))
    print(f"rmse = {rmse / K}")
    # data_dmatrix = xgb.DMatrix(data=x, label=y)
    #
    # params = {"objective": "reg:squarederror", 'colsample_bytree': 0.3, 'learning_rate': 0.1,
    #           'max_depth': 10, 'alpha': 10}
    # cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=K,
    #                     num_boost_round=50, early_stopping_rounds=10, metrics="rmse", as_pandas=True, seed=123)
    # cv_results.to_csv(Address.XGB_RESULTS, index=None)

def doKFoldXGB(K):
    # prepareDatasetXGB()
    doXGBTraining(pd.read_csv(Address.XGB_DATASET_PERSONALITY), K)
    doXGBTraining(pd.read_csv(Address.XGB_DATASET_EMOTION), K)


if __name__ == '__main__':

    start_time = time.time()

    # personality()
    run()
    # doKFoldXGB(K=10)

    print(f"Elapsed Time = {round((time.time() - start_time) / 3600, 2)} hours")
