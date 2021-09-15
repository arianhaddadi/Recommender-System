import json
import math
import time
import pandas as pd


class TripAdvisorAddress:
    PERSONALITY_SCORES = "data/Normalized_pers_scores.csv"
    PERSONALITY_DIFFERENCES_EUCLIDEAN = "data/personality_differences_euclidean.csv"
    DEVIATIONS = "data/deviations.csv"
    HELPFULNESS = "data/helpfulness.csv"
    TRUST_SCORE = "data/trust_scores.csv"
    ALL_REVIEWS = "data/reviews.csv"
    NEIGHBOURS_DICT = "data/neighbours_dict.json"
    EMOTIONS = "data/emotions.csv"
    TRAIN_SET = "data/reviews_train_set.csv"
    TRAIN_SET_DICT = "data/train_set_dict.json"
    TEST_SET = "data/reviews_test_set.csv"
    TEST_SET_DICT = "data/test_set_dict.json"
    ITEM_BIASES = "data/item_biases.json"
    USER_BIASES = "data/user_biases.json"


class AmazonAddress:
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

class RecommenderModel:
    def __init__(self, isAmazon):
        self.address = AmazonAddress if isAmazon else TripAdvisorAddress

    def getPersonalityTraits(self):
        return ["Extraversion", "Neuroticism", "Agreeableness", "Conscientiousness", "Opennes"]

    def calPersonalityDiff(self):
        personalities = pd.read_csv(self.address.PERSONALITY_SCORES)

        users = list(personalities["username"])
        personality_traits = self.getPersonalityTraits()
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
                    # user_emotions_mean = user_emotions[personality_traits].mean()
                    destination_emotions = personalities.loc[personalities["username"] == destination].iloc[0]
                    # destination_emotions_mean = destination_emotions[personality_traits].mean()

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
        personality_matrix.to_csv(self.address.PERSONALITY_DIFFERENCES_EUCLIDEAN)


    def calHelpfulness(self, skewness, K):
        return K / (1 + math.e ** -(K * skewness))


    def calEmotionDiff(self, user_reviewed_hotels, destination_reviewed_hotels, common_hotels):
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


    def calFactors(self, reviews, helpfulness_k, command):
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
                        user_helpfulness = self.calHelpfulness(user_reviews["skewness"], helpfulness_k)
                        destination_helpfulness = self.calHelpfulness(destination_reviews["skewness"], helpfulness_k)

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
                            emotion_series.append(self.calEmotionDiff(user_reviewed_hotels, destination_reviewed_hotels, common_hotels))

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
            helpfulness_matrix.to_csv(self.address.HELPFULNESS)
        if command["D"]:
            deviation_matrix.to_csv(self.address.DEVIATIONS)
        if command["E"]:
            emotion_matrix.to_csv(self.address.EMOTIONS)


    def calDeviation(self, user_rating, average_rating, N):
        if user_rating == 0:
            return 0
        return (-1 / N) * abs(user_rating - average_rating)


    def generateReviewsDict(self, reviews, generate_new_one=True):
        if not generate_new_one:
            with open(self.address.TRAIN_SET_DICT) as file:
                return json.loads(file.readline())
        else:
            users_reviews = dict()
            average_rating = float(reviews["rating"].mean())
            users = reviews["username"].unique()

            for user in users:
                user_reviews = reviews.loc[reviews["username"] == user]
                reviewed_hotels = user_reviews["item"]

                sum_of_helpfulness = float(user_reviews["helpfulness"].sum())
                sum_of_sentiments = float(user_reviews["sentiment"].sum())
                sum_of_ratings = float(user_reviews["rating"].sum())

                reviews_dict = dict()
                for hotel in reviewed_hotels:
                    user_review = user_reviews.loc[user_reviews["item"] == hotel]
                    user_rating = float(user_review["rating"].iloc[0])
                    user_review_emotions_dict = user_review["emotions"].apply(eval).iloc[0]

                    reviews_dict[hotel] = dict()
                    reviews_dict[hotel]["deviation"] = self.calDeviation(user_rating, average_rating, N=4)
                    reviews_dict[hotel]["rating"] = user_rating
                    reviews_dict[hotel]["emotion"] = user_review_emotions_dict
                    reviews_dict[hotel]["helpfulness"] = float(user_review["helpfulness"].iloc[0])
                    reviews_dict[hotel]["sentiment"] = float(user_review["sentiment"].iloc[0])

                users_reviews[user] = dict()
                users_reviews[user]["reviews"] = reviews_dict
                users_reviews[user]["skewness"] = sum_of_helpfulness / len(reviewed_hotels)
                users_reviews[user]["ratings_average"] = sum_of_ratings / len(reviewed_hotels)
                users_reviews[user]["sentiment_average"] = sum_of_sentiments / len(reviewed_hotels)

            with open(self.address.TRAIN_SET_DICT, "w") as file:
                file.write(json.dumps(users_reviews))

            return users_reviews


    def calVicinity(self, reviews, user, candidate):
        user_reviews = reviews[user]["reviews"]
        candidate_reviews = reviews[candidate]["reviews"]

        num_of_common_reviews = 0
        for review in user_reviews:
            if review in candidate_reviews:
                num_of_common_reviews += 1
        return num_of_common_reviews


    def getNeighbours(self, reviews, K, calculate=True):
        if not calculate:
            with open(self.address.NEIGHBOURS_DICT) as file:
                return json.loads(file.readline())
        else:
            users_neighbours = dict()
            users = reviews.keys()
            for user in users:
                neighbours_dict = dict()
                for candidate in users:
                    if candidate != user:
                        num_of_common_reviews = self.calVicinity(reviews, user, candidate)
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

            with open(self.address.NEIGHBOURS_DICT, "w") as file:
                file.write(json.dumps(users_neighbours))

            return users_neighbours


    def calTrustScore(self, reviews, command):
        if command["H"]:
            helpfulness_values = pd.read_csv(self.address.HELPFULNESS)
            helpfulness_values.columns.values[0] = "users"
            helpfulness_values = helpfulness_values.set_index("users")

        if command["D"]:
            deviation_values = pd.read_csv(self.address.DEVIATIONS)
            deviation_values.columns.values[0] = "users"
            deviation_values = deviation_values.set_index("users")

        if command["E"]:
            emotion_values = pd.read_csv(self.address.EMOTIONS)
            emotion_values.columns.values[0] = "users"
            emotion_values = emotion_values.set_index("users")

        if command["P"]:
            personality_diffs = pd.read_csv(self.address.PERSONALITY_DIFFERENCES_EUCLIDEAN)
            personality_diffs.columns.values[0] = "users"
            personality_diffs = personality_diffs.set_index("users")

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
        trust_scores_matrix.to_csv(self.address.TRUST_SCORE)


    def predict(self, test_set_dict, reviews, neighbours, biases, *,
                threshold=4, predictions_dict=None, biases_cache=None):
        mae, rmse = 0, 0
        relevant, recommended, relevant_recommended = 0, 0, 0

        trust_scores = pd.read_csv(self.address.TRUST_SCORE)
        trust_scores.columns.values[0] = "username"
        trust_scores = trust_scores.set_index("username")

        user_biases = biases[0]
        user_biases_cache = biases_cache[0] if (biases_cache is not None) else None
        item_biases = biases[1]
        item_biases_cache = biases_cache[1] if (biases_cache is not None) else None
        users = test_set_dict.keys()
        num_of_predicted_items = 0
        for user in users:
            user_neighbours = neighbours[user]

            items_to_predict = test_set_dict[user]

            for item in items_to_predict:
                num_of_predicted_items += 1
                numerator = 0
                denominator = 0
                neighbours_that_reviewed_item = set()

                for neighbour in user_neighbours:
                    if item in reviews[neighbour]["reviews"]:
                        neighbours_that_reviewed_item.add(neighbour)
                        neighbour_review = reviews[neighbour]["reviews"][item]
                        trust_score = trust_scores[user][neighbour]
                        bmi = user_biases[neighbour] + reviews[neighbour]["ratings_average"] + item_biases[item]

                        numerator += trust_score * (neighbour_review["rating"] - bmi)
                        denominator += abs(trust_score)

                bui = user_biases[user] + reviews[user]["ratings_average"] + item_biases[item]
                prediction = bui + ((numerator / denominator) if denominator != 0 else 0)
                real = items_to_predict[item]["rating"]

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

                relevant += real >= threshold
                recommended += prediction >= threshold
                relevant_recommended += real >= threshold and prediction >= threshold

                mae += abs(prediction - real)
                rmse += ((prediction - real) ** 2)

        return {
            "mae": mae / num_of_predicted_items,
            "rmse": math.sqrt(rmse / num_of_predicted_items),
            "precision": relevant_recommended / recommended,
            "recall": relevant_recommended / relevant,
        }

    def generateTestSetDict(self, test_set, generate_new_one=True):
        if not generate_new_one:
            with open(self.address.TEST_SET_DICT) as file:
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
                    helpfulness = row[3]
                    sentiment = row[4]
                    emotion_dict = eval(row[5])

                    test_set_dict[user][item] = dict()
                    test_set_dict[user][item]["rating"] = rating
                    test_set_dict[user][item]["sentiment"] = sentiment
                    test_set_dict[user][item]["helpfulness"] = helpfulness
                    test_set_dict[user][item]["emotions"] = emotion_dict


            with open(self.address.TEST_SET_DICT, "w") as file:
                file.write(json.dumps(test_set_dict))

            return test_set_dict

    def updateBiases(self, biases, biases_cache, predictions_dict, test_set_dict, lambda_value, learning_rate):
        user_biases, item_biases = biases
        user_biases_cache, item_biases_cache = biases_cache

        for user in test_set_dict:
            derivative_as_neighbour = 0
            if user in user_biases_cache:
                user_caches = user_biases_cache[user]
                for cache in user_caches:
                    derivative_as_neighbour += 2 * (cache["prediction"] - cache["real_rating"]) * (-cache["trust_score"] / cache["denominator"])

            derivative_as_rater = 0
            user_predicted_items = test_set_dict[user]
            for item in user_predicted_items:
                derivative_as_rater += 2 * (predictions_dict[user][item] - user_predicted_items[item]["rating"])

            derivative = derivative_as_neighbour + derivative_as_rater + 2 * lambda_value * user_biases[user]
            user_biases[user] -= learning_rate * derivative / 100

        for item in item_biases:
            derivative_as_neighbour_item = 0
            if item in item_biases_cache:
                item_caches = item_biases_cache[item]
                for cache in item_caches:
                    derivative_as_neighbour_item += 2 * (cache["prediction"] - cache["real_rating"]) * (-cache["trust_score"] / cache["denominator"])

            derivative_as_user_item = 0
            for user in test_set_dict:
                if item in test_set_dict[user]:
                    derivative_as_user_item += 2 * (predictions_dict[user][item] - test_set_dict[user][item]["rating"])

            derivative = derivative_as_neighbour_item + derivative_as_user_item + 2 * lambda_value * item_biases[item]
            item_biases[item] -= learning_rate * derivative / 100


    def trainBiases(self, test_set_dict, reviews, neighbours, biases, num_of_epochs, lambda_value, learning_rate):
        predict_func_args = [test_set_dict, reviews, neighbours, biases]
        predictions_dict = dict()
        biases_cache = [dict(), dict()]
        self.predict(*predict_func_args, predictions_dict=predictions_dict, biases_cache=biases_cache)

        for i in range(num_of_epochs):
            print(f'epoch {i+1}')
            self.updateBiases(biases, biases_cache, predictions_dict, test_set_dict, lambda_value, learning_rate)
            biases_cache = [dict(), dict()]
            predictions_dict = dict()
            self.predict(*predict_func_args, predictions_dict=predictions_dict, biases_cache=biases_cache)


    def getBiases(self, test_set_dict, generate_new_ones):
        biases = list()
        if not generate_new_ones:
            with open(self.address.USER_BIASES) as file:
                biases.append(json.loads(file.readline()))
            with open(self.address.ITEM_BIASES) as file:
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

            with open(self.address.USER_BIASES, "w") as file:
                file.write(json.dumps(user_biases))
            with open(self.address.ITEM_BIASES, "w") as file:
                file.write(json.dumps(item_biases))

        return biases

    def prepareSummary(self, predictions_dict, neighbours, test_set_dict, train_set_dict):
        trust_scores = pd.read_csv(self.address.TRUST_SCORE)
        trust_scores.columns.values[0] = "username"
        trust_scores = trust_scores.set_index("username")

        personality_scores = pd.read_csv(self.address.PERSONALITY_SCORES)
        personality_scores.columns.values[0] = "users"
        personality_scores = personality_scores.set_index("users")

        columns =   ['username', 'item', 'rating', 'helpfulness', 'emotions', 'sentiment',
                     'neighbour_username', 'neighbour_rating', 'neighbour_helpfulness',
                     'neighbour_emotions', 'neighbour_sentiment', 'neighbour_text',
                     'predicted_rating', 'baseline_predict', 'Extraversion', 'Neuroticism',
                     'Agreeableness', 'Conscientiousness', 'Opennes', 'trust_score_baseline',
                     'trust_score_enhanced', 'neighbour_Extraversion', 'neighbour_Neuroticism',
                     'neighbour_Agreeableness', 'neighbour_Conscientiousness',
                     'neighbour_Opennes']
        summary = pd.DataFrame(columns=columns)

        for user in test_set_dict:
            user_neighbours = neighbours[user]
            for item in test_set_dict[user]:
                for neighbour in user_neighbours:
                    if item in train_set_dict[neighbour]["reviews"]:
                        user_personality = personality_scores.loc[personality_scores["username"] == user].iloc[0]
                        neighbour_personality = personality_scores.loc[personality_scores["username"] == neighbour].iloc[0]
                        # user_review = train_set_dict[user][item]
                        neighbour_review = train_set_dict[neighbour]["reviews"][item]
                        summary = summary.append({
                            'username': user,
                            'item': item,
                            'rating': test_set_dict[user][item]["rating"],
                            'helpfulness': test_set_dict[user][item]["helpfulness"],
                            'emotions': test_set_dict[user][item]["emotions"],
                            'sentiment':  test_set_dict[user][item]["sentiment"],
                            'neighbour_username': neighbour,
                            'neighbour_rating': neighbour_review["rating"],
                            'neighbour_helpfulness': neighbour_review["helpfulness"],
                            'neighbour_emotions': neighbour_review["emotion"],
                            'neighbour_sentiment': neighbour_review["sentiment"],
                            # 'neighbour_text':,
                            'predicted_rating': predictions_dict[user][item],
                            # 'baseline_predict': predictions_dict[user][item],
                            'Extraversion': user_personality["Extraversion"],
                            'Neuroticism': user_personality["Neuroticism"],
                            'Agreeableness': user_personality["Agreeableness"],
                            'Conscientiousness': user_personality["Conscientiousness"],
                            'Opennes': user_personality["Opennes"],
                            # 'trust_score_baseline': trust_scores[user][neighbour],
                            'trust_score_enhanced': trust_scores[user][neighbour],
                            'neighbour_Extraversion': neighbour_personality["Extraversion"],
                            'neighbour_Neuroticism': neighbour_personality["Neuroticism"],
                            'neighbour_Agreeableness': neighbour_personality["Agreeableness"],
                            'neighbour_Conscientiousness': neighbour_personality["Conscientiousness"],
                            'neighbour_Opennes': neighbour_personality["Opennes"]
                        }, ignore_index=True)

        summary.to_csv("data/report.csv", index=False)



    def calFeaturesAndPredict(self, train_set, test_set, command):
        reviews = self.generateReviewsDict(train_set, generate_new_one=True)

        neighbours = self.getNeighbours(reviews, K=20, calculate=True)

        self.calFactors(reviews, helpfulness_k=1, command=command)

        self.calTrustScore(reviews, command=command)

        test_set_dict = self.generateTestSetDict(test_set, generate_new_one=True)

        biases = self.getBiases(test_set_dict, generate_new_ones=True)

        self.trainBiases(test_set_dict, reviews, neighbours, biases, num_of_epochs=150, lambda_value=0.2, learning_rate=0.1)

        predictions_dict = dict()
        predictions_res = self.predict(test_set_dict, reviews, neighbours, biases, predictions_dict=predictions_dict)

        # prepareSummary(predictions_dict, neighbours, test_set_dict, reviews)

        return predictions_res


    def splitTrainTest(self, all_reviews, numOfReviewsPerUserTestSet):
        users = all_reviews["username"].unique()
        samples = set()
        for user in users:
            user_reviews = all_reviews.loc[all_reviews["username"] == user]
            if len(user_reviews) == numOfReviewsPerUserTestSet:
                continue
            choices = list(user_reviews.index.values)[:numOfReviewsPerUserTestSet]
            # choices = random.sample(list(user_reviews.index.values), k=numOfReviewsPerUserTestSet)
            assert len(choices) == numOfReviewsPerUserTestSet
            for choice in choices:
                samples.add(choice)

        test_set = all_reviews.loc[samples]
        train_set = all_reviews.loc[set(all_reviews.index.values) - samples]

        train_set.to_csv(self.address.TRAIN_SET, index=False)
        test_set.to_csv(self.address.TEST_SET, index=False)



    def getCommands(self):
        commands = [{
            "D": True,
            "H": False,
            "S": True,
            "E": False,
            "P": True,
            "Coeff": False
        }]
        return commands


    def start(self):
        all_reviews = pd.read_csv(self.address.ALL_REVIEWS)

        commands = self.getCommands()
        for command in commands:

            # splitTrainTest(all_reviews, numOfReviewsPerUserTestSet=1)

            train_set = pd.read_csv(self.address.TRAIN_SET)
            test_set = pd.read_csv(self.address.TEST_SET)

            print(f"Data Check: Train Set Length = {len(train_set)}, Test Set Length = {len(test_set)}, Total Length = {len(all_reviews)}")

            accuracyParams = self.calFeaturesAndPredict(train_set, test_set, command)

            self.printResults(command=command, accuracyParams=accuracyParams)


    def printResults(self, command, accuracyParams):
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

    def run(self):
        # self.personality()
        self.run()

if __name__ == '__main__':
    start_time = time.time()

    model = RecommenderModel()
    model.run()

    print(f"Elapsed Time = {round((time.time() - start_time) / 3600, 2)} hours")


