# This code creates the dataset from Corpus.csv which is downloadable from the
# internet well known dataset which is labeled manually by hand. But for the text
# of tweets you need to fetch them with their IDs.
import tweepy
import traceback
from utils.const import consumer_key, consumer_key_secret, access_token, access_token_secret, bearer_token
# Twitter Developer keys here
# It is CENSORED

auth = tweepy.OAuthHandler(
    consumer_key, consumer_key_secret, access_token, access_token_secret)
# auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
# client = tweepy.Client(bearer_token=bearer_token, consumer_key=consumer_key,
#                        consumer_secret=consumer_key_secret, access_token=access_token, access_token_secret=access_token_secret)
# This method creates the training set


def createTrainingSet(corpusFile, targetResultFile):
    import csv
    import time

    counter = 0
    corpus = [{"tweet_id": '1063600737832308736'}]

    # with open(corpusFile, 'r') as csvfile:
    #     lineReader = csv.reader(csvfile, delimiter=',', quotechar="\"")
    #     for row in lineReader:
    #         corpus.append(
    #             {"tweet_id": row[2], "label": row[1], "topic": row[0]})

    sleepTime = 2
    trainingDataSet = []

    for tweet in corpus:
        try:
            # client.get_list(['1063600737832308736'])
            print(api.verify_credentials().screen_name)
            tweetFetched = api.get_status(tweet["tweet_id"])
            print("Tweet fetched" + tweetFetched.text)
            tweet["text"] = tweetFetched.text
            trainingDataSet.append(tweet)
            time.sleep(sleepTime)

        except Exception as ex:
            print("Inside the exception - no:2")
            err_msg = traceback.format_exc()
            print(err_msg)
            continue

    with open(targetResultFile, 'w') as csvfile:
        linewriter = csv.writer(csvfile, delimiter=',', quotechar="\"")
        for tweet in trainingDataSet:
            try:
                linewriter.writerow(
                    [tweet["tweet_id"], tweet["text"], tweet["label"], tweet["topic"]])
            except Exception as e:
                print(e)
    return trainingDataSet


# Code starts here
# This is corpus dataset
corpusFile = "datasets/corpus.csv"
# This is my target file
targetResultFile = "datasets/targetResultFile.csv"
# Call the method
resultFile = createTrainingSet(corpusFile, targetResultFile)
