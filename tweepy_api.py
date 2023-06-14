import tweepy
from utils.const import consumer_key, consumer_key_secret, access_token, access_token_secret, bearer_token

auth = tweepy.OAuth2BearerHandler(bearer_token)
api = tweepy.API(auth)

public_tweets = api.home_timeline()
for tweet in public_tweets:
    print(tweet.text)
