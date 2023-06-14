import os
import json
from tqdm import tqdm

PATH = '/home/jinmyeong/code/HEARD'
os.chdir(PATH)


def erase_dot(file_list):
    return [file for file in file_list if file[0] != '.']


def get_tweet(eid, path, is_source):
    tweet_type = 'source-tweet' if is_source == True else 'reactions'
    tweet_path = os.path.join(path, eid, tweet_type)

    tweet_name_list = erase_dot(os.listdir(tweet_path))
    tweet_obj = {}

    for tweet_name in tweet_name_list:
        tweet_name_path = os.path.join(tweet_path, tweet_name)
        with open(tweet_name_path, 'r') as f:
            tweet_json = json.load(f)

        tweet_id, tweet_text, tweet_time = tweet_json['id'], tweet_json['text'], change_time_form(
            tweet_json["created_at"])
        tweet_obj[tweet_id] = {'text': tweet_text, 'time': tweet_time}

    return tweet_obj


def preprocess_PHEME(PHEME_path='pheme-rnr-dataset'):
    data = {}
    idx = 0

    file_list = erase_dot(os.listdir(PHEME_path))

    for event in tqdm(file_list):
        rumor_instances_path = os.path.join(
            PHEME_path, f"{event}", 'rumours')
        non_rumor_instances_path = os.path.join(
            PHEME_path, f"{event}", 'non-rumours')

        # rumor_instances_list = [3241233, 4312134, 31244, ...]
        rumor_instances_list = erase_dot(os.listdir(rumor_instances_path))
        non_rumor_instances_list = erase_dot(
            os.listdir(non_rumor_instances_path))

        for eid in tqdm(rumor_instances_list):
            # rumor_source = {id0}
            rumor_source = get_tweet(
                eid, path=rumor_instances_path, is_source=True)
            # rumor_reaction = {id1, id2}
            rumor_reaction = get_tweet(
                eid, path=rumor_instances_path,  is_source=False)

            rumor_source.update(rumor_reaction)

            info = rumor_source
            label = 1

            data[f"eid{idx}"] = {'label': label, 'info': info}
            idx += 1

        for eid in tqdm(non_rumor_instances_list):
            non_rumor_source = get_tweet(
                eid, path=non_rumor_instances_path, is_source=True)  # non_rumor_source = {id0}
            # non_rumor_reaction = {id1, id2}
            non_rumor_reaction = get_tweet(
                eid, path=non_rumor_instances_path,  is_source=False)

            non_rumor_source.update(non_rumor_reaction)

            info = non_rumor_source
            label = 0

            data[f"eid{idx}"] = {'label': label, 'info': info}
            idx += 1

    return data


def change_time_form(created_at):
    _, month, day, time, _, year = created_at.split(' ')
    month_to_num_dict = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05',
                         'Jun': '06', 'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
    time_form = f"{month_to_num_dict[month]}-{day}-{year[2:]} {time}"

    return time_form
