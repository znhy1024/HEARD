from sklearn.feature_extraction.text import TfidfVectorizer

# After downloading the Tweets content, all data could be arranged as this format:
# data = {
#     eid:label,info:{tid:{text,time}}
# }

# Please sort the posts in each instance by time to get timeline and then merge the posts following your strategy.
# In our paper, the default strategy is that merge every 10 posts for BEARD.


def get_timeline(info):
    tids = []
    timeline = []
    texts = []
    tid_time = {}
    for tid, item in info.items():
        nt = item['time']
        if nt not in tid_time:
            tid_time[nt] = [tid]
        else:
            tid_time[nt].append(tid)
            tid_time[nt] = sorted(tid_time[nt])
    time_s = sorted(tid_time.keys())

    for nt in time_s:
        ids = tid_time[nt]
        for tid in ids:
            tids.append(tid)
            timeline.append(info[tid]['time'])
            texts.append(info[tid]['text'])

    return tids, timeline, texts

# data = {
#     eid:label,info:{tid:{text,time}},timeline:[time]
# }


def timeline_convert_merge_post(data, interval=10):

    for eid, _ in data.items():
        timeLine = data[eid]['timeline']
        texts = data[eid]['texts']
        tids = get_sorted_tid_list(data_eid=data[eid])

        merge_index = list(range(len(timeLine)))[0::interval]
        merge_texts, merge_times, merge_tids = [], [], []
        for i, index in enumerate(merge_index):
            try:
                next_index = merge_index[i+1]
            except:
                next_index = index+len(timeLine)+2
            assert next_index != index
            merge_text = [x for x in texts[index:next_index]]
            merge_time = [x for x in timeLine[index:next_index]]
            merge_tid = [x for x in tids[index:next_index]]

            merge_texts.append(merge_text)
            merge_times.append(merge_time)
            merge_tids.append(merge_tid)

        data[eid]['merge_seqs'] = {'merge_times': merge_times,
                                   'merge_texts': merge_texts, 'merge_tids': merge_tids}
    return data

# Compute vecs
# data = {
#     eid:label,info:{tid:{text,time}},timeline:[time],merge_seqs:{merge_times,merge_texts}
# }


def compute_tfidf(data):

    vec_data = {}
    corpus = []
    vecs = {}
    for eid, info in data.items():
        merge_seqs = info['merge_seqs']
        merge_texts = merge_seqs['merge_texts']

        vecs[eid] = [0]*len(merge_texts)
        for ti, text in enumerate(merge_texts):
            f_text = ' '.join(text).lower()
            # raw text should be pre-processed before that, the unit could be either single text or merged texts
            corpus.append(f_text)
            vecs[eid][ti] = len(corpus)-1

    vectorizer = TfidfVectorizer(
        analyzer='word', stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(corpus)

    for _, (eid, _) in enumerate(data.items()):
        X_index = vecs[eid]
        f_vecs = []
        for index in X_index:
            f_vecs.append(X[index])
        tmp = [x.toarray().tolist()[0] for x in f_vecs]
        vec_data[eid] = tmp
        data[eid]['merge_seqs']['merge_vecs'] = tmp
        data[eid].pop('info')
        data[eid].pop('timeline')
    return data


def add_timeline(data):
    for eid, value in data.items():
        info = data[eid]['info']
        tids, timeline, texts = get_timeline(info=info)

        data[eid]['timeline'] = timeline
    return data


def get_sorted_tid_list(data_eid):  # {info: {tid0: {'time': ...}}}
    tid_time_obj = {tid: data_eid['info'][tid]['time']
                    for tid in data_eid['info']}
    sorted_tid_time_list = sorted(
        tid_time_obj.items(), key=lambda item: item[1])
    tid_list = [tid_time_tuple[0] for tid_time_tuple in sorted_tid_time_list]

    return tid_list
    pass
