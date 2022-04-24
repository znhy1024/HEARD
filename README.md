# Early Rumor Detection Using Neural Hawkes Process with a New Benchmark Dataset (Accepted at NAACL 2022) 

This is the official repository of EArly Rumor Detection (EARD) benchmark BEARD and model HEARD. 

- BEARD is the first EARD-oriented dataset, collected by including early-stage information relevant to the concerned claims from a fact-checking website [Snopes](https://www.snopes.com/fact-check/). BEARD contains 1,198 rumors and non-rumors reported during 2015/03-2021/01 with around 3.3 million relevant posts. 
- HEARD is a novel EARD model based on the Neural Hawkes Process to automatically determine an optimal time point for stable early detection.

<p align="center">
<img src="misc/HEARD.jpg" height=350>
</p>


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train HEARD with default hyperparameters, run this command:

```
python -u Main.py
```
After training, it will automatically output the results in terms of accuracy, recall, precision, recall, early rate and SEA.
You may also set the hyperparameters and data dir in in `config.json` to train your custom dataset and hyperparameters.

## Evaluation
By setting *evaluate_only* in `config.json`to true, you can use the same command ```python -u Main.py``` to test the trained model without training.

## Datasets
| Dataset  |   Class  |  #instance   | #post | #AvgLen (hrs)|
| :-------- | :-------- | :-------- | :-------------| :-------------|
| TWITTER  | rumor | 498 | 182,499 | 2,538    |
|         | non-rumor| 494 | 48,064 | 1,456  |
| PHEME| rumor      | 1,972 | 31,230 | 10 |
| | non-rumor | 3,830 | 71,210 | 19 |
| BEARD| rumor      | 531 | 2,644,807 | 1,432 |
| | non-rumor | 667 | 657,925 | 1,683 |

## Results

HEARD achieves the following performance on general classification metrics(accuracy and F1 score) and EARD-specific metrics(early rate and SEA):

| **Dataset** | **Acc**            | **F1**             | **ER**             | **SEA**            |
|:-----------:|:---------:|:------------------:|:------------------:|:------------------:|
|     TWITTER        | 0\.716 | 0\.714 | 0\.348             | 0\.789 |
|     PHEME        | 0\.823             | 0\.805             | 0\.284             | 0\.841 |
|      BEARD       | 0\.789 | 0\.788 | 0\.490             | 0\.765 |

 Results of Early Detection Accuracy Over Time:

<p align="left">
 TWITTER
<br>
  <img alt="img-name" src="misc/TWITTER_EDAOT.png" width="400">
</p>

<p align="left">
 PHEME
<br>
  <img alt="img-name" src="misc/PHEME_EDAOT.png" width="400">
</p>
<p align="left">
  BEARD
<br>
  <img alt="img-name" src="misc/BEARD_EDAOT.png" width="400">
</p>

## BEARD Dataset
This section aims to help users obtain BEARD dataset and prepare data. 
### Obtain BEARD dataset
Relevant files can be found [here](https://drive.google.com/drive/folders/1_l4_E-yJgFx7nAhj_S8VSHTEATfri4BL?usp=sharing).
#### BEARD_info.json
This file provides the claim information and search query for [Twitter search](https://twitter.com/search-advanced?lang=en). Each key-value pair of an instance contains: instance id, claim content, query and claim publish time. The data format is:
 ```
 {
    "instance id": {"claim": claim, "seed query":[sub_query,sub_query...],"time":time }
 }
 ```
#### BEARD.json
This file contains tweet ids for 1198 instances. Note that we cannot release the specific content of tweets due to the terms of use of Twitter data. Users can download the content via [Twitter API](https://developer.twitter.com/en/docs/twitter-api).
The data format is:
 ```
 {
    "instance id": {"eid": instance id, "post_ids":[id,id...] }
 }
 ```
### Prepare Input data
After obtaining the tweets content from Twitter, user needs to prepare data for model. User can follow [BERTweet](https://aclanthology.org/2020.emnlp-demos.2/)[[code](https://github.com/VinAIResearch/BERTweet)] to pre-process text and this [example](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) to generate tf-idf vectors. The data format should be:
 ```
  {
  "eid": {
          "label": "1", # 1 for rumor, 0 for non-rumor
          "merge_seqs": { # post are merged to shorten sequence
              "merge_times": [[timestamp,timestamp,...], [timestamp,timestamp,...], ...], # post publish timestamp
              'merge_vecs': [[...], [...], ...], # tf-idf vecs[1000] for each interval, so the shape of merge_vecs should be [num of intervals,1000] 
              }}
  ...
  }
  ```

## Citation

If you use this code in your research, please cite our [paper]().

```
@inproceedings{

  year={2022}
}
```

## Contact for Issues
- Fengzhu ZENG, fzzeng.2020@phdcs.smu.edu.sg

## References & Opensources

- **BERT** : [[BERT]](https://aclanthology.org/N19-1423.pdf)[[code]](https://github.com/huggingface/transformers), [[earliest rumor detection task]](https://link.springer.com/chapter/10.1007/978-3-030-88480-2_45#:~:text=Inspired%20by%20rumor%20psychology%2C%20we,features%20compared%20with%20non%2Drumor.)
- **CED** : [[paper]](https://arxiv.org/pdf/1811.04175.pdf), [[code]](https://github.com/thunlp/CED)
- **ERD** : [[paper]](https://aclanthology.org/N19-1163/), [[code]](https://github.com/DeepBrainAI/ERD)
- **STN** : [[paper]](https://aclanthology.org/2020.emnlp-main.727/)
- **TWITTER** : [[paper]](https://www.ijcai.org/Proceedings/16/Papers/537.pdf)
- **PHEME** : [[paper]](https://arxiv.org/pdf/1610.07363.pdf)
