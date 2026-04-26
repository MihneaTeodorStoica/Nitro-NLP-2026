# Eye-Tracking-Based Reading Time Prediction (⌐ ͡■ ͜ʖ ͡■)

Have you ever wondered how our eyes move when we read (⌐■_■)? Cognitive scientists use eye-tracking to understand language processing in the human brain. The time a reader spends fixating on a word provides deep insights into the word's complexity, its predictability, and its role within the sentence context. In this hackathon, your challenge is to step into the intersection of Natural Language Processing and Cognitive Science by building a machine learning model that predicts how long Romanian readers will spend on each word in a given text.

## Dataset Description

The dataset consists of reading sessions from multiple participants reading a variety of Romanian texts, ranging from literature to encyclopedic and popular science articles. We provide you with individual word-level Total Reading Time (TRT) measurements. The TRT is the sum of all fixations on a word, measured in milliseconds. Some words are skipped.

- *train.csv*: Contains the reading data with specific word features and the TRT for different participants. You will find the word itself, the text document it belongs to, a contextual word_id (encoding the text, page, and word index), the participant_id, and the answer representing the individual TRT in milliseconds.

Your task is to predict the reading times for additional unseen participants. Reading times are strongly correlated to word length and frequency. Consider starting with simple regression methods and looking at single-value variables such as [surprisal](https://aclanthology.org/2024.emnlp-main.921.pdf) or the probability of words in certain contexts.
One wonders whether all types of texts matter equally in predicting the average of other participants or whether all participants are equally important... Help us solve this puzzle!

## Output Format

The output file (.csv) must contain 3 columns:

| subtaskID | datapointID | answer |
| --------- | ----------- | ------ |
| 1         | 0           | 300    |
| 1         | 1           | 0      |
| 1         | 2           | 151    |
| 1         | 3           | 110    |
| 1         | 4           | 590    |
| 1         | 5           | 10     |

You must only send one .csv file for the subtask (see sample_submission.csv for the exact format).

## Evaluation Metric

We use a custom prediction metric consisting in:
    - `(max(0, R^2) + abs(pearson))/2`
    - [R^2 is the standard regression metric](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score), here we do not allow negative values
    - [pearson](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html) - is the Pearson correlation
    - The baseline is 36/100, any score under this will show up as 0 points on the leaderboard

A good solution would:
    - have values that are close to the actual reading times
    - have values that are correlated well to the reading times
    - one can obtain a good correlation with values that are outside of the range or inversely proportional, e.g. with word frequency measures
    - a constant predictor will get 0 and pearson will be NaN

```python
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

def eval_metric(y_true, preds):
    y_true = y_true.astype(float)
    preds = preds.astype(float)
    r2 = r2_score(y_true, preds, sample_weight=None, force_finite=True)
    r2 = max(0, r2)
    pears = pearsonr(y_true, preds)[0]
    if np.isnan(pears):
        pears = 0.0
    pears = np.abs(pears)
    return 100*(pears + r2)/2
```