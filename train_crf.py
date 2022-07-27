import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split

import sklearn_crfsuite
from sklearn_crfsuite import metrics

from lib.utils import *
from joblib import dump

import spacy
import os
nlp = spacy.load("fr_core_news_md")

df_annotation = load_annot_label("data/project-4-at-2022-04-25-08-38-855b93ee.csv")
df_annotation = df_annotation[df_annotation.opinion_type == "Proposition"]
df_annotation = df_annotation[df_annotation.annotator_id == 1]
df_annotation = df_annotation.drop_duplicates(subset="id annotator_id x y".split())

if os.path.exists("annotation.csv"):
    df_annotation = pd.read_csv("./annotation.csv",sep="\t")
    df_annotation = df_annotation[df_annotation.keep == 1]

lemmas = []
pos = []
texts = []
start_pos = []
for text in nlp.pipe(df_annotation.all_text.values):
    lemmas.append([token.lemma_ for token in text])
    pos.append([token.pos_ for token in text])
    texts.append([token.text for token in text])
    start_pos.append([token.idx for token in text])
df_annotation["lemmas"] = lemmas
df_annotation["pos"] = pos
df_annotation["all_text"] = texts
df_annotation["start_pos"] = start_pos


lemmas = dict(df_annotation.drop_duplicates(subset="id")["id lemmas".split()].values)
pos = dict(df_annotation.drop_duplicates(subset="id")["id pos".split()].values)
texts = dict(df_annotation.drop_duplicates(subset="id")["id all_text".split()].values)
start_pos = dict(df_annotation.drop_duplicates(subset="id")["id start_pos".split()].values)

data = {}
for ix,row in df_annotation.iterrows():
    if not row.id in data:
        data[row.id]= {"text":texts[row.id],
                       "annot":['O' for i in range(len(lemmas[row.id]))],
                       "pos":pos[row.id],
                       "lemma":lemmas[row.id]}
    for ix,j in enumerate(start_pos[row.id]):
        if j < row.y and j > row.x:
            data[row.id]["annot"][ix] = "I-"+row.opinion_type
        #if j+len(texts[row.id][ix]) == row.y:
            #data[row.id]["annot"][ix] = "end-"+row.opinion_type
        if j == row.x:
            data[row.id]["annot"][ix] = "B-"+row.opinion_type


data = list(data.values())

train_sents, test_sents = train_test_split(data)
X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]

X_test = [sent2features(s) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]

# define fixed parameters and parameters to search
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    max_iterations=1000,
    all_possible_transitions=True
)
params_space = {
    'c1': scipy.stats.expon(scale=0.5),
    'c2': scipy.stats.expon(scale=0.05),
}

labels = df_annotation.opinion_type.unique().tolist()
# use the same metric for evaluation
f1_scorer = make_scorer(metrics.flat_f1_score,
                        average='weighted', labels=labels)

# search
rs = RandomizedSearchCV(crf, params_space,
                        cv=3,
                        verbose=1,
                        n_jobs=-1,
                        n_iter=50,
                        scoring=f1_scorer)
rs.fit(X_train, y_train)
print('best params:', rs.best_params_)
print('best CV score:', rs.best_score_)
print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))
print(metrics.flat_classification_report(
    y_test, rs.best_estimator_.predict(X_test), digits=3
))


dump(rs.best_estimator_,"crf_trained.pkl")
