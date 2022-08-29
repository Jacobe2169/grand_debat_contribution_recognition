import pandas as pd



def load_annot_label(fn,csv_sep = ","):
    df = pd.read_csv(fn, sep=csv_sep).iloc[:, 1:].fillna("[]")
    df.label = df.label.apply(eval)
    data = df["id label annotator answer".split()].values
    new_data = {}
    for id_, labels, annotator, text in [tuple(x) for x in data]:
        if not id_ in new_data:
            new_data[id_] = {}
        if not annotator in new_data[id_]: new_data[id_][annotator] = []
        new_data[id_][annotator].extend(labels)
        new_data[id_]["text"] = text

    data = []
    for id_ in new_data:
        for annotator in new_data[id_]:
            if annotator == "text": continue
            for label in new_data[id_][annotator]:
                data.append([id_, annotator, label["start"], label["end"], label["labels"][0], label["text"],
                             new_data[id_]["text"]])
    return pd.DataFrame(data, columns="id annotator_id x y opinion_type text_selected all_text".split())


def word2features(sent, i):
    word = sent["text"][i]
    postag = sent["pos"][i]
    lemma = sent["lemma"][i]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
        'lemma': lemma,
        'lemma[:2]': lemma[:2],
    }
    if i > 0:
        word1 = sent["text"][i-1]
        postag1 = sent["pos"][i-1]
        lemma = sent["lemma"][i-1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
            '-1:lemma': lemma,
            '-1:lemma[:2]': lemma[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent["text"])-1:
        word1 = sent["text"][i+1]
        postag1 = sent["pos"][i+1]
        lemma = sent["lemma"][i+1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
            '+1:lemma': lemma,
            '+1:lemma[:2]': lemma[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent["text"]))]

def sent2labels(sent):
    return sent["annot"]

def sent2tokens(sent):
    return [token for token, postag, label in sent]