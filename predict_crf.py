import os

from joblib import load
import spacy
from tqdm import tqdm
from lib.utils import *
import argparse

def foo(x):
    try:
        int(x)
        return True
    except :
        return False

parser = argparse.ArgumentParser()
parser.add_argument("gdb_data_filename")
parser.add_argument("model_filename")
parser.add_argument("output_dir")

args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

# LOAD MODEL
crf = load(args.model_filename)

nlp = spacy.load("fr_core_news_md")

df_data = pd.read_csv(args.gdb_data_filename,dtype={"authorZipCode":str}).fillna("")
for col in df_data.columns[11:]:
    name_col = col.split(" - ")[0]
    print(name_col)
    with open(f"{args.output_dir}/{name_col}.txt",'w') as f:
        data=[]
        entry_line_number=0
        for text in tqdm(nlp.pipe(df_data[col],n_process=4),total=len(df_data)):
            props = []
            t, l, p = [], [], []
            for token in text:
                l.append(token.lemma_)
                p.append(token.pos_ )
                t.append(token.text)
            sent = {"text":t,"pos":p,"lemma":l}
            pred = crf.predict([sent2features(sent)])[0]
            curr_prop = ""
            curr_propx,curr_propy = 0,0
            for ix,p in enumerate(pred):
                if p.startswith("I"):
                    if curr_prop != "":
                        curr_propy = ix
                        text_cleansed = text[curr_propx:curr_propy].text.replace('\t','')
                        f.write(f"{entry_line_number}\t{text_cleansed}\t{curr_propx}\t{curr_propy}\n")
                    curr_propx = ix
                    curr_prop = ""
                if not p == "O":
                    curr_prop = curr_prop + " " + t[ix]

            entry_line_number +=1

            entry_line_number +=1

id_to_zip_code = dict(zip(range(len(df_data)),df_data["authorZipCode"].values))
id_to_reference = dict(zip(range(len(df_data)),df_data["reference"].values))
id_to_id = dict(zip(range(len(df_data)),df_data["id"].values))
id_to_author_id = dict(zip(range(len(df_data)),df_data["authorId"].values))

for col in df_data.columns[11:]:
    name_col = col.split(" - ")[0]
    try:
        dd= pd.read_csv(f"{args.output_dir}/{name_col}.txt",sep="\t",lineterminator="\n",header=None,error_bad_lines=False,warn_bad_lines=False,names = "id_ prop start end".split())
    except:
        continue
    dd = dd[dd.id_.apply(foo)] # still some errors of parsing... maybe JSON would be a better solution
    if len(dd) <1:continue # if dataframe empty
    dd["code_postal"] = dd.id_.astype(int).map(id_to_zip_code)
    dd["reference"] = dd.id_.astype(int).map(id_to_reference)
    dd["id_contrib"] = dd.id_.astype(int).map(id_to_id)
    dd["author_id"] = dd.id_.astype(int).map(id_to_author_id)
    dd.to_csv(f"{args.output_dir}/{name_col}.txt_cleaned",sep="\t",index=None)