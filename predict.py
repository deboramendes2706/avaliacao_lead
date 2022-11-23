import pickle
import pandas as pd
from config_links import *
from functions import *
import nltk

def preprocess(sample):
    sample =sample.drop('id', axis=1)
    sample = removing_white_spaces_comments(sample)

    df = pd.DataFrame(
        columns=['comment_text', 'stemm_comments'])
    # print(f"---------type_school_column: {df_new_sample['type_school']}")
    df["comment_text"] = sample["comment_text"]
    df = stemming_sentences(df, "comment_text", "stemm_comments")

    return df


def perform_prediction(sample):

    new_sample = preprocess(sample)

    with open(CLASSIFICADOR, 'rb') as f:
        clf = pickle.load(f)

    new_sample["Toxic"] = clf.predict(new_sample["stemm_comments"])

    results = new_sample["Toxic"].tolist()

    new_sample['Toxic_str'] = new_sample['Toxic'].replace([1,0], ['Toxic','Non-Toxic'])
    display(new_sample)

    return new_sample["Toxic"].tolist(), new_sample["Toxic_str"].tolist()

df_test = pd.read_csv(DATAFRAME_TEST)
list_predict, list_predict_str = perform_prediction(df_test)