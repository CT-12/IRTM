# Import Packages
import pandas as pd
import numpy as np
import string
import math
import re
from nltk.stem import PorterStemmer
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# Load Dataset
print("Loading Dataset...")

num_of_doc = 1095

with open("./data/training.txt", 'r') as f:
    trainset_data = f.readlines()
    trainset_data: list[list[int]] = [list(map(int, train.strip(" \n").split()) )for train in trainset_data]
    trainset_data: dict[int, list[int]] = {train[0]: train[1:] for train in trainset_data}

dataset: pd.DataFrame = pd.DataFrame(index=range(1, num_of_doc+1), columns=['doc_id', 'text', 'label', 'train or test'])

# 把所有 doc 以及他們的 text 放入 dataframe
for i in range(1, num_of_doc+1):
    with open(f"./data/{i}.txt", 'r') as f:
        text = f.read()
        dataset.loc[i, "doc_id"] = i
        dataset.loc[i, "text"] = text

# 把 doc 放入對應的 label
for label in trainset_data:
    for doc_id in trainset_data[label]:
        dataset.loc[doc_id, "label"] = label
        dataset.loc[doc_id, "train or test"] = 'train'

# 把沒有 label 的 doc 標記為 test
dataset.loc[dataset[dataset["label"].isnull()].index, "train or test"] = "test"
# 把沒有 test doc 的 label 標記為 0
dataset.loc[dataset[dataset["label"].isnull()].index, "label"] = 0


# Split into trainset and testset
print("Splitting Dataset into trainset and testset...")

trainset: pd.DataFrame = dataset[dataset["train or test"] == "train"]
testset: pd.DataFrame = dataset[dataset["train or test"] == "test"]

print(f"trainset size: {trainset.shape[0]}")
print(f"testset size: {testset.shape[0]}")


# Text Preprocessing
print("Text Preprocessing...")

with open("./stopwords.txt", 'r') as f:
    stopwords = f.readlines()
    stopwords: list[str] = [stopword.replace("\n", "") for stopword in stopwords]

def remove_punctuation(text: str) -> str:
    punctuation = string.punctuation + '-'
    return ''.join([char for char in text if char not in punctuation])

def remove_number(text: str) -> str:
    return re.sub(r"\d+", '', text)

def lowercase(text: str) -> str:
    return text.lower()

def tokenize(text: str) -> list[str]:
    return text.split()

def remove_stopwords(text: list[str]) -> list[str]:
    return [token for token in text if token not in stopwords]

def stemming(text: list[str]) -> list[str]:
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in text]

def text_preprocessing_pipeline(text: str) -> list[str]:
    text = remove_punctuation(text)
    text = remove_number(text)
    text = lowercase(text)
    text = tokenize(text)
    text = remove_stopwords(text)
    text = stemming(text)
    return text

trainset.loc[:, "text"] = trainset.loc[:, "text"].apply(text_preprocessing_pipeline)
testset.loc[:, "text"] = testset.loc[:, "text"].apply(text_preprocessing_pipeline)


# Make a vocabulary dictionary
def make_dictionary(trainset: pd.DataFrame) -> list[str]:
    dictionary = []
    for i in trainset.index:
        dictionary.extend(trainset.loc[i, "text"])
    dictionary = sorted(list(set(dictionary)))

    return dictionary


# Feature Selection
# Use Log Likelihood Ratio to do feature selection. Select top 500 terms for training.
print("Feature Selection...")

def get_contingency_table(trainset: pd.DataFrame, term: str) -> pd.DataFrame:
    contingency_table = pd.DataFrame(index=trainset["label"].unique(), columns=["present", "absent"])

    for c in contingency_table.index:
        # 有多少在 class c 的文件包含 term
        present = trainset[trainset["label"] == c].loc[:, "text"].apply(
            lambda text: term in text
        ).sum()
        # 有多少在 class c 的文件不包含 term
        absent = trainset[trainset["label"] == c].shape[0] - present
        contingency_table.loc[c, :] = [present, absent]

    return contingency_table

def log_likelihoold_ratio(contingency_table: pd.DataFrame):
    N = contingency_table.loc[:, "present"].sum() + contingency_table.loc[:, "absent"].sum()
    pt = contingency_table.loc[:, "present"].sum() / N
    p = dict()

    for c in contingency_table.index:
        p[c] = contingency_table.loc[c, "present"] / (contingency_table.loc[c, "present"] + contingency_table.loc[c, "absent"])

    H1, H2 = 1, 1
    for c in contingency_table.index:
        H1 *= math.comb(contingency_table.loc[c, "present"]+contingency_table.loc[c, "absent"], contingency_table.loc[c, "present"]) * (pt ** contingency_table.loc[c, "present"]) * ((1 - pt) ** contingency_table.loc[c, "absent"])
        H2 *= math.comb(contingency_table.loc[c, "present"]+contingency_table.loc[c, "absent"], contingency_table.loc[c, "present"]) * (p[c] ** contingency_table.loc[c, "present"]) * ((1 - p[c]) ** contingency_table.loc[c, "absent"])

    LLR = -2 * np.log(H1 / H2)

    return LLR

def feature_selection(trainset: pd.DataFrame, top_k: int = 500):
    dictionary: list[str] = make_dictionary(trainset)
    selected_features = []

    for term in tqdm(dictionary):
        contingency_table = get_contingency_table(trainset, term)
        LLR = log_likelihoold_ratio(contingency_table)
        selected_features.append((term, LLR))

    selected_features = sorted(selected_features, key=lambda x: x[1], reverse=True)

    return [feature[0] for feature in selected_features[:top_k]]

selected_terms = feature_selection(trainset=trainset)    


# Training
print("Training...")

def concat_all_text_in_class_c(trainset: pd.DataFrame, c: int) -> list[str]:
    text = []
    for i in trainset[trainset["label"] == c].index:
        text.extend(trainset.loc[i, "text"])
        
    return text

def count_term_in_class(term: str, text_c: list[str]) -> int:
    return text_c.count(term)

def train_multiNomial_naive_bayes(trainset: pd.DataFrame, dictionary: list[str]) -> tuple[list[int], dict[int, float], pd.DataFrame]:
    # 分類的類別
    classes = trainset["label"].unique().tolist()
    M = len(dictionary)
    N = num_of_doc

    prior_prob = {}
    # conditional probability: P(t|c), size is M *|C|
    cond_prob = pd.DataFrame(np.zeros((M, len(classes))), index=dictionary, columns=classes)
    for c in tqdm(classes):
        # N_c: number of documents in class c
        N_c = trainset["label"].value_counts()[c]
        prior_prob[c] = N_c / N
        text_c: list[str] = concat_all_text_in_class_c(trainset, c)

        for term in dictionary:
            # T_ct: number of term t appears in class c
            T_ct = count_term_in_class(term, text_c)
            cond_prob.loc[term, c] = (T_ct + 1) / (len(text_c) + M)
    
    return classes, prior_prob, cond_prob

classes, prior_prob, cond_prob = train_multiNomial_naive_bayes(trainset=trainset, dictionary=selected_terms)


# Inference
print("Inference...")

def inference(test_doc: list[str], dictionary: list[str], classes: list[int], prior_prob: dict[int, float], cond_prob: pd.DataFrame) -> list[int]:
    scores = {}

    # W: 只留下有在 dictionary 裡面的 term, 其餘忽略掉
    W = [term for term in test_doc if term in dictionary]

    for c in classes:
        scores[c] = np.log(prior_prob[c])

        for term in W:
            scores[c] += np.log(cond_prob.loc[term, c])
    
    return max(scores, key=scores.get)

testset.loc[:, "label"] = testset.loc[:, "text"].apply(lambda text: inference(text, selected_terms, classes, prior_prob, cond_prob))


# Submit to Kaggle
print("Generate result.csv...")

result = {"Id": testset["doc_id"].values, "Value": testset["label"].values}
result = pd.DataFrame(result)
result.to_csv("result.csv", index=False)