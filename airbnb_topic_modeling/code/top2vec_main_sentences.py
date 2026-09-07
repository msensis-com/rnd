import pandas as pd
from top2vec import Top2Vec
import sys

file_path = "/home/msensis/Airbnb_paper/data/airbnb_en_negatives_only_neg_sen_2.csv"
df = pd.read_csv(
    file_path, encoding='latin-1', lineterminator='\n'
)
df.head()
docs_bad = df["comments"].values.tolist()
print(docs_bad[1])
print(docs_bad[28286])
print(len(docs_bad))

topic_model = Top2Vec(
    docs_bad,
    embedding_model="universal-sentence-encoder-multilingual-large",
    speed="deep-learn",
    ngram_vocab=True,
    min_count=25,
    ngram_vocab_args={"connector_words": "phrases.ENGLISH_CONNECTOR_WORDS"},
    hdbscan_args={"random_state": 42},
    umap_args={"random_state": 42}
)

print(topic_model.get_num_topics())
num_of_topics = topic_model.get_num_topics()
topic_words, word_scores, topic_nums = topic_model.get_topics(num_of_topics)

for i in range(10):
    print(f"Topic {topic_nums[i]}: {', '.join(topic_words[i])}")

