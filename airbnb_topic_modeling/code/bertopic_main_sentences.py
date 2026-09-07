#conda env bertopic

import numpy as np
import os
import bertopic
import pandas as pd
import re
from bertopic import BERTopic
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import random
from sentence_transformers import SentenceTransformer
import tensorflow_hub

embedding_model = tensorflow_hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

path = os.getcwd()
seed = 280493
np.random.seed(seed)

nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords



def preprocessing(df):
    data_ = df
    data_['Sentences_new'] = data_['comments'].map(lambda x: re.sub('[,\.!?=:()]', '', x))
    data_['Sentences_new'] = data_['Sentences_new'].map(lambda x: x.lower())
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'apartment', 'great', 'us', 'stay', 'place', 'nice', 'good', 'also', 'another', 'would',
                   'gave', 'get', 'everything', 'perfect', 'well', 'need', 'recommend', 'could', 'made', 'thank',
                   'really', 'super', 'excellent', 'amazing', 'definitely', 'thanks', 'recommended'])
    data_ = data_.Sentences_new.values.tolist()

    for i in range(len(data_)):
        text_tokens = word_tokenize(data_[i])
        tokens_without_sw = [word for word in text_tokens if not word in stop_words]
        data_[i] = ' '.join(tokens_without_sw)
    return data_


df_ = pd.read_csv('/home/msensis/Airbnb_paper/data/airbnb_en_negatives_only_neg_sen_2.csv',lineterminator='\n',encoding = "latin-1")

data = preprocessing(df_)


'''
Embedder structure
'''

from bertopic.representation._visual import VisualRepresentation


n_of_topics = 10
min_topic_size_ = 150
topic_model = BERTopic(verbose=True, calculate_probabilities=False, nr_topics=n_of_topics,
                           low_memory=True, embedding_model = embedding_model, representation_model="keybert", umap_args={"random_state": 42}, hdbscan_args={"random_state": 42})
topics, probs = topic_model.fit_transform(data)

tp = []
for i in range(len(topics)):
    tp.append([topics[i], probs[i]])
np.set_printoptions(linewidth=100000)
topics_probs = pd.DataFrame(data=tp, columns=['Topic', 'Prob'])
# topics_probs.to_csv(path + '\\csvs\\TopicProb_' + str(n_of_topics) + '_' +'.csv', sep=';')
topics_probs.to_csv('TopicProb_' + str(n_of_topics) + '_' +'.csv',sep=';')
# Reduce to 3 Topics
# new_topics, new_probs = topic_model.reduce_topics(data, topics, probs, nr_topics=n_of_topics)
# Get Topic Info
# print(topic_model.get_topics())
# representative_docs_ = topic_model.get_representative_docs()
# print(representative_docs_)
topic_info = topic_model.get_topic_info()
print(topic_info)
print(topic_info["Name"])
print(topic_info["Representation"].to_list())
print(topic_info["Representative_Docs"].to_list())
print('------------------')

'''
# Get html visualization
vis_top_class = topic_model.visualize_topics_per_class(topics_per_class=topics_per_class, top_n_topics=len(np.unique(topics)))
vis_top_class.write_html(path + '\\data\\vis_topics_per_class_' + str(min_topic_size_) + '_' + str(top_n_words_) + '_' + str(n_of_topics) + '_' + info + '.html')
vis_top = topic_model.visualize_topics(top_n_topics=len(np.unique(topics)))
vis_top.write_html(path + '\\data\\vis_topics_' + str(min_topic_size_) + '_' + str(top_n_words_) + '_' + str(n_of_topics) + '_' + info + '.html')
vis_heat = topic_model.visualize_heatmap()
vis_heat.write_html(path + '\\data\\vis_heatmap_' + str(min_topic_size_) + '_' + str(top_n_words_) + '_' + str(n_of_topics) + '_' + info + '.html')
vis_bar = topic_model.visualize_barchart(n_words=5)
vis_bar.write_html(path + '\\data\\vis_barchart_' + str(min_topic_size_) + '_' + str(top_n_words_) + '_' + str(n_of_topics) + '_' + info + '.html')

'''