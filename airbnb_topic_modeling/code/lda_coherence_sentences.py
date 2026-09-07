import pandas as pd
import re
# from wordcloud import WordCloud
import matplotlib.pyplot as plt
import gensim
from gensim.utils import simple_preprocess
import nltk
from nltk.corpus import stopwords
import gensim.corpora as corpora
from gensim.models import CoherenceModel    # Compute Coherence Score
import numpy as np
from pprint import pprint
# import IPython
import pickle
import pyLDAvis.gensim_models 
import importlib
from multiprocessing import Process
import os


def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield gensim.utils.simple_preprocess(str(sentence), deacc=True)


def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc))
             if word not in stop_words] for doc in texts]


def plot_wordcloud(args):
    # Create a WordCloud object
    wordcloud = WordCloud(background_color="white", max_words=1000, contour_width=3, contour_color='steelblue')
    # Generate a word cloud
    wordcloud.generate(args[0])
    # Visualize the word cloud
    wordcloud.to_image()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


if __name__ == '__main__':
    # Read data into papers
    path = '/home/msensis/Airbnb_paper/data'

    papers = pd.read_csv(path + '/airbnb_en_negatives_only_neg_sen_2.csv', encoding='latin-1', lineterminator='\n')
    papers['paper_text_processed'] = papers['comments'].map(lambda x: re.sub('[,\.!?=:()]', '', x))

    papers['paper_text_processed'] = papers['paper_text_processed'].map(lambda x: re.sub('<br/>', '', x))
    papers['paper_text_processed'] = papers['paper_text_processed'].map(lambda x: x.lower())
    long_string = ','.join(list(papers['paper_text_processed'].values))
    print(papers['paper_text_processed'])

    nltk.download('stopwords')
    stop_words = stopwords.words('english')

    stop_words.extend(['from', 'apartment', 'great', 'us', 'stay', 'place', 'nice', 'good', 'also', 'another', 'would',
                       'gave', 'get', 'everything', 'perfect', 'well', 'need', 'recommend', 'could', 'made', 'thank',
                       'really', 'super', 'excellent', 'amazing', 'definitely', 'thanks', 'recommended'])

    data = papers.paper_text_processed.values.tolist()
    data_words = list(sent_to_words(data))
    data_words = remove_stopwords(data_words)
    id2word = corpora.Dictionary(data_words)
    id2word.save_as_text('Dictionary.txt')
    texts = data_words

    corpus = [id2word.doc2bow(text) for text in texts]
    num_of_topics = [8]
    topics_range = range(6, 20)    # [7, 13, 17]
    alpha = 5
    beta = 0.1      # ['symmetric', 50]
    info = 'stopwords_ext'
    coherence_list_umass = []
    coherence_list_npmi = []
    for num_topics in num_of_topics:
        print('Model with ' + str(num_topics) + ' topics')
        lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=num_topics,
                                               alpha=alpha, eta=beta, random_state=42)
        topics = lda_model.print_topics()
        lda_model.save("../models/model_" + str(num_topics) + str(info))
        filename = "models_" + str(num_topics) + str(info) + ".txt"
        print(filename)
        fptr = open(filename, "w")
        for temp_topic in topics:
            fptr.write(str(temp_topic) + "\n")
        fptr.close()
        doc_lda = lda_model[corpus]
        LDAvis_data_filepath = '../models/file_' + str(num_topics) + str(info)
        if 1 == 1:
            LDAvis_prepared = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
            with open(LDAvis_data_filepath, 'wb') as f:
                pickle.dump(LDAvis_prepared, f)
        with open(LDAvis_data_filepath, 'rb') as f:
            LDAvis_prepared = pickle.load(f)
        #pyLDAvis.save_html('../models/file_' + str(num_topics) + str(info) + '.html')
        coherence_model_lda_umass = CoherenceModel(model=lda_model, texts=data_words, dictionary=id2word, coherence='u_mass')
        coherence_model_lda_npmi = CoherenceModel(model=lda_model, texts=data_words, dictionary=id2word, coherence='c_npmi')
        coherence_lda_umass = coherence_model_lda_umass.get_coherence()
        coherence_lda_npmi = coherence_model_lda_npmi.get_coherence()
        coherence_list_umass.append(coherence_lda_umass)
        coherence_list_npmi.append(coherence_lda_npmi)
    print(coherence_list_umass)
    print(coherence_list_npmi)
    r = np.arange(6, 20)
    fig, ax = plt.subplots()  # Create the figure and axes object
    plt.xlabel('Number of topics')
    plt.ylabel('Coherence values')
    #plt.plot(r, coherence_list_umass / np.linalg.norm(coherence_list_umass), label="coherence_umass")
    #plt.plot(r, coherence_list_npmi / np.linalg.norm(coherence_list_npmi), label="coherence_npmi")
    #plt.legend(loc="upper right")
    plt.title("Airbnb Topics for Topics " + str(num_topics) + str(info), loc='center')
    plt.show()
