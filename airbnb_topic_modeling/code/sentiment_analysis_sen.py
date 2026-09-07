from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
import pandas as pd


MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)

# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.save_pretrained(MODEL)
tokenizer.save_pretrained(MODEL)

#read data
df = pd.read_csv('./data/airbnb_en_n.csv',lineterminator='\n',on_bad_lines='skip')
df_neg = pd.DataFrame(columns=["listing_id","id","date","reviewer_id","reviewer_name","comments"])
for idx, row in df.iterrows():
    neg_flag = False
    sentences = []
    if idx % 1000 == 0:
        print(idx)
    try:
        text = row['comments']
        sentences = text.split('.')
        new_comment=""
    except Exception as e:
        print(e)
    for sentence in sentences:
        if sentence != "":
            try:
                encoded_input = tokenizer(sentence, return_tensors='pt')
                output = model(**encoded_input)
                scores = output[0][0].detach().numpy()
                scores = np.argsort(scores)
                print(scores)
                if scores[2] == 0:
                    new_comment = new_comment + sentence 
                    neg_flag = True
            except Exception as e:
                print(e)
    if neg_flag:
        df_neg.loc[len(df_neg)] = row
        df_neg.loc[len(df_neg)-1,'comments'] = new_comment
df_neg.to_csv("airbnb_en_negatives_only_neg_sen_2.csv", encoding='utf-8')





