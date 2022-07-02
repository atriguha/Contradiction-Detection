
import torch
from fairseq.models.roberta import RobertaModel
import time
import os
import torch.nn as nn
import pylcs
import argparse
import re
import nltk
import math

nltk.download('brown')

nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize


from transformers import pipeline
question_answering=pipeline("question-answering")



#!python textblob.download_corpora
from textblob import TextBlob

# from transformers import AutoTokenizer, AutoModel
# tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
# model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
print("yes")
soft = torch.nn.Softmax(dim=1)
from fairseq.data.data_utils import collate_tokens
import pandas as pd
from transformers import AutoModel, AutoTokenizer 
# Define the model repo
model_name = "nlpaueb/legal-bert-base-uncased" 
# Download pytorch model
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

question_list = []
answer_list = []
score_list = []
def jaccard_score(ans1,ans2):
  a1 = (word_tokenize(ans1))
  a2 = (word_tokenize(ans2))
  return jaccard_similarity(a1,a2)

def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))
roberta = RobertaModel.from_pretrained(
    '/content/drive/MyDrive/contradiction/RoBERTa_QQP_output/checkpoints/',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='/content/drive/MyDrive/contradiction/RoBERTa_QQP_output/QQP-bin'
)
roberta.cuda()
roberta.eval()
start_time = time.time()
path='/content/drive/MyDrive/contradiction/data_preprocessed_csv/Dixon v. Chen 03-12-20 Chun-Ming Chen.csv'
df = pd.read_csv(path)
import pylcs
qa_merged=[]
notmatching_threshold=4
q_a=[]
text = list (df['text'])
text_type = list (df['text_type'])
# for i in range(0,len(text)):
#   if(text_type[i]=='q'):
#     c=i+1
#     while(c<len(text) and text_type[c]!='a'):
#       c=c+1
#     if(c<len(text)):
#       q_a.append((text[i],text[c]))
#       qa_merged.append((text[i]+" "+text[c]))

# for i in range(len(qa_merged)-1):
#   question1 = qa_merged[i]
#   batch_of_pairs_long=[]
#   for j in range(i+1,len(qa_merged)):
#     question2 = qa_merged[j]
#     q_similarity = pylcs.lcs(question1, question2)
    
#     if(q_similarity < notmatching_threshold):
      
#       continue
    
#     batch_of_pairs_long.append([question1,question2])
    #make batch of answers
# And my understanding from looking at the Christiana Care emergency room records were that you complained to
# them of neck and right shoulder pain.  Is that correct?
# A.    No, my feet hurt.


# What part of your body hurt the most?
# A.    My elbows and knees.

# Q.  Do you see your blue truck in that photo?
# A.  As far as I can tell, that's my truck on the right.



# Q.  Is that your police vehicle on the right there?
# A.  No, that truck belongs to my supervisor.



# Q.  Okay.  Do you know who the person in the hard hat is next to that vehicle?
# A.  No.  I don't recall who it is.

# Do you recollect anyone other than

# yourself being there that morning?

#  A.    I do not, because it's been two years ago.

#  Q.    Who attended the meeting

#  that morning?
pair0="Do you recollect anyone other than yourself being there that morning? "
pair1= "Who attended the meeting that morning?"
#  A.    John Smith and Sal Minella.
from sentence_transformers import CrossEncoder
model = CrossEncoder('model_name')
scores = model.predict([(pair0, pair1)])
print(scores)
# batch_of_pairs=[]
# batch_of_pairs_long=[]
# noun_1=" "
# noun_2=" "

# # noun0=TextBlob(pair0)
# # noun1=TextBlob(pair1)
# # noun_A=noun0.noun_phrases

# # noun_B=noun1.noun_phrases

# # for noun in noun_A:
# #   noun_1= noun_1+" "+noun

# # for noun in noun_B:
# #   noun_2=noun_2+" "+noun
# torch.cuda.empty_cache()
# batch = collate_tokens(
#   [roberta.encode(pair0, pair1)], pad_idx=1
#   )

# logprobs = roberta.predict('sentence_classification_head', batch)
# with torch.no_grad():
#   prediction_prob = soft(logprobs.detach())
#   print("roberta ",prediction_prob)
# #     threshold=0.80
# #     for z in range(0,len(prediction_prob)):
# #       if(prediction_prob[z][1]>threshold or (jaccard_score(batch_of_pairs[z][0],batch_of_pairs[z][1])>.50 and prediction_prob[z][1]>0.70)):
# #         question_list.append(batch_of_pairs[z])
# #         answer_list.append([q_a[i][1],q_a[j+z+1][1]])

# # Download RoBERTa already finetuned for MNLI
# roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
# roberta.eval()  # disable dropout for evaluation

# # # Encode a pair of sentences and make a prediction
# # # tokens = roberta.encode('Roberta is a heavily optimized version of BERT.', 'Roberta is not very optimized.')
# # # print(roberta.predict('mnli', tokens).argmax())  # 0: contradiction

# # # Encode another pair of sentences
# tokens = roberta.encode(pair0,pair1)
# print(roberta.predict('mnli',tokens).argmax())
#   # 2: entailment
# # def sentiment_scores(s1,s2):
# #     sid_obj = SentimentIntensityAnalyzer()
 
# #     # polarity_scores method of SentimentIntensityAnalyzer
# #     # object gives a sentiment dictionary.
# #     # which contains pos, neg, neu, and compound scores.
# #     sentiment_dict1 = sid_obj.polarity_scores(s1)
# #     sentiment_dict2 = sid_obj.polarity_scores(s2)
# #     return abs(sentiment_dict1['compound'] - sentiment_dict2['compound'])
# lcs_score = pylcs.lcs(pair0, pair1)
# # polarity = sentiment_scores(pair0,pair1)

# # print("jaccard score: ", jaccard_score(pair0,pair1))
# print("lcs score:",lcs_score/min(len(pair0),len(pair1)))
# # print("polarity: ",polarity)
# # def find_answer(question):
# #   for item in q_a:
# #     if(item[0]==question):
# #       return item[1]

# cos = nn.CosineSimilarity(dim=1, eps=1e-6)
# ## similarity score out of 100%
# inputs = tokenizer([pair0,pair1], return_tensors="pt", padding=True)
#   # Model apply
# outputs = model(**inputs)
# output = torch.mean(cos(outputs[0][0], outputs[0][1])).item()
# print(output)

