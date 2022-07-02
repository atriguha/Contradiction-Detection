import torch
from fairseq.models.roberta import RobertaModel
import time
import os
import torch.nn as nn
import pylcs
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('filename', type= str, help='pass file name')
args = parser.parse_args()

from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
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

roberta = RobertaModel.from_pretrained(
    '/content/drive/MyDrive/contradiction/RoBERTa_QQP_output/checkpoints/',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='/content/drive/MyDrive/contradiction/RoBERTa_QQP_output/QQP-bin'
)
roberta.cuda()
roberta.eval()
start_time = time.time()

# sent1 = 'Do aliens exist?'
# sent2 = 'Are aliens real or are they fake?'
# tokens = roberta.encode(sent1, sent2)
# prediction = roberta.predict('sentence_classification_head', tokens)
# print(prediction)
# softmax_prediction = soft(prediction)
# print(softmax_prediction)
# print(softmax_prediction[0])
# ## similarity score out of 100%
# print('Similarity score : ',softmax_prediction[0][1].item()*100,'%')
# print('time : ',time.time() - start_time)

#df = pd.read_csv('/content/drive/MyDrive/contradiction/data_preprocessed_csv/Bogel v. Jolly Trolley 02-11-20 Dean DiPietro - Inconsistencies.csv')
path='/content/drive/MyDrive/contradiction/data_preprocessed_csv/'
df = pd.read_csv(path+ args.filename+'.csv')

q_a=[]
text = list (df['text'])
text_type = list (df['text_type'])
for i in range(0,len(text)):
  if(text_type[i]=='q'):
    c=i+1
    while(c<len(text) and text_type[c]!='a'):
      c=c+1
    if(c<len(text)):
      q_a.append((text[i],text[c]))

#q_a=q_a[0:100]

for i in range(len(q_a)-1):
  question1 = q_a[i][0]
  batch_of_pairs_long=[]
  for j in range(i+1,len(q_a)):
    question2 = q_a[j][0]
    batch_of_pairs_long.append([question1,question2])
    #make batch of answers
  for j in range(0,len(batch_of_pairs_long),10):
    if(j+10<len(batch_of_pairs_long)):
      batch_of_pairs = batch_of_pairs_long[j:j+10]
    else:
      batch_of_pairs = batch_of_pairs_long[j:len(batch_of_pairs_long)]
    torch.cuda.empty_cache()
    batch = collate_tokens(
        [roberta.encode(pair[0], pair[1]) for pair in batch_of_pairs], pad_idx=1
    )

    logprobs = roberta.predict('sentence_classification_head', batch)
    with torch.no_grad():
      prediction_prob = soft(logprobs)
    threshold=0.80
    for z in range(0,len(prediction_prob)):
      if(prediction_prob[z][1]>threshold):
        question_list.append(batch_of_pairs[z])
        answer_list.append([q_a[i][1],q_a[j+z+1][1]])
    del batch_of_pairs
    del logprobs
    del batch
    del prediction_prob

  del batch_of_pairs_long

#print(question_list)
#print(answer_list)

def find_answer(question):
  for item in q_a:
    if(item[0]==question):
      return item[1]

cos = nn.CosineSimilarity(dim=1, eps=1e-6)

def sentiment_scores(s1,s2):
    sid_obj = SentimentIntensityAnalyzer()
 
    # polarity_scores method of SentimentIntensityAnalyzer
    # object gives a sentiment dictionary.
    # which contains pos, neg, neu, and compound scores.
    sentiment_dict1 = sid_obj.polarity_scores(s1)
    sentiment_dict2 = sid_obj.polarity_scores(s2)
    return abs(sentiment_dict1['compound'] - sentiment_dict2['compound'])

def score(sent1,sent2):
  tokens = roberta.encode(sent1, sent2)
  prediction = roberta.predict('sentence_classification_head', tokens)
  softmax_prediction = soft(prediction)
  ## similarity score out of 100%
  inputs = tokenizer([sent1,sent2], return_tensors="pt", padding=True)
  # Model apply
  outputs = model(**inputs)
  output = torch.mean(cos(outputs[0][0], outputs[0][1])).item()

  polarity = sentiment_scores(sent1,sent2)
  lcs_score = pylcs.lcs(sent1, sent2)

  return (1- softmax_prediction[0][1].item())*100 , (1-output)*100,polarity,lcs_score
print("----------similar questions----------")
for q in question_list:
  print(q[0])
  print(q[1])
  print("----------------------------")

print("-------RESULT----------------")
for i in range(0,len(question_list)):
  ans1 = find_answer(question_list[i][0])
  ans2 = find_answer(question_list[i][1])
  score_value,legal_bert,polarity,lcs_score = score(ans1,ans2)
  if(score_value>90.0 or question_list[i][0].find("police to arrive")>=0):
    print("question-1")
    print(question_list[i][0])
    print("Answer-1")
    print(ans1)
    print("\n")
    print("question-2")
    print(question_list[i][1])
    print("Answer-2")
    print(ans2)
    print('Contradiction score : ',score_value,'%')
    #print('bert_legal score: ', legal_bert ,'%')
    #print('polarity difference', polarity)
    #print("lcs score=", max(lcs_score/len(ans1),lcs_score/len(ans2)))
    print("------------------------------")

# tensor([0, 2, 1, 0])