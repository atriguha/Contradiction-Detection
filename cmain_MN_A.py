import torch
from fairseq.models.roberta import RobertaModel
import time
import os
import torch.nn as nn
import pylcs
import argparse
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize


from transformers import pipeline
question_answering=pipeline("question-answering")
parser = argparse.ArgumentParser()
parser.add_argument('filename', type= str, help='pass file name')
args = parser.parse_args()


print("yes")
soft = torch.nn.Softmax(dim=1)
from fairseq.data.data_utils import collate_tokens
import pandas as pd
from transformers import AutoModel, AutoTokenizer 

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

roberta = RobertaModel.from_pretrained(
    '/content/drive/MyDrive/contradiction/RoBERTa_QQP_output/checkpoints/',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='/content/drive/MyDrive/contradiction/RoBERTa_QQP_output/QQP-bin'
)
print("cuda availability =", torch.cuda.is_available())
roberta.cuda()
roberta.eval()
start_time = time.time()

def ans_chk(answer):
  if(re.search("\?",answer)):
      return 1
  else:
      return 0
      

cos = nn.CosineSimilarity(dim=1, eps=1e-6)

def sentiment_scores(s1,s2):
    sid_obj = SentimentIntensityAnalyzer()
 
    # polarity_scores method of SentimentIntensityAnalyzer
    # object gives a sentiment dictionary.
    # which contains pos, neg, neu, and compound scores.
    sentiment_dict1 = sid_obj.polarity_scores(s1)
    sentiment_dict2 = sid_obj.polarity_scores(s2)
    return abs(sentiment_dict1['compound'] - sentiment_dict2['compound'])
    
def jaccard_score(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))

roberta2 = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
roberta2.cuda()

def score(sent1,sent2):
  
  tokens = roberta2.encode(sent1, sent2)
  prediction = roberta2.predict('mnli', tokens)
  softmax_prediction = soft(prediction.detach())
  
  tokens = roberta.encode(sent1, sent2)
  prediction = roberta.predict('sentence_classification_head', tokens)
  softmax_prediction2 = soft(prediction.detach())
  
  polarity = sentiment_scores(sent1,sent2)
  lcs_score = pylcs.lcs(sent1, sent2)

  return (softmax_prediction[0][0].item()*100 , 1,polarity,lcs_score, (softmax_prediction2[0][1].item())*100)


def ob_sub(q,a):
    threshold_ob = 30
    threshold_sub = 90
    if(len(a)<threshold_ob):
          if(re.search("\?",q)):
              return 1
          if(re.search("[Ll]et's\stalk ",q)):
              return 0
          if(re.search("[Ll]et's\s",q)):
              return 1
          if(re.search("(Am|Is|Are|Was|Were|Will)\s",q)):
              return 1
          else:
              return 0        
    else:
          if(re.search("([Cc]orrect\?|[Rr]ight\?|[Oo]kay\?)$",q)):
              return 1
          if(re.search("[Ll]et's\stalk ",q)):
              return 0
          if(re.search("(Am|Is|Are|Was|Were|Will)\s",q)):
              return 1
          if(re.search("[Ww]hat happened",q)):
              return 0
          if(re.search("([Ww]hen|How long)",q)):
              return 1
          if(re.search("[Ww]here",q)):
              return 1
          if(re.search("[Ww]hy",q)):
              return 0
          if(re.search("[Ww]ho",q)):
              return 1
          if(re.search("(How| how)",q)):
              return 0
          if(re.search("[Ww]hat('s|\sis|\sare)",q) and len(a)<90):
              return 1
          if(re.search("(Do|Does|Did)\s",q)):
              return 1
          if(re.search("(Have|Has|Had)\s",q)):
              return 1
          if(re.search("[Ww]hat",q)):
              return 0
          if(re.search("(do|does|did)",q)):
              return 1
          if(re.search("(have|has|had)\s",q)):
              return 1
          if(re.search("([Gg]ive.*summary|Explain|[Tt]ell me|briefly)",q)):
              return 0
          if(re.search("([Ii]s|[Aa]re|[Aa]m|[Ww]ere|[Ww]as)\s(I|you|he|she|it|we|they|this|these|that|those)",q)):
              return 1
          if(re.search("(Yes|Yeah|No|[Nn]ot really|Right|Correct|That's right).",a)):
              return 1
          if(re.search("\?$",q)):
              return 0
          else:
              return 0
def QA(context,question):
  result = question_answering(question=question, context=context)
  return result['answer']

##return 1=yes, 2=no,3=indeterminate
def y_n_(q,a):
  
  if(re.search("(Yes|Yeah|Right|Correct|That's right)",a)):
      return 1
  elif(re.search("(No|[Nn]ot really)",a)):
      return 2
  else:
      return 3

def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))
    
def jaccard_score(ans1,ans2):
  a1 = (word_tokenize(ans1))
  a2 = (word_tokenize(ans2))
  return jaccard_similarity(a1,a2)


#q_a=[]
def find_answer(question):
  for item in q_a:
    if(item[0]==question):
      return item[1]
      
def find_question(answer):
  for item in q_a:
    if(item[1]==answer):
      return item[0]
          
# def calculate_score(temp_dict):
#   questions = temp_dict['que']
#   answers = temp_dict['ans']
#   for i in range(len(questions)):
#     q_a.append((questions[i],answers[i]))
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
  
  question_list = []
  answer_list = []
  score_list = []
  batch_size=16
  notmatching_threshold=4
  for i in range(len(q_a)-1):
    answer1 = q_a[i][1]
    
    batch_of_pairs_long=[]
    for j in range(i+1,len(q_a)):
      answer2 = q_a[j][1]
      q_similarity = pylcs.lcs(answer1, answer2)
      if(q_similarity < notmatching_threshold):
        continue
      batch_of_pairs_long.append([answer1,answer2])
      #make batch of answers
      
    
    for j in range(0,len(batch_of_pairs_long),batch_size):

        if(j+10<len(batch_of_pairs_long)):
            batch_of_pairs = batch_of_pairs_long[j:j+batch_size]
        else:
            batch_of_pairs = batch_of_pairs_long[j:len(batch_of_pairs_long)]
        torch.cuda.empty_cache()
        batch = collate_tokens(
            [roberta2.encode(pair[0], pair[1]) for pair in batch_of_pairs], pad_idx=1
        )
        
        logprobs = roberta2.predict('mnli', batch)
        with torch.no_grad():

            prediction = logprobs.argmax(dim=1)
        
        for z in range(0,len(prediction)):
            if(prediction[z]==0):
                print(batch_of_pairs)
                answer_list.append(batch_of_pairs[z])
                question_list.append([q_a[i][0],q_a[j+z+1][0]])
            
        del batch_of_pairs
        del logprobs
        del batch
        del prediction
        #del prediction_prob
    del batch_of_pairs_long

print(question_list)
  
# for q in question_list:
    

#     s,_,_,_,_ = score(q[0],q[1])
#     thres=60
#     if(s>thres):



        

#         a=ob_sub(q[0],find_answer(q[0]))
#         q.append(a)
#         b= (ob_sub(q[1],find_answer(q[1])))
#         q.append(b)
#         if(ans_chk(find_answer(q[0]))==0 and ans_chk(find_answer(q[1]))==0):





#             #print("SCORES------------------------------")
#             if(a==1 and b==1):

                
#                 yes_no1=y_n_(q[0],find_answer(q[0]))
#                 yes_no2=y_n_(q[1],find_answer(q[1]))
#                 if((yes_no1==1 or yes_no1==2) and (yes_no2==1 or yes_no2==2)):
#                     if(yes_no1!=yes_no2):

#                         # print("qiestion 1:",q[0],"answer 1:",find_answer(q[0]),yes_no1,"qiestion 2:",q[1],"answer 2:",find_answer(q[1]),yes_no2)
#                         score1=99.0
#                         score_output=[q[0],find_answer(q[0]),q[1],find_answer(q[1]),score1]
#                         score_list.append(score_output)
#                     #print("contradiction")
#                     #if both says yes/both says no.
#                     else:


#                     ##contradict for higher scores
#                         score1=0
#                         ans1 = QA(find_answer(q[0]),q[0])
#                         ans2 = QA(find_answer(q[1]),q[1])
#                         if(len(ans1)>20 or len(ans2)>20):
#                             score1,_,_,s = score(ans1,ans2)
#                         else:
#                             s,_,_,_ = score(ans1,ans2)
#                             score1 = s
                            
                        

#                         score_full_ans,_,_,_ = score(find_answer(q[0]),find_answer(q[1]))
#                         score_final=max(score1,score_full_ans)
#                         score_output=[q[0],find_answer(q[0]),q[1],find_answer(q[1]),score_final]
#                         score_list.append(score_output)

#                 else:
#                     ans1 = QA(find_answer(q[0]),q[0])
#                     ans2 = QA(find_answer(q[1]),q[1])
#                     if(len(ans1)>20 or len(ans2)>20):


#                         score1,_,_,s = score(ans1,ans2)
#                         score_full_ans,_,_,_ = score(find_answer(q[0]),find_answer(q[1]))
#                         score_final=max(score1,score_full_ans)
#                         score_output=[q[0],find_answer(q[0]),q[1],find_answer(q[1]),score_final]
#                         score_list.append(score_output)
                    
#                     else:
#                         score1,_,_,s = score(ans1,ans2)
#                         score_lcs=(1-((s/min(len(ans1),len(ans2)))))*100
#                         score_output=[q[0],find_answer(q[0]),q[1],find_answer(q[1]),score_lcs]
#                         score_list.append(score_output)

#             else:
#                 ans1 = find_answer(q[0])
#                 ans2 = find_answer(q[1])
#                 score1,_,_,s = score(ans1,ans2)
#                 score_output=[q[0],find_answer(q[0]),q[1],find_answer(q[1]),score1]
#                 score_list.append(score_output)
        
    
  
# print("----------------RESULT-----------")
# solution = []
# for i in score_list:
#     if(i[4]>70.0):

#         dic = {}
#         dic['question1'] = i[0]
#         dic['answer1'] = i[1]
#         dic['question2'] = i[2]
#         dic['answer2'] = i[3]
#         dic['score'] = i[4]
#         solution.append(dic)
        
#         print("question1",i[0])
#         print("answer1",i[1])
#         print("question2",i[2])
#         print("answer2",i[3])
#         print("score",i[4])

# #return solution




