import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow
from tensorflow import keras
from sentence_transformers import SentenceTransformer
#import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# from sentence_transformers import SentenceTransformer
import time 


def collaborative_filtering_w(templates,cases):
    print('collaborative filtering start')
    #temp = templates[templates.MainCorpNo == Corp].reset_index(drop = True)
    #case = cases[cases.CorpNo == Corp].reset_index(drop = True)
    temp = templates
    case = cases
    n = len(temp)
    m = len(case)
    #print('Number of template for district ',Corp, ' is ',n)
    #print('Number of cases for this district is ',m)
    
    mat = [[0 for _ in range(n)] for __ in range(m)]
    
    temp_list = list(temp.TemplateId)
    
    df = pd.DataFrame(mat, columns = temp_list)
    
    for i in range(m-1):
        similarities = []
        for j in range(m-1):
            if i != j:
                similarity = np.dot(case.qq_embeddings[i],case.qq_embeddings[j].T).item()
                similarities.append((j,similarity))
        similarities.sort(key = lambda X:X[1],reverse = True)
        similarities = similarities[:10]
        for pair in similarities:
            if not np.isnan(case.TemplateId[pair[0]]):
                df.loc[i,case.TemplateId[pair[0]]] += pair[1]
    
    
    return df

def jaccard_similarity_sentence(sentence1, sentence2):
    # Tokenize the sentences into words
    words1 = set(sentence1.split())
    words2 = set(sentence2.split())

    # Compute the intersection and union
    intersection = words1.intersection(words2)
    union = words1.union(words2)

    # Calculate Jaccard Similarity
    similarity = len(intersection) / len(union)
    
    return similarity

def jaccard_matrix(templates, cases):
    print('jaccard similarity start')
    #temp = templates[templates.MainCorpNo == Corp].reset_index(drop = True)
    #case = cases[cases.CorpNo == Corp].reset_index(drop = True)
    temp = templates
    case = cases
    n = len(temp)
    m = len(case)
    
    mat = [[0 for _ in range(n)] for __ in range(m)]
    
    temp_list = list(temp.TemplateId)
    
    similarity_matrix = pd.DataFrame( mat, columns = temp_list)
    for case_id, case_body in zip(case.index, case['cleaned_description']):
        for template_id, template_body in zip(temp['TemplateId'], temp['cleaned_MessageBody']):
            similarity = jaccard_similarity_sentence(case_body, template_body)
            similarity_matrix.loc[case_id, template_id] = similarity
    return similarity_matrix

def compute_similarity_matrix(templates, cases):
    print('similarity matrix start')
    #temp = templates[templates.MainCorpNo == Corp].reset_index(drop = True)
    #case = cases[cases.CorpNo == Corp].reset_index(drop = True)
    
    temp = templates
    case = cases
    n = len(temp)
    m = len(case)
    
    mat = [[0 for _ in range(n)] for __ in range(m)]
    
    temp_list = list(temp.TemplateId)
    
    similarity_matrix = pd.DataFrame( mat, columns = temp_list)
    for case_id, case_embedding in zip(case.index, case['qa_embeddings']):
        for template_id, template_embedding in zip(temp['TemplateId'], temp['embeddings']):
            similarity = np.dot(case_embedding, template_embedding.T).item()
            similarity_matrix.loc[case_id, template_id] = similarity
    return similarity_matrix

def classification_df(templates,cases):
    #temp = templates[templates.MainCorpNo == Corp].reset_index(drop = True)
    #case = cases[cases.CorpNo == Corp].reset_index(drop = True)
    temp = templates
    case = cases
    
    sim_start = time.time()
    similarity_df = compute_similarity_matrix(temp, case)
    sim_end = time.time()
    print(f"Similarity Matrix: {sim_start - sim_end} seconds")
    
    collab_start = time.time()
    collaborative_df = collaborative_filtering_w(temp,case)
    collab_end = time.time()
    print(f"Collaborataive Filtering: {collab_start - collab_end} seconds")
    
    jaccard_start = time.time()
    jaccard_df = jaccard_matrix(temp,case)
    jaccard_end = time.time()
    print(f"Jaccard Similarity: {collab_start - collab_end} seconds")
    
    n = len(temp)
    m = len(case)
    temp_list = list(temp.TemplateId)
    
    mat = [[0 for _ in range(4)] for __ in range(m*n)]
    
    df = pd.DataFrame(mat,columns = ['similarity_score','collaborative_score','jaccard_score','match'])
    
    cnt = 0
    
    for i in range(m):
        for temp in temp_list:
            df.loc[cnt,'similarity_score'] = similarity_df.loc[i,temp]
            df.loc[cnt,'collaborative_score'] = collaborative_df.loc[i,temp]
            df.loc[cnt,'jaccard_score'] = jaccard_df.loc[i,temp]
            if (not np.isnan(case.TemplateId[i])) and case.TemplateId[i] == temp:
                df.loc[cnt,'match'] = 1
            cnt += 1
    return df




template_matched = pd.read_parquet('fine_tuned_template_matched.parquet')
case_matched = pd.read_parquet('fine_tuned_case_matched.parquet')

comm = pd.read_csv('comm_clean.csv', encoding='ISO-8859-1')

################ create training set ####################
def select_top_10_percent(df):  
    sorted_df = df['TemplateId'].value_counts().reset_index()
    sorted_df.columns = ['TemplateId', 'count']
    top_10_percent = sorted_df.head(round(0.1 * len(sorted_df)))
    return top_10_percent['TemplateId'].tolist()

# Group by corpno and apply the function to each district
corp_case_num = case_matched.groupby('CorpNo')['CaseID'].count().reset_index()
corp_case_num = corp_case_num.rename(columns={'CaseID': 'CaseID_count'})
corp_case_num = corp_case_num.sort_values(by='CaseID_count', ascending=False)
big_district = corp_case_num[corp_case_num.CaseID_count>=100]['CorpNo']
case_matched_f = case_matched[case_matched['CorpNo'].isin(big_district)]

selected_templates = case_matched_f.groupby('CorpNo').apply(select_top_10_percent)

template_id_select = [template_id for template_ids in selected_templates for template_id in template_ids]

template_train = template_matched[template_matched['TemplateId'].isin(template_id_select)].reset_index(drop=True)

comm_filter= comm[comm['TemplateId'].isin(template_id_select)]

case_id_select = comm_filter['CaseID'].tolist()

case_test = case_matched[~case_matched['CaseID'].isin(case_id_select)].reset_index(drop=True)

case_train = case_matched[case_matched['CaseID'].isin(case_id_select)].reset_index(drop=True)


########### generate classification df ####################
start_time = time.time()  # Start timing
model_df = classification_df(template_train, case_train)
end_time = time.time()  # End timing

print(f"Total Time taken: {end_time - start_time} seconds")

model_df.to_parquet('classification_df.parquet')