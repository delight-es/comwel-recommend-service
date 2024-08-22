from transformers import BertTokenizer, BertModel
import torch
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import pickle

# # 저장된 임베딩을 피클 파일에서 불러옵니다.
# with open('title_embeddings.pkl', 'rb') as f:
#     title_embeddings = pickle.load(f)

# BERT 토크나이저와 모델 초기화
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 텍스트를 BERT 임베딩으로 변환하는 함수
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

# bert 코사인 유사도 계산 함수
def calculate_cosine_similarity_bert(text, titles):
    # 텍스트 전처리
    processed_text = preprocess_text(text)
    processed_titles = [preprocess_text(title) for title in titles]

    # BERT 임베딩 계산
    text_embedding = get_bert_embedding(processed_text).detach().numpy().squeeze()
    title_embeddings = np.array([get_bert_embedding(title).detach().numpy().squeeze() for title in processed_titles])
    
    # 코사인 유사도 계산
    similarities = cosine_similarity([text_embedding], title_embeddings)
    return similarities.flatten()

# def calculate_cosine_similarity_bert_with_precomputed(input_text, titles):
#     text_embedding = get_bert_embedding(input_text).detach().numpy().squeeze()
#     title_embeddings_precomputed = np.array([title_embeddings[title] for title in titles])
    
#     similarities = cosine_similarity([text_embedding], title_embeddings_precomputed)
#     return similarities.flatten()

def preprocess_text(text):
    text = text.lower()  # 소문자 변환
    text = re.sub(r'\s+', ' ', text)  # 모든 공백을 하나의 공백으로 변환
    text = re.sub(r'[^\w\s]', '', text)  # 특수 문자 제거
    return text

# vectorize 코사인 유사도 계산 함수
def calculate_cosine_similarity_vectorized(input_text, titles, ngram_range=(1, 3)):
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=ngram_range)
    titles_preprocessed = [preprocess_text(title) for title in titles]
    texts = [preprocess_text(input_text)] + titles_preprocessed
    tfidf_matrix = vectorizer.fit_transform(texts)
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    return cosine_similarities.flatten()

def find_disease_and_similar_job(df, disease, input_disease, input_industry, input_job, priority):
    if disease.lower() not in ['뇌심혈관계', '근골격계', '직업성암']:
        raise ValueError("본 함수는 뇌심혈관계 또는 근골격계 질병에만 적용됩니다.")

    new_df = df[df['disease'] == disease].copy()
    
    # 전체 데이터에 대한 유사도 점수 계산
    similarity_scores = calculate_cosine_similarity_vectorized(input_disease, new_df['titles'])
    new_df['disease_similarity'] = similarity_scores
    
    # 유사도 점수가 1이거나 입력 질병 단어를 포함하는 행만 필터링
    new_df = new_df[(new_df['disease_similarity'] == 1.0) | new_df['titles'].str.contains(input_disease)]
    

    def rank_rows(row, priority):
        match_disease = np.isclose(row['disease_similarity'], 1.0)
        match_industry = row['Y_industry'] == input_industry
        match_job = row['Y_job'] == input_job

        # 질병, 업종, 직종 모두 일치하는 경우
        if match_disease and match_industry and match_job:
            return 1, -row['disease_similarity']

        # 질병 일치 및 우선순위에 따른 추가 조건 일치
        if match_disease:
            if priority == 'industry' and match_industry:
                return 2, -row['disease_similarity']
            if priority == 'job' and match_job:
                return 2, -row['disease_similarity']
            if priority == 'industry' and match_job:
                return 3, -row['disease_similarity']
            if priority == 'job' and match_industry:
                return 3, -row['disease_similarity']
            # 질병 일치하지만 업종, 직종이 일치하지 않는 경우
            if priority == 'industry':
                # 업종 우선순위일 때 업종 유사도에 따라 순위 결정
                industry_similarity = calculate_cosine_similarity_vectorized(input_industry, [row['Y_industry']])[0]
                return 4, -industry_similarity
            elif priority == 'job':
                # 직종 우선순위일 때 직종 유사도에 따라 순위 결정
                job_similarity = calculate_cosine_similarity_vectorized(input_job, [row['Y_job']])[0]
                return 4, -job_similarity

        # 질병 유사도가 완전히 일치하지 않는 경우의 추가 순위, 여기서도 우선순위 고려
        if row['disease_similarity'] < 1:
            if priority == 'industry':
                if match_industry and match_job:
                    return 5, -row['disease_similarity']
                if match_industry:
                    return 6, -row['disease_similarity']
                if match_job:
                    return 7, -row['disease_similarity']
                return 8, -row['disease_similarity']
            elif priority == 'job':
                if match_job and match_industry:
                    return 5, -row['disease_similarity']
                if match_job:
                    return 6, -row['disease_similarity']
                if match_industry:
                    return 7, -row['disease_similarity']
                return 8, -row['disease_similarity']

        return 9, -row['disease_similarity']
    
    # new_df 데이터프레임이 비어 있는지 확인
    if new_df.empty:
        print("No matching data found.")
        return pd.DataFrame()  # 빈 데이터프레임 반환

    # apply 함수 사용하여 rank와 similarity_rank 계산
    ranks = new_df.apply(lambda row: rank_rows(row, priority), axis=1)

    # 결과가 비어 있지 않은 경우 처리
    if not ranks.empty:
        new_df['rank'], new_df['similarity_rank'] = zip(*ranks)
    else:
        print("No valid ranks found.")
        return pd.DataFrame()  # 빈 데이터프레임 반환

    # new_df['rank'], new_df['similarity_rank'] = zip(*new_df.apply(lambda row: rank_rows(row, priority), axis=1))
    result_df = new_df.sort_values(by=['rank', 'similarity_rank'], ascending=True).head(30)
    
    # 각 입력값과의 유사도 계산
    similarity_with_disease = calculate_cosine_similarity_bert(input_disease, result_df['titles'])
    similarity_with_industry = calculate_cosine_similarity_bert(input_industry, result_df['Y_industry'])
    similarity_with_job = calculate_cosine_similarity_bert(input_job, result_df['Y_job'])
    
    # 유사도를 result_df 데이터프레임에 추가
    result_df['similarity_with_disease'] = similarity_with_disease
    result_df['similarity_with_industry'] = similarity_with_industry
    result_df['similarity_with_job'] = similarity_with_job

    # overall_similarity 계산
    # 이전 단계에서 이미 new_df 데이터프레임을 필터링했으므로, 그 결과에 따라 overall_similarity 계산
    for index, row in result_df.iterrows():
        if row['disease_similarity'] == 1.0:
            # 질병 이름이 완전히 일치하는 경우
            result_df.at[index, 'overall_similarity'] = (row['similarity_with_disease'] + row['similarity_with_industry'] + row['similarity_with_job']) / 3
        else:
            # 질병 이름이 부분적으로 일치하는 경우
            result_df.at[index, 'overall_similarity'] = (row['disease_similarity'] + row['similarity_with_industry'] + row['similarity_with_job']) / 3


    # 불필요한 유사도 열 제거
    result_df.drop(columns=['similarity_with_disease', 'similarity_with_industry', 'similarity_with_job', 'similarity_rank', 'disease_similarity'], inplace=True)

    # result_df.drop(columns=['disease_similarity'], inplace=True)

    # 유사도 점수와 정렬 순위에 따라 데이터 프레임 정렬
    return result_df.sort_values(by=['overall_similarity', 'rank'], ascending=False)
#유사도 순서로 하고 싶으면 위에 주석풀어