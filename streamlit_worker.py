
##### 💡[1] 라이브러리 #####
#import os
from io import StringIO, BytesIO
import re 
import pickle
import requests
import pandas as pd
import streamlit as st
# 페이지 설정
st.set_page_config(page_title="행복근복", page_icon="🍀", layout="wide")

import similarity #유사도 점수 계산 파일명
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import numpy as np
from streamlit_extras.tags import tagger_component
from streamlit_echarts import st_echarts

##### 💡[2] 설정(변수/함수) #####
# 1. 경로
#PATH = os.path.dirname(os.path.abspath(__file__))  #로컬
#GITHUB
PATH = "https://raw.githubusercontent.com/delight-es/comwel-recommend/main/data/"
CSS_PATH = "https://raw.githubusercontent.com/delight-es/comwel-recommend/main/streamlit_worker.css"

# 2. 세션 초기화
if 'page' not in st.session_state:
    st.session_state['page'] = '소개'

# 3. CSS 설정
# font
font_css = """<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet"> """
st.markdown(font_css, unsafe_allow_html=True)
# css
def load_css():
    response = requests.get(CSS_PATH)
    if response.status_code == 200:
        css_content = response.text
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    else:
        st.error("css 파일 로딩 실패")
load_css()        

# 4. 함수 - 데이터 로딩
@st.cache_data
def load_data(file_path):
    response = requests.get(file_path)
    response.raise_for_status()
    csv_data = StringIO(response.text) 
    data = pd.read_csv(csv_data)
    return data

@st.cache_data
def load_word_data(file_path):
    response = requests.get(file_path)
    response.raise_for_status()
    csv_data = StringIO(response.text) 
    data = pd.read_csv(csv_data, header=None, names=['단어', 'ID1', 'ID2',
    '가중치', '고유명사', '의미분류', '종성유무', '읽기', '타입', '첫번째품사', '마지막품사', '표현', '색인표현'])
    return data

@st.cache_data
def load_pickle_list(file_path):
    response = requests.get(file_path)
    response.raise_for_status()
    with BytesIO(response.content) as f:
        pickle_list = pickle.load(f)
    return pickle_list

# 5. 변수 - PATH

#로컬 
#data_path = PATH+"\data\data.csv"
#df_all = load_data(data_path)
#job_worddic_path = PATH+"\data\job_words.csv"
#df_job_word = load_word_data(job_worddic_path)
#indust_worddic_path = PATH+"\data\industry_words.csv"
#df_indust_word = load_word_data(indust_worddic_path)
#titles_file_path = PATH+"\data\multi_select_list\\titles_list.pkl" # 피클 파일
#titles_list = load_pickle_list(titles_file_path)
#indust_file_path = PATH+"\data\multi_select_list\indust_list.pkl"
#indust_list = load_pickle_list(indust_file_path)
#special_indust_file_path = PATH+"\data\multi_select_list\special_indust_list.pkl"
#special_indust_list = load_pickle_list(special_indust_file_path)
#job_file_path = PATH+"\data\multi_select_list\job_list.pkl"
#job_list = load_pickle_list(job_file_path)
#body_file_path = PATH+"\data\multi_select_list\\body_list.pkl"
#body_list = load_pickle_list(body_file_path)

#GITHUB
data_path = PATH+"data_reduce.csv"
df_all = load_data(data_path)
print("df_all column: ",df_all.columns)
print("df_all head: ", df_all.head(5))
job_worddic_path = PATH+"job_words.csv"
df_job_word = load_word_data(job_worddic_path)
indust_worddic_path = PATH+"industry_words.csv"
df_indust_word = load_word_data(indust_worddic_path)
titles_file_path = PATH+"multi_select_list/titles_list.pkl" # 피클 파일
titles_list = load_pickle_list(titles_file_path)
indust_file_path = PATH+"multi_select_list/indust_list.pkl"
indust_list = load_pickle_list(indust_file_path)
special_indust_file_path = PATH+"multi_select_list/special_indust_list.pkl"
special_indust_list = load_pickle_list(special_indust_file_path)
job_file_path = PATH+"multi_select_list/job_list.pkl"
job_list = load_pickle_list(job_file_path)
body_file_path = PATH+"multi_select_list/body_list.pkl"
body_list = load_pickle_list(body_file_path)


##### 💡[3] 스트림릿 본문 #####
def run():
    ### 🏠 1) 홈페이지
    # 1-1) 사이드바
    st.sidebar.info("🖥️ 메뉴")
    menu_items = {
        "💡 판정서 추천": "추천",
        "📄 사전 편집": "편집",
        "🏠 소개": "소개",
    }
    for key, value in menu_items.items():
        if st.sidebar.button(key, key=key):
            st.session_state.page = value
            st.rerun()

    ### 🏠 2) 메인 페이지
    if st.session_state.page == "추천":     
        col1, col2 = st.columns([5,1])
        with col1:
            st.write("### 업무상 질병 판정서 추천 서비스")
        with col2:
            st.write("")    
            st.write(':blue[*️⃣ 실무자]')
        
        st.divider()
        st.write("")
        st.markdown("""
            :gray[🔔 세부질병-직종-업종이 서로 관련있어야 정상적인 결과가 출력됩니다!]\n
            (예시) 근골격계 - 요추추간판탈출 - 건설관련기능종사자 - 건축건설공사 \n""")
        st.write("")
        st.write("")
        st.write("")

        if 'selected_index' not in st.session_state:
            st.session_state.selected_index = None


        #업무상 질병, 세부질병
        col1, col2, _, col3, col4 = st.columns([1,2,1,1,2])
        with col1:
            st.write("")
            st.write("")
            st.write("**업무상 질병***")
        with col2:
            pain_op = st.selectbox(" ", options=["근골격계", "뇌심혈관계", "직업성암"])
        with col3:
            st.write("")
            st.write("")
            st.write("**세부 질병***")
        with col4:
            #text_search_pain = st.text_input(" ", value="", key="text_search_pain")
            text_search_pain = st.multiselect(" ", titles_list, key="text_search_pain")
            text_search_pain = ', '.join(text_search_pain)

        #직종, 업종
        st.write("")
        st.write("")
        col5, col6, _, col7, col8 = st.columns([1,2,1,1,2])
        with col5:
            st.write("")
            st.write("")
            st.write("**직종**")
            
        with col6:
            # MultiSelect - 입력 외 검색 X, 여러 입력
            text_search_job = st.multiselect(" ", job_list, key="text_search_job")
            text_search_job = ', '.join(text_search_job)
            st.caption(text_search_job)

        with col7:
            st.write("")
            st.write("")
            st.write("**업종**")
        with col8:
            # MultiSelect - 입력 외 검색 X, 여러 입력
            text_search_indust = st.multiselect(" ", indust_list, key="text_search_indust")
            text_search_indust = ', '.join(text_search_indust)
            st.caption(text_search_indust)
            
        
       
        #연도, 연령대, 부위
        st.write("") 
        st.write("")
        col9, col10, _, col11, col12, _ , col13, col14 = st.columns([1,2, 1, 1,2, 1, 1,2])
        with col9:
            st.write("")
            st.write("")
            st.write("**연도**")
        with col10:
            multi_select_year = st.multiselect(' ', ['전체', '2017', '2018', '2019', '2020', '2021'], default='전체')
            
            if '전체' in multi_select_year:
                multi_select_year = ['전체']
            else:
                if '전체' in multi_select_year:
                    multi_select_year.remove('전체')
        with col11:
            st.write("")
            st.write("")
            st.write("**연령대**")
        with col12:
            multi_select_age = st.multiselect(' ', ['전체', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100'], default='전체')
            
            if '전체' in multi_select_age:
                multi_select_age = ['전체']
            else:
                if '전체' in multi_select_age:
                    multi_select_age.remove('전체')

        with col13:
            st.write("")
            st.write("")
            st.write("**부위**")
        with col14:
            # MultiSelect - 입력 외 검색 X, 여러 입력
            multi_select_body = st.multiselect(" ", body_list, key="multi_select_body", default="전체")
   
            if '전체' in multi_select_body:
                multi_select_body = ['전체']
            else:
                if '전체' in multi_select_body:
                    multi_select_body.remove('전체')

        
        st.write("")
        st.write("")
        st.write("")
        st.write("")

        # 인정여부, 성별, 검색
        st.write("")
        st.write("")
        col15,col16,_, col17,col18,_,  colt1,col19,colt2,_, col20 = st.columns([1,1,0.6, 1,1,0.6, 0.5,0.5,0.8,0.6, 1])
        
        with col15:
            st.write("**인정여부**")
        with col16:
            check_agree = st.checkbox('인정', value=True)
            check_disagree = st.checkbox('불인정', value=True)

        with col17:
            st.write("**성별**")
        with col18:
            check_male = st.checkbox('남성', value=True)
            check_female = st.checkbox('여성', value=True)

        with colt1:
            st.write("")
            st.write("")
            st.write("")
            st.write("업종")
        with col19:
            st.write("**우선**")
            toggle_top_job_indust = st.toggle("", value=True, key=f"toggle_top_job_indust")
        with colt2:
            st.write("")
            st.write("")
            st.write("")
            st.write("직종")

        with col20:
            button_search =  st.button("검색")


        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")

        if button_search:
            toast_text = ':blue[🔍입력한 검색어]\n'

            # 질병
            if text_search_pain == '':
                st.toast("⚠️ **질병** 을 입력하세요! (필수)\n")
            else:
                toast_text += f'* **업무상질병** : {pain_op}\n'
                toast_text += f'* **질병** : {text_search_pain}\n'


            #직종
            if text_search_job != '':
                toast_text += f'* **직종** : {text_search_job}\n'
            else: #직종 값 비었는데
                if toggle_top_job_indust == True: #직종 우선 선택되면
                    toggle_top_job_indust = False #업종 우선 강제
                    st.toast("⚠️ **직종**이 입력되지 않아 **업종 우선순위**로 판정서가 추천됩니다!\n")

            #업종
            if text_search_indust != '':
                toast_text += f"* **업종** : {text_search_indust}"
                if text_search_indust in special_indust_list:
                    toast_text += "\n:red[→ 특진 ⭕]"
            else: #업종 값 비었는데
                if (toggle_top_job_indust == False): #업종 우선 선택되면
                    toggle_top_job_indust = True #직종 우선 강제
                    st.toast("⚠️ **업종**이 입력되지 않아 **직종 우선순위**로 판정서가 추천됩니다!\n")
                    
            if (text_search_pain != '') and (pain_op != ''):
                st.toast(toast_text)

            if toggle_top_job_indust: #True면 직종
                top_job_indust = 'job'
            else: #False면 업종
                top_job_indust = 'industry'


            result = similarity.find_disease_and_similar_job(df_all, pain_op, text_search_pain, text_search_indust, text_search_job, top_job_indust)
            print(f'{pain_op}: ',result)
            print(f'result-columnns: ',result.columns)

            # 더 많은 결과를 위한다면?
            #result = result.head(10) #수치 조정

            try:
                #'disease', 'titles', 'agrees', 'ids', 'orders', 'details',
                #'opinions', 'facts', 'conclusions', 'body', 'new_facts', 'gender'
                #'death', 'age', 'special', 'expert_investigation', 'X_job',
                #'X_industry', 'Y_job', 'Y_industry', 'summarize_conclusions', 'summarize_facts', 
                #'rank', 'overall_similarity'

                result.columns = ['업무상질병', '질병', '인정', '판정번호', '주문', '신청내용', '신청인주장', '인정사실', '위원회결론', '부위', 'new_facts', '성별', 
                '유족', '나이', '특진', '전문조사', '직종키워드', 
                '업종키워드', '직종', '업종', '위원회요약', '인정요약', 'rank', '유사도']

                result.reset_index(drop=True, inplace=True)

                result['순위'] = range(1, len(result) + 1)
                result['연도'] = result['판정번호'].str[:4]
                result['판정번호'] = result['판정번호'].astype(str)
                result['유사도'] = (result['유사도'] * 100).astype(int).astype(str)
                result = result.astype(str)
                part_df = result[['순위', '유사도', '판정번호', '인정', '업무상질병', '질병', '업종', '직종', '연도']]

                #연도 필터링
                if '전체' not in multi_select_year:
                    result = result[result['연도'].isin(multi_select_year)]
                    result['순위'] = range(1, len(result) + 1)
                    result.reset_index(drop=True, inplace=True)
                    part_df = result[['순위', '유사도', '판정번호', '인정', '업무상질병', '질병', '업종', '직종', '연도']]
                
                #나이 필터링
                if '전체' not in multi_select_age:
                    result = result[result['나이'].isin(multi_select_age)]
                    result['순위'] = range(1, len(result) + 1)
                    result.reset_index(drop=True, inplace=True)
                    part_df = result[['순위', '유사도', '판정번호', '인정', '업무상질병', '질병', '업종', '직종', '연도']]

                #부위 필터링
                if '전체' not in multi_select_body:
                    result = result[result['부위'].isin(multi_select_body)]
                    result['순위'] = range(1, len(result) + 1)
                    result.reset_index(drop=True, inplace=True)
                    part_df = result[['순위', '유사도', '판정번호', '인정', '업무상질병', '질병', '업종', '직종', '연도']]

                #인정 필터링
                if not (check_agree and check_disagree):
                    if check_agree and not check_disagree:
                        result = result[result['인정'] == '인정']
                        result['순위'] = range(1, len(result) + 1)
                        result.reset_index(drop=True, inplace=True)
                        part_df = result[['순위', '유사도', '판정번호', '인정', '업무상질병', '질병', '업종', '직종', '연도']]
                    elif check_disagree and not check_agree:
                        result = result[result['인정'] == '불인정']
                        result['순위'] = range(1, len(result) + 1)
                        result.reset_index(drop=True, inplace=True)
                        part_df = result[['순위', '유사도', '판정번호', '인정', '업무상질병', '질병', '업종', '직종', '연도']]

                #성별 필터링
                if not (check_male and check_female):
                    if check_male and not check_female:
                        result = result[result['성별'] == '남성']
                        result['순위'] = range(1, len(result) + 1)
                        print(result[['순위', '성별']])
                        result.reset_index(drop=True, inplace=True)
                        part_df = result[['순위', '유사도', '판정번호', '인정', '업무상질병', '질병', '업종', '직종', '연도']]
                    elif check_female and not check_male:
                        result = result[result['성별'] == '여성']
                        result['순위'] = range(1, len(result) + 1)
                        result.reset_index(drop=True, inplace=True)
                        part_df = result[['순위', '유사도', '판정번호', '인정', '업무상질병', '질병', '업종', '직종', '연도']]

                if len(result) == 0:
                    st.info("⚠️ 조건에 완전히 부합하는 판정서가 존재하지 않아 조건과 유사한 판정서 추천을 진행합니다.")

                    result = similarity.find_disease_and_similar_job(df_all, pain_op, text_search_pain, text_search_indust, text_search_job, top_job_indust)

                    if len(result) > 0:
                        st.write("")
                        st.write("")
                        st.write("")
                        st.write("")
                        st.write("")
                        st.write("")
                        
                        result = similarity.find_disease_and_similar_job(df_all, pain_op, text_search_pain, text_search_indust, text_search_job, top_job_indust)

                        result.columns = ['업무상질병', '질병', '인정', '판정번호', '주문', '신청내용', '신청인주장', '인정사실', '위원회결론', '부위', 'new_facts', '성별', 
                        '유족', '나이', '특진', '전문조사', '직종키워드', 
                        '업종키워드', '직종', '업종', '위원회요약', '인정요약', 'rank', '유사도']

                        result.reset_index(drop=True, inplace=True)

                        result['순위'] = range(1, len(result) + 1)
                        result['연도'] = result['판정번호'].str[:4]
                        result['판정번호'] = result['판정번호'].astype(str)
                        result['유사도'] = (result['유사도'] * 100).astype(int).astype(str)
                        result = result.astype(str)
                        part_df = result[['순위', '유사도', '판정번호', '인정', '업무상질병', '질병', '업종', '직종', '연도']]
                        
                    elif len(result) == 0:
                        raise Exception("여전히 검색결과가 없습니다.")      
                            
                st.write("#### 📄 판정서 분석")
                st.write("")
                st.write("")
                
                _, col1, _,col2, _ = st.columns([0.2,1,0.2,1,0.2])
                with col1:
                    st.write('인정률 그래프')
                with col2:
                    st.write('연도별 인정건수')
                
                _, col1, _,col2, _ = st.columns([0.2,1,0.2,1,0.2])
                with col1:
                    total = len(result)
                    agree_total = len(result[result['인정']=='인정'])
                    agree_per = int((agree_total/total) * 100)
                    disagree_per = 100 - agree_per
                    option = {
                        "tooltip": {
                            "trigger": 'item'
                        },
                        "legend": {
                            "top": '5%',
                            "left": 'center'
                        },
                        "series": [
                            {
                                "name": '인정률 파이 그래프',
                                "type": 'pie',
                                "radius": ['40%', '75%'],
                                "avoidLabelOverlap": "false",
                                "itemStyle": {
                                    "borderRadius": "10",
                                    "borderColor": '#fff',
                                    "borderWidth": "2"
                                },
                                "label": {
                                    "show": "false",
                                    "position": 'center'
                                },
                                "emphasis": {
                                    "label": {
                                        "show": "true",
                                        "fontSize": '20',
                                        "fontWeight": 'bold'
                                    }
                                },
                                "labelLine": {
                                    "show": "true"
                                },
                                "data": [
                                    {"value": agree_per, "name": '인정'},
                                    {"value": disagree_per, "name": '불인정'}
                                ]
                            }
                        ]
                    };

                    st_echarts(options=option, key="2")
                    
                with col2:
                    #연도 그래프
                    option = {
                        "tooltip": {
                            "trigger": 'axis'
                        },
                        "legend": {
                            "data": ['인정', '불인정']
                        },
                        "xAxis": {
                            "type": 'category',
                            "data": ['2017', '2018', '2019', '2020', '2021', '2022']
                        },
                        "yAxis": {
                            "type": 'value'
                        },
                        "series": [
                            {
                                "name": '인정',
                                "type": 'line',
                                "data": [
                                    len(result[(result['연도'] == '2017') & (result['인정'] == '인정')]),
                                    len(result[(result['연도'] == '2018') & (result['인정'] == '인정')]),
                                    len(result[(result['연도'] == '2019') & (result['인정'] == '인정')]),
                                    len(result[(result['연도'] == '2020') & (result['인정'] == '인정')]),
                                    len(result[(result['연도'] == '2021') & (result['인정'] == '인정')]),
                                    len(result[(result['연도'] == '2022') & (result['인정'] == '인정')])]
                            },
                            {
                                "name": '불인정',
                                "type": 'line',
                                "data": [
                                    len(result[(result['연도'] == '2017') & (result['인정'] == '불인정')]),
                                    len(result[(result['연도'] == '2018') & (result['인정'] == '불인정')]),
                                    len(result[(result['연도'] == '2019') & (result['인정'] == '불인정')]),
                                    len(result[(result['연도'] == '2020') & (result['인정'] == '불인정')]),
                                    len(result[(result['연도'] == '2021') & (result['인정'] == '불인정')]),
                                    len(result[(result['연도'] == '2022') & (result['인정'] == '불인정')])]
                            }
                        ]
                    }
                    st_echarts(options=option, key="line_chart")
                    

                st.write("")
                st.write("")
                st.write("")

                st.write("#### 📅 판정서 추천")
                same = False
                st.write("")

                #동일사례
                same_total = len(result[result['유사도']=='100'])
                if same_total > 0:
                    st.write(f'📖 동일 사례 : {same_total}건')
                    same = True
                    st.write("")
                #유사사례
                else:
                    st.write(f'📖 유사 사례 : {len(result)}건')
                    st.write("")

                col0, col1, col2, col3, col4 = st.columns([1, 1, 1, 1, 2])
                with col0:
                    st.write("**순위**")
                with col1: 
                    st.write("**유사도**")
                with col2: 
                    st.write("**판정번호**")
                with col3: 
                    st.write("**인정**")
                with col4: 
                    st.write("**질병**")


                for index, row in part_df.iterrows():
                    # 동일사례 / 유사사례       
                    if same_total > 0:
                            if (str(row['유사도']) != '100') & (same == True):
                                st.write("")
                                st.write("")
                                st.write(f'📖유사 사례 : {len(result) - same_total}건')
                                same = False
                                
                    # 고정 너비 문자열 포맷 사용
                    순위 = f"{str(row['순위']).center(58)}"
                    유사도 = f"{(str(row['유사도'])+'%').center(30)}"
                    인정 = f"{str(row['인정']).center(22)}"
                    판정번호 = f"{str(row['판정번호']).center(47)}"
                    질병 = f"{row['질병'].center(55)}"


                    row_str = f"{순위}{유사도}{판정번호}{인정}{질병}"
                    with st.expander(row_str, expanded=False):
                        df = result.loc[index]
                        
                        st.write("")
                        col1, col2, col3 = st.columns([1, 3, 1])
                        with col1:
                            st.markdown('''
                            :green[**유사도**]
                            ''')
                            tagger_component("",
                            [f"{df['순위']}위"],color_name=["green"])
                        with col2:
                            st.write("")
                            st.markdown(f"""#### :gray[ [{df['업무상질병']}] ] :green[{df['질병']}]""")
                        with col3:
                            st.write("")
                            st.write("")

                        col4, col5 = st.columns([1, 3])
                        with col4:
                            similarity_score = int(df['유사도'])
                            values = [similarity_score, 100 - similarity_score]
                            colors = ['#036635', '#FAFAFA']  #코랄,회색
                            explode = (0.1, 0)
                            fig, ax = plt.subplots()
                            wedges, texts, autotexts = ax.pie(values, autopct='', startangle=90, colors=colors,
                            explode=explode)
                            ax.axis('equal') #원형
                            wedge_center = wedges[0].theta2 - (wedges[0].theta2 - wedges[0].theta1) / 2
                            x = wedges[0].r * 0.7 * np.cos(np.deg2rad(wedge_center)) #값x좌표
                            y = wedges[0].r * 0.7 * np.sin(np.deg2rad(wedge_center)) #값y좌표
                            ax.text(x, y, f'{similarity_score}%', horizontalalignment='center', verticalalignment='center', fontsize=22, color='white') #좌표설정
                            st.pyplot(fig) #그래프 표기
                
                        with col5:
                            col_a, col_b, col_c, col_d = st.columns([1,2,1,2])
                            with col_a:
                                st.write("")
                                st.write('**판정번호**')
                            with col_b:
                                st.code(df['판정번호'])
                            with col_c:
                                st.write("")
                                st.write("**인정여부**")
                            with col_d:
                                st.code(df['인정'])

                            col_e, col_f, col_g, col_h = st.columns([1,2,1,2])
                            with col_e:
                                st.write("")
                                st.write("**업무상질병**")
                            with col_f:
                                st.code(df['업무상질병'])
                            with col_g:
                                st.write("")
                                st.write("**질병**")
                            with col_h:
                                st.code(df['질병'])

                            col_i, col_j, col_k, col_l =  st.columns([1,2,1,2])
                            with col_i:
                                st.write("")
                                st.write("**부위**")
                            with col_j:
                                st.code(df['부위'])
                            with col_k:
                                st.write("")
                                st.write("**연도**")
                            with col_l:
                                st.code(df['연도'])
                        
                    
                        
                        st.write("")
                        st.write("")
                        
                        
                        judge_tab_titles = ['💻 분석', '📃 원본']
                        judge_tab1, judge_tab2 = st.tabs(judge_tab_titles)
                        with judge_tab1:
                            st.write("")
                            st.write("")
                            st.write("")

                            col1, col2 = st.columns([1,4])
                            with col1:
                                st.write("**👧🏻 개인정보**")
                            with col2:         
                                cola, colb, colc, cold = st.columns([1,2,1,2])
                                with cola:
                                    st.write("**성별**")
                                with colb: 
                                    st.write(df['성별'])
                                    
                                with colc:
                                    st.write("**연령대**")
                                with cold:
                                    st.write(df['나이'])
                                
                                cola, colb, colc, cold = st.columns([1,2,1,2])
                                with cola:
                                    st.write("**직종**")
                                with colb:
                                    st.write(df['직종'])
                                with colc:
                                    st.write("**업종**")
                                with cold:
                                    st.write(df['업종'])

                                colb, cold = st.columns([3,3])
                                with colb:
                                    job_lst = df['직종키워드'].replace(' ', '').split(',')
                                    job_lst = ['#' + job for job in job_lst]

                                    tagger_component("", job_lst, color_name="blue")
                                    
                                with cold:
                                    industry_lst = df['업종키워드'].replace(' ', '').split(',')
                                    industry_lst = ['#' + industry for industry in industry_lst]

                                    tagger_component("", job_lst, color_name="violet")




                            st.write("")
                            st.write("")
                            st.write("")
                            st.write("")
                            st.write("")

                            col1, col2 = st.columns([1,4])
                            with col1:
                                st.write("**✒️ 요약**")
                            with col2:
                                st.markdown(f"""* **판정서 결론** \n 
                                \n{df['위원회요약']}""")
                                st.markdown(f"""* **인정사실 요약** \n 
                                \n{df['인정요약']}""")

                            st.write("")
                            st.write("")
                            st.write("")
                            st.write("")
                            st.write("")

                            col1, col2, col3, col4 = st.columns([1,1,1,1])
                            idx = index

                            with col1:
                                st.write("**💡 절차**")
                            with col2:
                                if df['특진'] == 'Y':
                                    special_toggle = st.toggle("특진 여부", value=True, key=f"toggle_special_{idx}")
                                else: 
                                    special_toggle = st.toggle("특진 여부", 
                                    key=f"toggle_special_{idx}")
                            with col3:
                                if df['전문조사'] == 'Y':
                                    detail_toggle = st.toggle("전문조사 여부", value=True, key=f"toggle_detail_{idx}")
                                else:
                                    detail_toggle = st.toggle("전문조사 여부", key=f"toggle_detail_{idx}")
                            with col4:
                                if df['유족'] == 'Y':
                                    death_toggle = st.toggle("유족급여 여부", value=True, key=f"toggle_death_{idx}")
                                else:
                                    death_toggle = st.toggle("유족급여 여부",  key=f"toggle_death_{idx}")
                            st.write("")
                            st.write("")
                            st.write("")
                            st.write("")
                            st.write("")
                        

                        with judge_tab2:
                            def replace_text_patterns(text):
                                # 연속공백 -> 한공백
                                text = re.sub(r'\s{2,}', ' ', text)
                                # ~ -> /~
                                text = text.replace('~', '\~')
                                # '(공백?)-(공백?)' -> '\- '
                                text = re.sub(r'(?<=\s)-\s*|\s-(?=\s)', '\n-', text)
                                # '○' (연속 X) -> 줄바꿈'○'
                                text = re.sub(r'(?<!○)\s*○\s*(?!○)', '\n\n○', text)
                                # '.' -> '.줄바꿈'
                                text = re.sub(r'(?<=[가-힣])\.(?!\d)', '.\n', text)
                                # '- ' -> '줄바꿈-'
                                text = text.replace('- ', '\n-')
                                return text

                            # 각 열(1개)에 위 함수 적용
                            for col in ['주문', '신청내용', '신청인주장', '인정사실', '위원회결론']:
                                df[col] = replace_text_patterns(df[col])
                            
                            st.write("")
                            st.write("")

                            # 각 열(1개) 실제 출력
                            for col in ['주문', '신청내용', '신청인주장', '인정사실', '위원회결론']:
                                st.markdown(f"""<h4 style='text-align: center; color: DarkGreen;'>{col}</h4>""",unsafe_allow_html=True)
                                st.write("")
                                st.write(df[col])
                                st.write("")
                                st.write("")
                                st.write("")
                                st.write("")
                                st.write("")
            except Exception as e:
                #st.write(e)
                st.error("🚨 **검색 결과**가 없습니다. 다시 **검색어**를 입력해주세요!")

    ### 🏠 2) 편집
    elif st.session_state.page == "편집":
        df_diff_in_edit_judge = pd.DataFrame()
        df_diff_in_edit_job = pd.DataFrame()
        df_diff_in_edit_indust = pd.DataFrame()

        col1, col2 = st.columns([5,1])
        with col1:
            st.write("### 업무상 질병 판정서 데이터 수정")
        with col2:
            st.write("")    
            st.write(':blue[*️⃣ 실무자]')
        st.write("")
        rewrite_tab = st.tabs(['📝 판정서', '📗 직종 단어사전', '📘 업종 단어사전' ])
        st.write("")
        st.write("")
        with rewrite_tab[0]:
            st.write("#### 📝 판정서 편집")
            edit_judge = st.data_editor(df_all.head(1000), num_rows="dynamic")
            diff_index = df_all.head(1000)[df_all.head(1000) != edit_judge].dropna(how='all').index
            df_diff_in_edit_judge = edit_judge.loc[diff_index]
            st.write("* 수정된 정보")
            st.dataframe(df_diff_in_edit_judge)
            _, col1, _ = st.columns([1.5,1,1])
            with col1:
                st.write("")
                #저장버튼
                button_judge_save =  st.button("판정서 저장")
                if button_judge_save:
                    if len(df_diff_in_edit_judge) > 0:
                        st.toast('💡 수정된 **판정서** 저장 완료!')
                    else:
                        st.toast('⚠️ **판정서**를 수정해주세요!')

           

        with rewrite_tab[1]:
            st.write("#### 📗 직종 단어사전 편집")
            edit_job_word = st.data_editor(df_job_word, num_rows="dynamic")
            diff_index_job = df_job_word[df_job_word != edit_job_word].dropna(how='all').index
            df_diff_in_edit_job = edit_job_word.loc[diff_index_job]
            st.write("* 수정된 정보")
            st.write(df_diff_in_edit_job)
            _, col1, _ = st.columns([1.5,1,1])
            with col1:
                st.write("")
                #저장버튼
                button_job_save = st.button("직종사전 저장")
                if button_job_save:
                    if len(df_diff_in_edit_job) > 0:
                        st.toast('💡 수정된 **직종사전** 저장 완료!')
                    else:
                        st.toast('⚠️ **직종사전**를 수정해주세요!')

        with rewrite_tab[2]:
            st.write("#### 📘 업종 단어사전 편집")
            edit_indust_word = st.data_editor(df_indust_word, num_rows="dynamic")
            diff_index_indust = df_indust_word[df_indust_word != edit_indust_word].dropna(how='all').index
            df_diff_in_edit_indust = edit_indust_word.loc[diff_index_indust]
            st.write("* 수정된 정보")
            st.write(df_diff_in_edit_indust)
            _, col1, _ = st.columns([1.5,1,1])
            with col1:
                st.write("")
                #저장버튼
                button_indust_save = st.button("업종 사전 저장")
                if button_indust_save:
                    if len(df_diff_in_edit_indust) > 0:
                        st.toast('💡  수정된 **업종사전** 저장 완료!')
                    else:
                        st.toast('⚠️ **업종사전**를 수정해주세요!')
            
    
    ### 🏠 2) 소개
    elif st.session_state.page == "소개":
        style_1 = """
            <style>
            .title-text { 
                color: #202632;
                font-size: 30px; 
                font-weight: 700; 
            }
            .title2-text { 
                color: #2982f0; 
                font-size: 40px;
                font-weight: 600; 
                margin-left: -20px;
            }
            .compass-image { 
                width: 180px; 
                height: auto; 
                margin-right: 20px;
            }
            .nav-link, .nav-link:visited {
                color: #202632 !important; 
                font-size: 20px; 
                text-decoration: none; 
                font-family: 'Noto Sans KR', sans-serif; 
                font-weight: 500; 
            }
            </style>"""
        st.markdown(style_1, unsafe_allow_html=True)

        st.write("## 행복근복 🍀")     
        st.write("")
        st.markdown("""
            :green[행복한 근로복지공단]의 업무 생활을 지원하기 위해,\n
            업무상질병에 대해 **판정서 기반 맞춤 추천 서비스**를 제공해 
            근로자의 업무 효율화를 도와주는 서비스입니다. \n""")

        # 각 카드 내용
        cards = [
            {
                "icon": "📑",
                "title": "맞춤형 사례 분석",
                "description": "신청 건의 핵심 내용 및 원인 파악을 위해, 유사한 업무상 질병 판정서 지원",
                "tags": "#판정서 #분석 #요약"
            },
            {
                "icon": "🧭",
                "title": "업무 효율화", 
                "description": "재해조사를 더 빠르고 공정할 수 있도록, 방향성 및 절차 안내",
                "tags": "#업무효율화 #특진여부 #전문조사"
            }
        ]
        st.markdown("---")

        # 카드 레이아웃 생성
        for card in cards:
            with st.container():
                col1, col2 = st.columns([1, 4])  # 컬럼비율 - 아이콘:1, 내용:4
                with col1:
                    st.markdown(f"<h1 style='color: blue;'>{card['icon']}</h1>", unsafe_allow_html=True)  # 아이콘 표시
                with col2:
                    st.subheader(card["title"])
                    st.caption(card["description"])
                    st.markdown(f"<p style='color: gray;'>{card['tags']}</p>", unsafe_allow_html=True)  # 태그 표시
                st.markdown("---")  # 구분선 추가

        st.markdown("""
            - 입력한 신청 건의 정보와 **유사한 판정서**를 확인할 수 있습니다.
            - 판정서의 **분석 결과**를 확인할 수 있습니다.
            - 분석 결과를 통해 **주의깊게 볼 사항, 절차**를 안내받을 수 있습니다.""")

# 실행
if __name__ == "__main__":
    run()
