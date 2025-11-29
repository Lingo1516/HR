import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score

st.set_page_config(page_title="IBM HR 綜藝競賽系統", layout="wide")

# ==========================================
# 0. 核心資料處理
# ==========================================
@st.cache_data
def load_and_process_data(file):
    try:
        df = pd.read_csv(file)
        columns_map = {
            'Age': '年齡', 'Attrition': '離職', 'DailyRate': '日薪', 'Department': '部門',
            'DistanceFromHome': '通勤距離', 'JobRole': '職位', 'JobSatisfaction': '工作滿意度',
            'MonthlyIncome': '月收入', 'OverTime': '加班', 'TotalWorkingYears': '年資',
            'YearsAtCompany': '公司年資'
        }
        values_map = {
            'Attrition': {'Yes': '是', 'No': '否'},
            'OverTime': {'Yes': '有', 'No': '無'},
            'Gender': {'Female': '女性', 'Male': '男性'},
        }
        for col, trans_dict in values_map.items():
            if col in df.columns:
                df[col] = df[col].replace(trans_dict)
        df.rename(columns=columns_map, inplace=True)
        if '離職' in df.columns:
            df['離職_數值'] = df['離職'].apply(lambda x: 1 if x == '是' else 0)
        return df
    except:
        return pd.DataFrame()

# ==========================================
# 1. 系統初始化
# ==========================================
st.title("🎰 IBM HR 戰情室 & 綜藝競賽系統")

uploaded_file = st.sidebar.file_uploader("📂 老師請上傳 CSV", type=["csv"])
if uploaded_file is not None:
    df = load_and_process_data(uploaded_file)
else:
    st.info("請先上傳資料檔案 (WA_Fn-UseC_-HR-Employee-Attrition.csv)")
    st.stop()

tab1, tab2 = st.tabs(["📊 數據分析教學", "🎡 分組留才大賭桌"])

# ==========================================
# 分頁 1: 數據分析 (維持原樣，精簡顯示)
# ==========================================
with tab1:
    st.header("1. 離職原因探索 (EDA)")
    # 簡單列出幾個關鍵圖表
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("加班 vs 離職率")
        if '加班' in df.columns:
            otp = df.groupby('加班')['離職_數值'].mean().reset_index()
            fig = px.bar(otp, x='加班', y='離職_數值', title="加班者的離職率顯著較高", color='離職_數值', color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("月收入 vs 離職 (盒鬚圖)")
        if '月收入' in df.columns:
            fig2 = px.box(df, x='離職', y='月收入', color='離職', title="離職者薪資普遍較低")
            st.plotly_chart(fig2, use_container_width=True)

# ==========================================
# 分頁 2: 綜藝大賭桌 (全新設計)
# ==========================================
with tab2:
    st.header("🎡 HR 留才大賭桌 (Group Battle)")
    st.markdown("### 規則：\n1. 系統發出 5 張員工牌。\n2. 六個小組同時下注，勾選要挽留的人。\n3. 轉動幸運輪盤，被選中的組別 **本局分數加倍**！")

    # 初始化遊戲狀態
    if 'scores' not in st.session_state:
        st.session_state['scores'] = {f"第{i}組": 0 for i in range(1, 7)}
    if 'round_data' not in st.session_state:
        st.session_state['round_data'] = None
    if 'lucky_team' not in st.session_state:
        st.session_state['lucky_team'] = None

    # 控制區
    c_ctrl_1, c_ctrl_2, c_ctrl_3 = st.columns([1, 1, 3])
    with c_ctrl_1:
        if st.button("🎲 重新發牌 (New Round)", type="primary"):
            st.session_state['round_data'] = df.sample(5).reset_index(drop=True)
            st.session_state['lucky_team'] = None
            # 清空上一局的勾選狀態 (透過 Rerun)
            st.rerun()
            
    with c_ctrl_2:
        if st.button("🧹 重置分數"):
            st.session_state['scores'] = {f"第{i}組": 0 for i in range(1, 7)}
            st.session_state['lucky_team'] = None

    # 顯示戰場
    if st.session_state['round_data'] is not None:
        round_df = st.session_state['round_data']
        
        # --- A. 員工牌面 ---
        st.divider()
        st.subheader("🧐 員工機密檔案")
        cols = st.columns(5)
        for i, row in round_df.iterrows():
            with cols[i]:
                st.info(f"員工 #{i+1}")
                st.write(f"**月薪**: ${row.get('月收入', 0):,}")
                ot = row.get('加班', '無')
                if ot == '有': st.error(f"加班: {ot}")
                else: st.success(f"加班: {ot}")
                st.write(f"**滿意度**: {row.get('工作滿意度', 0)}")
                st.write(f"**年資**: {row.get('年資', 0)} 年")

        # --- B. 下注大賭桌 (矩陣顯示) ---
        st.divider()
        st.subheader("📝 各組決策看板 (Betting Board)")
        st.info("請老師詢問各組決定後，在此統一勾選。全班都看得到誰選了誰！")
        
        # 建立 6x5 的勾選矩陣
        # 使用 st.columns 建立表頭
        h1, h2, h3, h4, h5, h6 = st.columns([1.5, 1, 1, 1, 1, 1])
        h1.markdown("**組別**")
        h2.markdown("留 #1")
        h3.markdown("留 #2")
        h4.markdown("留 #3")
        h5.markdown("留 #4")
        h6.markdown("留 #5")
        
        team_picks = {}
        
        # 迴圈建立 6 組的勾選列
        for team_name in st.session_state['scores'].keys():
            r1, r2, r3, r4, r5, r6 = st.columns([1.5, 1, 1, 1, 1, 1])
            r1.markdown(f"### 🚩 {team_name}")
            
            # 每一組的 5 個勾選框
            p1 = r2.checkbox("", key=f"{team_name}_1")
            p2 = r3.checkbox("", key=f"{team_name}_2")
            p3 = r4.checkbox("", key=f"{team_name}_3")
            p4 = r5.checkbox("", key=f"{team_name}_4")
            p5 = r6.checkbox("", key=f"{team_name}_5")
            
            team_picks[team_name] = [p1, p2, p3, p4, p5]

        st.divider()

        # --- C. 幸運輪盤與結算 ---
        col_spin, col_submit = st.columns([1, 2])
        
        with col_spin:
            st.write("#### 🎡 Lucky Time")
            if st.button("轉動幸運輪盤！"):
                # 模擬轉動動畫
                placeholder = st.empty()
                teams = list(st.session_state['scores'].keys())
                for _ in range(15):
                    rand_team = np.random.choice(teams)
                    placeholder.markdown(f"### 🎰 {rand_team} ...")
                    time.sleep(0.1)
                
                lucky = np.random.choice(teams)
                st.session_state['lucky_team'] = lucky
                placeholder.markdown(f"### 🎉 幸運星：{lucky} (分數 x2)！")
                
            if st.session_state['lucky_team']:
                st.success(f"本局 **{st.session_state['lucky_team']}** 得分將加倍！")

        with col_submit:
            st.write("#### 🚀 結算時刻")
            if st.button("揭曉答案 & 計算總分", type="primary", use_container_width=True):
                st.write("### 📢 本局戰報")
                
                # 先顯示正確答案
                ans_cols = st.columns(5)
                answers = []
                for i, row in round_df.iterrows():
                    is_leaving = (row['離職'] == '是')
                    answers.append(is_leaving)
                    with ans_cols[i]:
                        if is_leaving: st.error(f"#{i+1} 其實想離職")
                        else: st.success(f"#{i+1} 其實很忠誠")

                # 計算每一組的分數
                for team, picks in team_picks.items():
                    round_score = 0
                    msg = []
                    
                    for i, picked in enumerate(picks):
                        actual_leaving = answers[i]
                        if picked: # 救
                            if actual_leaving: round_score += 10 # 救對了
                            else: round_score -= 5 # 浪費錢
                        else: # 不救
                            if actual_leaving: round_score -= 10 # 死掉了
                            else: round_score += 5 # 判斷正確
                    
                    # 幸運輪盤加成
                    is_lucky = (team == st.session_state['lucky_team'])
                    if is_lucky:
                        round_score *= 2
                        
                    st.session_state['scores'][team] += round_score
                    
                    # 顯示該組結果
                    luck_icon = "🍀" if is_lucky else ""
                    st.write(f"**{team}** {luck_icon}: 本局得 **{round_score}** 分 (目前總分: {st.session_state['scores'][team]})")

        # --- D. 總排行榜 ---
        st.header("🏆 總積分排行榜")
        lb_df = pd.DataFrame(list(st.session_state['scores'].items()), columns=['組別', '分數'])
        lb_df = lb_df.sort_values(by='分數', ascending=False)
        
        # 視覺化長條圖
        fig_lb = px.bar(lb_df, y='組別', x='分數', text='分數', orientation='h', 
                        color='分數', color_continuous_scale='Greens')
        st.plotly_chart(fig_lb, use_container_width=True)

    else:
        st.info("請點擊左上角「🎲 重新發牌」開始第一回合！")
