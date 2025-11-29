import streamlit as st
import pandas as pd
import numpy as np
import random

st.set_page_config(page_title="HR 戰情室：人才保衛戰", layout="wide")

# ==========================================
# 0. 資料載入與翻譯 (與之前相同)
# ==========================================
@st.cache_data
def load_data(file):
    try:
        df = pd.read_csv(file)
        # 簡易翻譯
        trans = {
            'Age': '年齡', 'Attrition': '離職', 'DailyRate': '日薪', 'Department': '部門',
            'DistanceFromHome': '通勤距離', 'JobRole': '職位', 'JobSatisfaction': '工作滿意度',
            'MonthlyIncome': '月收入', 'OverTime': '加班', 'TotalWorkingYears': '年資',
            'YearsAtCompany': '公司年資', 'TrainingTimesLastYear': '培訓次數'
        }
        df.rename(columns=trans, inplace=True)
        # 內容翻譯
        df['加班'] = df['加班'].replace({'Yes': '有', 'No': '無'})
        df['離職'] = df['離職'].replace({'Yes': '會走', 'No': '留任'})
        return df
    except:
        return pd.DataFrame()

# 初始化 Session State (用於記分)
if 'scores' not in st.session_state:
    st.session_state['scores'] = {f"第 {i} 組": 0 for i in range(1, 7)}
if 'round_data' not in st.session_state:
    st.session_state['round_data'] = None
if 'game_log' not in st.session_state:
    st.session_state['game_log'] = []

# ==========================================
# 1. 遊戲標題與上傳
# ==========================================
st.title("⚔️ HR 戰情室：人才保衛戰 (Talent Defense)")
st.markdown("""
### 📢 競賽規則：
系統會顯示 **5 位員工** 的機密檔案。請各組運用你們的 HR 數據直覺，判斷**誰是真的要離職的人？**
* 🎯 **精準留才 (+10分)**：你選擇留他，而他真的原本要走。 (這才是把錢花在刀口上)
* 💸 **浪費預算 (-5分)**：你選擇留他，但他其實根本不想走。 (你浪費了加薪預算)
* 👋 **人才流失 (-10分)**：你沒留他，結果他真的走了。 (公司損失慘重)
* 😎 **精準放生 (+5分)**：你沒留他，他也真的沒走。 (判斷正確)
""")

# 上傳區
uploaded_file = st.sidebar.file_uploader("老師請先上傳 IBM 資料集 (csv)", type=["csv"])
if uploaded_file:
    df = load_data(uploaded_file)
else:
    st.info("請先上傳資料集以開始遊戲")
    st.stop()

# ==========================================
# 2. 遊戲控制區 (老師操作)
# ==========================================
st.sidebar.divider()
st.sidebar.header("👮‍♂️ 裁判控制台")

# 按鈕：發牌 (隨機抽 5 人)
if st.sidebar.button("🎲 開始新的一局 (發牌)", type="primary"):
    # 隨機抽 5 人，故意讓離職者比例混合
    sample = df.sample(5)
    st.session_state['round_data'] = sample.reset_index(drop=True)
    st.session_state['reveal'] = False # 隱藏答案

# 按鈕：重置分數
if st.sidebar.button("🔄 重置所有分數"):
    st.session_state['scores'] = {f"第 {i} 組": 0 for i in range(1, 7)}
    st.session_state['game_log'] = []
    st.success("分數已歸零！")

# ==========================================
# 3. 戰場顯示區
# ==========================================
if st.session_state['round_data'] is not None:
    round_df = st.session_state['round_data']
    
    st.subheader("🧐 本局高風險名單 (請判斷：救？還是不救？)")
    
    # 顯示員工卡片 (隱藏答案)
    cols = st.columns(5)
    for i, row in round_df.iterrows():
        with cols[i]:
            st.info(f"員工編號 #{i+1}")
            st.write(f"**職位**: {row['職位']}")
            st.write(f"**月薪**: ${row['月收入']:,}")
            
            # 關鍵線索用顏色標示
            if row['加班'] == '有':
                st.error(f"加班: {row['加班']}")
            else:
                st.success(f"加班: {row['加班']}")
                
            st.write(f"**滿意度**: {row['工作滿意度']}/4")
            st.write(f"**年資**: {row['年資']} 年")
            st.write(f"**通勤**: {row['通勤距離']} km")

    st.divider()
    
    # ==========================================
    # 4. 各組下注區
    # ==========================================
    st.subheader("📝 各組決策面板")
    
    # 這裡讓老師輸入各組的決定
    # 為了簡化，假設每一組都針對這 5 個人做同樣的決策 (或是老師指定某組回答)
    # 我們這裡設計成：老師選定現在是哪一組在玩
    
    current_team = st.selectbox("現在是哪一組的回合？", list(st.session_state['scores'].keys()))
    
    st.write(f"請 **{current_team}** 決定要挽留哪幾號員工？ (勾選代表投入預算挽留)")
    
    # 建立 5 個勾選框
    decisions = []
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: save_1 = st.checkbox("留 #1")
    with c2: save_2 = st.checkbox("留 #2")
    with c3: save_3 = st.checkbox("留 #3")
    with c4: save_4 = st.checkbox("留 #4")
    with c5: save_5 = st.checkbox("留 #5")
    
    user_picks = [save_1, save_2, save_3, save_4, save_5]

    # ==========================================
    # 5. 揭曉答案與計分
    # ==========================================
    if st.button("🚀 確定決策 (揭曉答案)"):
        st.session_state['reveal'] = True
        
        # 計算分數
        score_change = 0
        details = []
        
        for i, is_saved in enumerate(user_picks):
            actual_attrition = round_df.iloc[i]['離職'] # 真實答案 (會走/留任)
            emp_name = f"員工 #{i+1}"
            
            if is_saved: # 學生決定救
                if actual_attrition == '會走':
                    res = "✅ 成功挽留！(得10分)"
                    score_change += 10
                else:
                    res = "💸 浪費錢 (他根本不想走) (-5分)"
                    score_change -= 5
            else: # 學生決定不救
                if actual_attrition == '會走':
                    res = "💀 人才流失 (他真的走了) (-10分)"
                    score_change -= 10
                else:
                    res = "😎 判斷正確 (本來就不用救) (+5分)"
                    score_change += 5
            
            details.append(f"{emp_name}: {res}")

        # 更新總分
        st.session_state['scores'][current_team] += score_change
        
        # 記錄 Log
        st.session_state['game_log'].append(f"{current_team} 本局得分: {score_change}")

        # 顯示結果
        st.success(f"🎉 本局結束！ {current_team} 獲得 **{score_change} 分**")
        
        # 顯示詳細答案卡
        st.write("### 🕵️ 真相揭曉")
        res_cols = st.columns(5)
        for i, row in round_df.iterrows():
            with res_cols[i]:
                if row['離職'] == '會走':
                    st.error(f"#{i+1} 其實想離職 😱")
                else:
                    st.success(f"#{i+1} 其實很忠誠 😄")
                
                # 顯示決策結果
                st.caption(details[i])

# ==========================================
# 6. 即時排行榜 (Leaderboard)
# ==========================================
st.divider()
st.header("🏆 戰況排行榜")

# 將字典轉為 DataFrame 並排序
leaderboard = pd.DataFrame(list(st.session_state['scores'].items()), columns=['組別', '總積分'])
leaderboard = leaderboard.sort_values(by='總積分', ascending=False).reset_index(drop=True)

# 用美觀的 Metric 顯示前三名
m1, m2, m3 = st.columns(3)
if len(leaderboard) > 0:
    m1.metric("🥇 第一名", f"{leaderboard.iloc[0]['組別']}", f"{leaderboard.iloc[0]['總積分']} 分")
if len(leaderboard) > 1:
    m2.metric("🥈 第二名", f"{leaderboard.iloc[1]['組別']}", f"{leaderboard.iloc[1]['總積分']} 分")
if len(leaderboard) > 2:
    m3.metric("🥉 第三名", f"{leaderboard.iloc[2]['組別']}", f"{leaderboard.iloc[2]['總積分']} 分")

st.dataframe(leaderboard, use_container_width=True)

# 顯示歷史紀錄
with st.expander("查看詳細對戰紀錄"):
    for log in reversed(st.session_state['game_log']):
        st.text(log)
