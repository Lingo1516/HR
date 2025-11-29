import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score

st.set_page_config(page_title="IBM HR 戰情室 & 競賽系統", layout="wide")

# ==========================================
# 0. 核心數據處理函式 (穩健版)
# ==========================================
@st.cache_data
def load_and_process_data(file):
    try:
        df = pd.read_csv(file)
        
        # 1. 欄位名稱對照表 (English -> Chinese)
        columns_map = {
            'Age': '年齡', 'Attrition': '離職', 'BusinessTravel': '商務差旅', 'DailyRate': '日薪',
            'Department': '部門', 'DistanceFromHome': '通勤距離', 'Education': '教育程度',
            'EducationField': '教育領域', 'EmployeeCount': '員工數量', 'EmployeeNumber': '員工編號',
            'EnvironmentSatisfaction': '環境滿意度', 'Gender': '性別', 'HourlyRate': '時薪',
            'JobInvolvement': '工作投入度', 'JobLevel': '職級', 'JobRole': '職位',
            'JobSatisfaction': '工作滿意度', 'MaritalStatus': '婚姻狀況', 'MonthlyIncome': '月收入',
            'MonthlyRate': '月費率', 'NumCompaniesWorked': '曾工作公司數量', 'Over18': '年滿18歲',
            'OverTime': '加班', 'PercentSalaryHike': '加薪百分比', 'PerformanceRating': '績效評級',
            'RelationshipSatisfaction': '人際關係滿意度', 'StandardHours': '標準工時',
            'StockOptionLevel': '股票期權級別', 'TotalWorkingYears': '年資',
            'TrainingTimesLastYear': '去年培訓次數', 'WorkLifeBalance': '工作生活平衡',
            'YearsAtCompany': '公司年資', 'YearsInCurrentRole': '目前職位年資',
            'YearsSinceLastPromotion': '距離上次晉升年資', 'YearsWithCurrManager': '與目前經理共事年資'
        }

        # 2. 內容值對照表
        values_map = {
            'Attrition': {'Yes': '是', 'No': '否'},
            'OverTime': {'Yes': '有', 'No': '無'},
            'Gender': {'Female': '女性', 'Male': '男性'},
            'MaritalStatus': {'Single': '單身', 'Married': '已婚', 'Divorced': '離婚'}
        }

        # 翻譯內容
        for col, trans_dict in values_map.items():
            if col in df.columns:
                df[col] = df[col].replace(trans_dict)

        # 翻譯欄位
        df.rename(columns=columns_map, inplace=True)
        
        # 建立數值化欄位 (給 AI 和 統計圖用)
        if '離職' in df.columns:
            df['離職_數值'] = df['離職'].apply(lambda x: 1 if x == '是' else 0)
            
        return df
    except Exception as e:
        return pd.DataFrame() # 回傳空表避免報錯

# ==========================================
# 1. 系統初始化與上傳
# ==========================================
st.title("🏢 IBM HR 戰情室 & 分組競賽系統")
st.markdown("本系統包含 **「數據分析教學」** 與 **「分組留才競賽」** 兩大模組。")

uploaded_file = st.sidebar.file_uploader("📂 請老師上傳 IBM 資料集 (csv)", type=["csv"])

if uploaded_file is not None:
    df = load_and_process_data(uploaded_file)
    if df.empty:
        st.error("資料讀取錯誤，請確認 CSV 格式。")
        st.stop()
    st.sidebar.success(f"✅ 資料載入成功！共 {len(df)} 筆")
else:
    st.info("👈 請先從左側選單上傳資料檔案 (WA_Fn-UseC_-HR-Employee-Attrition.csv)。")
    st.stop()

# 建立分頁
tab1, tab2 = st.tabs(["📊 數據分析戰情室 (教學用)", "⚔️ 分組留才大對決 (競賽用)"])

# ==========================================
# 分頁 1: 數據分析戰情室 (之前的完整功能)
# ==========================================
with tab1:
    st.header("1. 離職原因探索 (EDA)")
    
    # 欄位分類
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    ordinal_cols = ['工作滿意度', '環境滿意度', '人際關係滿意度', '工作投入度', '績效評級', '職級']
    categorical_cols = ['加班', '商務差旅', '部門', '性別', '婚姻狀況', '教育領域', '職位'] + ordinal_cols
    
    # 防呆過濾
    valid_options = [c for c in (numeric_cols + categorical_cols) if c in df.columns]
    if '離職_數值' in valid_options: valid_options.remove('離職_數值')
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("變數關聯分析")
        default_opt = '加班' if '加班' in df.columns else valid_options[0]
        target_factor = st.selectbox("請選擇分析因子：", valid_options, index=valid_options.index(default_opt) if default_opt in valid_options else 0)
        
        # 智慧判斷圖表
        is_categorical = (target_factor in categorical_cols) or \
                         (df[target_factor].dtype == 'object') or \
                         (df[target_factor].nunique() <= 5)
        
        if is_categorical:
            # 長條圖
            group_data = df.groupby(target_factor)['離職_數值'].agg(['mean', 'count']).reset_index()
            group_data.columns = [target_factor, '離職率', '人數']
            group_data['離職率%'] = (group_data['離職率'] * 100).round(1)
            
            fig = px.bar(group_data, x=target_factor, y='離職率%', 
                         title=f"【{target_factor}】各組別離職率",
                         text='離職率%', color='離職率%', color_continuous_scale='Reds')
            fig.update_traces(texttemplate='%{text}%', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        else:
            # 盒鬚圖
            fig = px.box(df, x="離職", y=target_factor, color="離職", 
                         title=f"離職者 vs 在職者的【{target_factor}】差異",
                         color_discrete_map={'是':'#FF4B4B', '否':'#1F77B4'})
            st.plotly_chart(fig, use_container_width=True)
            
            # 數字顯示
            avg_yes = df[df['離職']=='是'][target_factor].mean()
            avg_no = df[df['離職']=='否'][target_factor].mean()
            diff_pct = ((avg_yes - avg_no) / avg_no) * 100
            st.metric("離職者平均 vs 在職者", f"{avg_yes:,.1f} / {avg_no:,.1f}", f"差異 {diff_pct:+.1f}%", delta_color="inverse")

    with col2:
        st.subheader("相關性熱圖")
        corr_cols = ['離職_數值', '月收入', '年齡', '年資', '通勤距離', '工作滿意度']
        real_corr_cols = [c for c in corr_cols if c in df.columns]
        
        if len(real_corr_cols) > 1:
            corr_matrix = df[real_corr_cols].corr()[['離職_數值']].sort_values(by='離職_數值', ascending=False)
            fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', aspect="auto")
            st.plotly_chart(fig_corr, use_container_width=True)
    
    st.divider()
    st.subheader("2. AI 預測模型 & 策略提案")
    c_ai, c_strat = st.columns(2)
    
    with c_ai:
        if st.button("🚀 訓練 AI 模型"):
            # 簡單特徵工程
            drop_cols = ['離職', '員工數量', '員工編號', '年滿18歲', '標準工時', '離職_數值']
            real_drop = [c for c in drop_cols if c in df.columns]
            df_ml = pd.get_dummies(df.drop(real_drop, axis=1), drop_first=True)
            
            X = df_ml
            y = df['離職_數值']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # 特徵重要性
            imp = pd.Series(model.feature_importances_, index=X.columns)
            st.write("**影響離職的前 5 大關鍵因素：**")
            st.bar_chart(imp.nlargest(5), color='#ff4b4b')
            
    with c_strat:
        st.write("請勾選你們的發現與建議：")
        findings = st.multiselect("發現問題：", ["加班過多", "薪資偏低", "年齡太輕", "滿意度低"])
        actions = st.multiselect("建議方案：", ["控制工時", "調整薪資", "留任訪談", "改善環境"])
        if st.button("提交報告"):
            st.balloons()
            st.success("報告已提交！")

# ==========================================
# 分頁 2: 分組留才大對決 (修正後的遊戲)
# ==========================================
with tab2:
    st.header("⚔️ HR 戰情室：人才保衛戰")
    st.info("說明：系統將隨機抽出 5 位員工。請各組運用剛才的分析結果，判斷誰才是真的「高風險離職群」並進行挽留。")

    # 初始化遊戲狀態
    if 'scores' not in st.session_state:
        st.session_state['scores'] = {f"第 {i} 組": 0 for i in range(1, 7)}
    if 'round_data' not in st.session_state:
        st.session_state['round_data'] = None

    # 控制區
    c_ctrl_1, c_ctrl_2 = st.columns([1, 4])
    with c_ctrl_1:
        if st.button("🎲 發牌 (開始新局)", type="primary"):
            st.session_state['round_data'] = df.sample(5).reset_index(drop=True)
            st.session_state['reveal'] = False
            
    if st.button("🔄 重置分數"):
        st.session_state['scores'] = {f"第 {i} 組": 0 for i in range(1, 7)}

    # 顯示戰場
    if st.session_state['round_data'] is not None:
        round_df = st.session_state['round_data']
        
        st.subheader("🧐 員工機密檔案")
        cols = st.columns(5)
        for i, row in round_df.iterrows():
            with cols[i]:
                st.info(f"員工 #{i+1}")
                st.write(f"**職位**: {row.get('職位', 'N/A')}")
                st.write(f"**月薪**: ${row.get('月收入', 0):,}")
                
                ot_status = row.get('加班', '無')
                if ot_status == '有':
                    st.error(f"加班: {ot_status}")
                else:
                    st.success(f"加班: {ot_status}")
                    
                st.write(f"**滿意度**: {row.get('工作滿意度', 0)}")
                st.write(f"**年資**: {row.get('年資', 0)} 年")

        st.divider()
        st.subheader("📝 決策區")
        
        # 選擇組別
        current_team = st.selectbox("現在是哪一組的回合？", list(st.session_state['scores'].keys()))
        
        # 勾選決策
        st.write(f"請 **{current_team}** 決定要花預算挽留誰？(勾選 = 挽留)")
        d_cols = st.columns(5)
        picks = []
        for i in range(5):
            with d_cols[i]:
                picks.append(st.checkbox(f"留 #{i+1}", key=f"pick_{i}"))
        
        if st.button("🚀 確定決策 (揭曉答案)"):
            score_change = 0
            details = []
            
            for i, saved in enumerate(picks):
                is_leaving = (round_df.iloc[i]['離職'] == '是')
                
                if saved: # 救
                    if is_leaving:
                        score_change += 10
                        details.append("✅ 成功挽留 (+10)")
                    else:
                        score_change -= 5
                        details.append("💸 浪費預算 (-5)")
                else: # 不救
                    if is_leaving:
                        score_change -= 10
                        details.append("💀 人才流失 (-10)")
                    else:
                        score_change += 5
                        details.append("😎 精準放生 (+5)")
            
            st.session_state['scores'][current_team] += score_change
            
            st.success(f"本局得分：{score_change} 分！")
            
            # 顯示結果對照
            st.write("### 答案揭曉")
            res_cols = st.columns(5)
            for i, row in round_df.iterrows():
                with res_cols[i]:
                    actual = "想離職" if row['離職'] == '是' else "很忠誠"
                    color = "red" if row['離職'] == '是' else "green"
                    st.markdown(f":{color}[**{actual}**]")
                    st.caption(details[i])

        st.divider()
        st.subheader("🏆 目前排行榜")
        leaderboard = pd.DataFrame(list(st.session_state['scores'].items()), columns=['組別', '分數'])
        leaderboard = leaderboard.sort_values(by='分數', ascending=False)
        st.dataframe(leaderboard, use_container_width=True)
        
    else:
        st.info("請點擊上方的「🎲 發牌」按鈕開始遊戲。")
