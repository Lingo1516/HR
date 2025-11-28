import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score

st.set_page_config(page_title="IBM HR 離職預測黑客松", layout="wide")

# ==========================================
# 0. 強力翻譯函式 (確保英文欄位一定會變成中文)
# ==========================================
@st.cache_data
def load_and_translate_data(file):
    try:
        df = pd.read_csv(file)
        
        # 1. 欄位名稱對照表 (English -> Chinese)
        columns_map = {
            'Age': '年齡', 'Attrition': '離職', 'BusinessTravel': '商務差旅', 'DailyRate': '日薪',
            'Department': '部門', 'DistanceFromHome': '通勤距離', 'Education': '教育程度',
            'EducationField': '教育領域', 'EmployeeCount': '員工數量', 'EmployeeNumber': '員工編號',
            'EnvironmentSatisfaction': '環境滿意度', 'Gender': '性別', 'HourlyRate': '時薪',
            'JobInvolvement': '工作投入度', 'JobLevel': '職級', 'JobRole': '職位角色',
            'JobSatisfaction': '工作滿意度', 'MaritalStatus': '婚姻狀況', 'MonthlyIncome': '月收入',
            'MonthlyRate': '月費率', 'NumCompaniesWorked': '曾工作公司數量', 'Over18': '年滿18歲',
            'OverTime': '加班', 'PercentSalaryHike': '加薪百分比', 'PerformanceRating': '績效評級',
            'RelationshipSatisfaction': '人際關係滿意度', 'StandardHours': '標準工時',
            'StockOptionLevel': '股票期權級別', 'TotalWorkingYears': '總工作年資',
            'TrainingTimesLastYear': '去年培訓次數', 'WorkLifeBalance': '工作生活平衡',
            'YearsAtCompany': '在職年資', 'YearsInCurrentRole': '目前職位年資',
            'YearsSinceLastPromotion': '距離上次晉升年資', 'YearsWithCurrManager': '與目前經理共事年資'
        }

        # 2. 內容值對照表
        values_map = {
            'Attrition': {'Yes': '是', 'No': '否'},
            'BusinessTravel': {'Travel_Rarely': '很少出差', 'Travel_Frequently': '經常出差', 'Non-Travel': '不出差'},
            'Department': {'Sales': '銷售部', 'Research & Development': '研發部', 'Human Resources': '人力資源部'},
            'Gender': {'Female': '女性', 'Male': '男性'},
            'MaritalStatus': {'Single': '單身', 'Married': '已婚', 'Divorced': '離婚'},
            'OverTime': {'Yes': '是', 'No': '否'}
        }

        # 先翻譯內容
        for col, trans_dict in values_map.items():
            if col in df.columns:
                df[col] = df[col].replace(trans_dict)

        # 再翻譯欄位名稱
        df.rename(columns=columns_map, inplace=True)
        
        return df
    except Exception as e:
        st.error(f"資料讀取失敗，請確認檔案格式。錯誤訊息: {e}")
        return pd.DataFrame()

# ==========================================
# 1. 主程式開始
# ==========================================
st.title("📊 IBM HR Analytics：離職數據黑客松")
st.markdown("---")

# 側邊欄上傳
st.sidebar.header("📂 步驟 1：上傳資料")
uploaded_file = st.sidebar.file_uploader("請上傳英文版 csv 檔 (WA_Fn-UseC_-HR-Employee-Attrition.csv)", type=["csv"])

if uploaded_file is not None:
    df = load_and_translate_data(uploaded_file)
    if df.empty:
        st.stop()
    st.sidebar.success("✅ 資料載入並翻譯成功！")
else:
    st.info("👆 請從側邊欄上傳 CSV 檔案以開始分析。")
    st.stop()

# ==========================================
# 2. 數據探索 (EDA)
# ==========================================
st.header("1. 離職原因探索 (Data Discovery)")

# 建立離職數值欄位 (用於計算)
if '離職' in df.columns:
    df['離職_數值'] = df['離職'].apply(lambda x: 1 if x == '是' else 0)

# 定義各種欄位類型
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
ordinal_cols = ['工作滿意度', '環境滿意度', '人際關係滿意度', '工作投入度', '績效評級', '職級']
categorical_cols = ['加班', '商務差旅', '部門', '性別', '婚姻狀況', '教育領域', '職位角色'] + ordinal_cols

# 防呆機制：確保欄位存在
valid_options = [c for c in (numeric_cols + categorical_cols) if c in df.columns]
if '離職_數值' in valid_options: valid_options.remove('離職_數值')
default_opts = [c for c in ['月收入', '年齡', '加班', '工作滿意度'] if c in df.columns]

factors = st.multiselect("請選擇你們想分析的因子：", valid_options, default=default_opts)

col1, col2 = st.columns([2, 1])

with col1:
    if factors:
        target_factor = st.selectbox("詳細觀察哪一個因子？", factors)
        
        # 智慧圖表切換：類別/少數數值 -> 長條圖；連續數值 -> 盒鬚圖
        is_categorical = (target_factor in categorical_cols) or \
                         (df[target_factor].dtype == 'object') or \
                         (df[target_factor].nunique() <= 5)
        
        if is_categorical:
            # === 長條圖 (Bar Chart) ===
            group_data = df.groupby(target_factor)['離職_數值'].agg(['mean', 'count']).reset_index()
            group_data.columns = [target_factor, '離職率', '人數']
            group_data['離職率%'] = (group_data['離職率'] * 100).round(1)
            
            fig = px.bar(group_data, x=target_factor, y='離職率%', 
                         title=f"【{target_factor}】各組別離職率分析",
                         text='離職率%', color='離職率%', color_continuous_scale='Reds')
            fig.update_traces(texttemplate='%{text}%', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
            
            # 自動解讀
            max_row = group_data.loc[group_data['離職率%'].idxmax()]
            st.info(f"💡 發現： **{max_row[target_factor]}** 的群體離職率最高，達到 **{max_row['離職率%']}%**。")
        else:
            # === 盒鬚圖 (Box Plot) ===
            fig = px.box(df, x="離職", y=target_factor, color="離職", 
                         title=f"離職者 vs 在職者的【{target_factor}】分佈差異",
                         color_discrete_map={'是':'#FF4B4B', '否':'#1F77B4'})
            st.plotly_chart(fig, use_container_width=True)
            
            # 平均數比較
            avg_yes = df[df['離職']=='是'][target_factor].mean()
            avg_no = df[df['離職']=='否'][target_factor].mean()
            diff_pct = ((avg_yes - avg_no) / avg_no) * 100
            
            m1, m2, m3 = st.columns(3)
            m1.metric("離職者平均", f"{avg_yes:,.1f}")
            m2.metric("在職者平均", f"{avg_no:,.1f}")
            m3.metric("差異幅度", f"{diff_pct:+.1f}%", delta_color="inverse")

with col2:
    st.subheader("🔥 相關性熱圖")
    corr_cols = [c for c in factors if c in numeric_cols] + ['離職_數值']
    corr_cols = list(set([c for c in corr_cols if c in df.columns]))
    
    if len(corr_cols) > 1:
        corr_matrix = df[corr_cols].corr()[['離職_數值']].sort_values(by='離職_數值', ascending=False)
        fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', aspect="auto")
        st.plotly_chart(fig_corr, use_container_width=True)
        st.caption("紅色=離職推手(正相關) | 藍色=留任因子(負相關)")
    else:
        st.warning("請選擇更多數值因子以顯示熱圖")

# ==========================================
# 3. AI 預測模型
# ==========================================
st.divider()
st.header("2. AI 離職預測模型 (Prediction)")

c_AI_1, c_AI_2 = st.columns(2)
with c_AI_1:
    st.write("設定模型參數，訓練 AI 找出離職關鍵。")
    n_estimators = st.slider("決策樹數量 (Trees)", 10, 200, 100)
    
    # 簡單特徵工程
    drop_cols = ['離職', '員工數量', '員工編號', '年滿18歲', '標準工時', '離職_數值']
    real_drop = [c for c in drop_cols if c in df.columns]
    df_ml = pd.get_dummies(df.drop(real_drop, axis=1), drop_first=True)
    
    if st.button("🚀 訓練 AI 模型"):
        X = df_ml
        y = df['離職_數值']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        st.session_state['model_res'] = {'acc': acc, 'recall': recall, 'model': model, 'feat': X.columns}

with c_AI_2:
    if 'model_res' in st.session_state:
        res = st.session_state['model_res']
        st.subheader("🏆 模型分析結果")
        c1, c2 = st.columns(2)
        c1.metric("預測準確率", f"{res['acc']*100:.1f}%")
        c2.metric("召回率 (抓到離職者)", f"{res['recall']*100:.1f}%")
        
        st.write("**AI 認為影響離職的前 5 大關鍵因素：**")
        imp = pd.Series(res['model'].feature_importances_, index=res['feat'])
        st.bar_chart(imp.nlargest(5), color='#ff4b4b')

# ==========================================
# 4. 策略提案與診斷 (改為勾選式)
# ==========================================
st.divider()
st.header("3. 策略提案與診斷報告")
st.write("請各組根據上方的數據分析結果，勾選你們的發現與建議。")

col_diag, col_act = st.columns(2)

with col_diag:
    st.subheader("🧐 3-1. 數據診斷 (你發現了什麼？)")
    findings = st.multiselect(
        "請勾選數據顯示的離職主因 (可複選)：",
        [
            "加班 (OverTime) - 離職率顯著較高",
            "月收入 (Income) - 離職者薪資偏低",
            "年齡 (Age) - 年輕員工流失率高",
            "工作滿意度 (Satisfaction) - 低滿意度者易離職",
            "通勤距離 (Distance) - 住太遠容易離職",
            "職級 (JobLevel) - 初階員工流動率高",
            "環境滿意度 (Environment) - 工作環境不佳",
            "年資 (YearsAtCompany) - 新進員工撐不久"
        ]
    )

with col_act:
    st.subheader("💡 3-2. 行動方案 (你建議怎麼做？)")
    actions = st.multiselect(
        "請勾選建議採取的行動方案 (可複選)：",
        [
            "嚴格控管加班時數，實施準時下班政策",
            "調整關鍵職位薪資，確保具備市場競爭力",
            "針對年輕員工 (Gen Z) 設計留才計畫",
            "優化新進員工入職培訓 (Onboarding)",
            "實施遠距工作或彈性工時 (解決通勤問題)",
            "進行留任訪談 (Stay Interview) 了解不滿原因",
            "改善辦公室環境與設施",
            "優先招聘資深或穩定性高的人才"
        ]
    )

st.write("")
st.write("---")

# 提交按鈕
if st.button("📝 提交策略分析報告", type="primary"):
    if not findings or not actions:
        st.error("❌ 請至少勾選一個「發現」和一個「行動方案」才能提交！")
    else:
        st.balloons()
        st.success("✅ 報告已成功提交！")
        
        # 產生自動回饋
        st.markdown("### 🤖 【AI 助教回饋】")
        st.write(f"你們組發現了 **{len(findings)}** 個關鍵問題，並提出了 **{len(actions)}** 個解決方案。")
        
        # 簡單的邏輯檢查回饋
        if "加班 (OverTime) - 離職率顯著較高" in findings and "嚴格控管加班時數，實施準時下班政策" in actions:
            st.info("👍 **邏輯正確！** 你們發現了「加班」問題，並對應提出了「控管工時」的解法。")
        elif "加班 (OverTime) - 離職率顯著較高" in findings and "嚴格控管加班時數，實施準時下班政策" not in actions:
            st.warning("⚠️ **提示：** 你們發現了「加班」是主因，但似乎沒有提出對應的解決方案？建議勾選「控管加班」。")
            
        if "月收入 (Income) - 離職者薪資偏低" in findings and "調整關鍵職位薪資，確保具備市場競爭力" in actions:
            st.info("👍 **邏輯正確！** 針對薪資問題提出了調薪策略，這是最直接有效的留才手段。")
