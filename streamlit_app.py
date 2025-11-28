import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score

st.set_page_config(page_title="IBM HR 離職預測黑客松", layout="wide")

# ==========================================
# 0. 定義自動翻譯函式 (Translation Logic)
# ==========================================
@st.cache_data
def load_and_translate_data(file):
    df = pd.read_csv(file)
    
    # 1. 欄位名稱翻譯對照表
    columns_translation = {
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

    # 2. 內容值翻譯對照表
    values_translation = {
        'Attrition': {'Yes': '是', 'No': '否'},
        'BusinessTravel': {'Travel_Rarely': '很少出差', 'Travel_Frequently': '經常出差', 'Non-Travel': '不出差'},
        'Department': {'Sales': '銷售部', 'Research & Development': '研發部', 'Human Resources': '人力資源部'},
        'EducationField': {'Life Sciences': '生命科學', 'Other': '其他', 'Medical': '醫療', 'Marketing': '市場行銷', 'Technical Degree': '技術學位', 'Human Resources': '人力資源'},
        'Gender': {'Female': '女性', 'Male': '男性'},
        'JobRole': {'Sales Executive': '銷售主管', 'Research Scientist': '研究科學家', 'Laboratory Technician': '實驗室技術員', 'Manufacturing Director': '製造總監', 'Healthcare Representative': '醫療代表', 'Manager': '經理', 'Sales Representative': '銷售代表', 'Research Director': '研究總監', 'Human Resources': '人力資源專員'},
        'MaritalStatus': {'Single': '單身', 'Married': '已婚', 'Divorced': '離婚'},
        'Over18': {'Y': '是'},
        'OverTime': {'Yes': '是', 'No': '否'}
    }

    # 執行翻譯
    for col, trans_dict in values_translation.items():
        if col in df.columns:
            df[col] = df[col].replace(trans_dict)

    df.rename(columns=columns_translation, inplace=True)
    return df

# ==========================================
# 1. 介面開始
# ==========================================
st.title("📊 IBM HR Analytics：離職數據黑客松 (全中文版)")
st.markdown("""
### 競賽任務：
請上傳 IBM 原始英文資料集，系統將自動翻譯並進行分析。
找出 **「導致員工離職的 3 大關鍵元兇」**，並據此提出改善策略。
""")

# 資料上傳區
st.sidebar.header("📂 步驟 1：上傳資料集")
uploaded_file = st.sidebar.file_uploader("請上傳英文版 csv 檔 (WA_Fn-UseC_-HR-Employee-Attrition.csv)", type=["csv"])

if uploaded_file is not None:
    # 呼叫翻譯函式
    df = load_and_translate_data(uploaded_file)
    st.success("✅ 資料載入並翻譯成功！")
else:
    st.info("👆 請從側邊欄上傳 CSV 檔案。")
    st.stop()

# ==========================================
# 2. 數據概覽
# ==========================================
with st.expander("🔍 點擊檢視完整資料 (已中文化)", expanded=False):
    st.dataframe(df)
    st.write(f"總筆數：{df.shape[0]} 位員工 | 欄位數：{df.shape[1]}")

# ==========================================
# 3. 自動化關聯分析
# ==========================================
st.header("1. 離職原因探索 (EDA)")
st.write("系統自動分析各變數與 **離職** 的關係。")

# 將離職轉回數字以便計算 (是=1, 否=0)
if '離職' in df.columns:
    df['離職_數值'] = df['離職'].apply(lambda x: 1 if x == '是' else 0)
    
    # 排除非數值欄位，只留下適合分析的
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    # 加上一些重要的類別欄位
    categorical_cols = ['加班', '商務差旅', '部門', '性別', '婚姻狀況']
    
    all_factors = numeric_cols + categorical_cols
    if '離職_數值' in all_factors: all_factors.remove('離職_數值')

    factors = st.multiselect("請選擇你們懷疑的影響因子：", 
                             all_factors,
                             default=['月收入', '年齡', '通勤距離', '工作滿意度', '加班'])
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        target_factor = st.selectbox("詳細觀察哪一個因子？", factors)
        
        # 判斷是數值還是類別
        if df[target_factor].dtype != 'object':
            # 數值型用盒鬚圖
            fig = px.box(df, x="離職", y=target_factor, color="離職", 
                         title=f"離職與在職者的 {target_factor} 差異",
                         color_discrete_map={'是':'#FF4B4B', '否':'#1F77B4'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            # 類別型用長條圖 (計算離職率)
            # 先計算各組的離職率
            group_data = df.groupby(target_factor)['離職_數值'].mean().reset_index()
            group_data['離職率%'] = (group_data['離職_數值'] * 100).round(1)
            
            fig = px.bar(group_data, x=target_factor, y='離職率%', 
                         title=f"不同 {target_factor} 的離職率分析",
                         text='離職率%', color='離職率%')
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🔥 相關性熱圖 (數值型)")
        # 只取數值型欄位做熱圖
        corr_cols = [c for c in factors if c in numeric_cols] + ['離職_數值']
        if len(corr_cols) > 1:
            corr_matrix = df[corr_cols].corr()[['離職_數值']].sort_values(by='離職_數值', ascending=False)
            fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', aspect="auto")
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.write("請選擇更多數值型因子以顯示熱圖")

# ==========================================
# 4. AI 離職預測模型
# ==========================================
st.divider()
st.header("2. AI 預測模型競賽")

col_model_1, col_model_2 = st.columns(2)

with col_model_1:
    st.subheader("⚙️ 模型參數設定")
    n_estimators = st.slider("決策樹數量", 10, 200, 100)
    test_size = st.slider("測試集比例", 0.1, 0.5, 0.2)
    
    # 資料前處理：類別轉數字 (One-Hot Encoding)
    # 排除不必要的欄位
    drop_cols = ['離職', '員工數量', '員工編號', '年滿18歲', '標準工時', '離職_數值']
    df_ml = pd.get_dummies(df.drop(drop_cols, axis=1, errors='ignore'), drop_first=True)
    
    if st.button("🚀 訓練模型並預測"):
        X = df_ml
        y = df['離職_數值']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        st.session_state['model_result'] = {'acc': acc, 'recall': recall, 'model': model, 'features': X.columns}

with col_model_2:
    if 'model_result' in st.session_state:
        res = st.session_state['model_result']
        st.subheader("🏆 模型成績單")
        c1, c2 = st.columns(2)
        c1.metric("準確率 (Accuracy)", f"{res['acc']*100:.1f}%")
        c2.metric("召回率 (Recall)", f"{res['recall']*100:.1f}%", delta_color="inverse")
        
        st.write("---")
        st.write("**對離職影響最大的前 5 個特徵：**")
        feat_importances = pd.Series(res['model'].feature_importances_, index=res['features'])
        st.bar_chart(feat_importances.nlargest(5))

# ==========================================
# 5. 策略提案
# ==========================================
st.divider()
st.header("3. 策略提案")
st.text_area("Q1: 根據數據，哪三個因素是導致離職的主因？")
st.text_area("Q2: 針對這些主因，建議採取的具體行動？")
