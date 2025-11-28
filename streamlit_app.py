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

# 側邊欄上傳
st.sidebar.header("📂 步驟 1：上傳資料")
uploaded_file = st.sidebar.file_uploader("請上傳英文版 csv 檔", type=["csv"])

if uploaded_file is not None:
    df = load_and_translate_data(uploaded_file)
    if df.empty:
        st.stop()
    st.success("✅ 資料載入並翻譯成功！")
else:
    st.info("👆 請從側邊欄上傳 CSV 檔案 (WA_Fn-UseC_-HR-Employee-Attrition.csv)。")
    st.stop()

# ==========================================
# 2. 數據探索 (EDA) - 修復了圖表錯誤與KeyError
# ==========================================
st.header("1. 離職原因探索")

# 建立離職數值欄位 (用於計算)
if '離職' in df.columns:
    df['離職_數值'] = df['離職'].apply(lambda x: 1 if x == '是' else 0)

# 定義各種欄位類型
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
# 這些是我們希望強制用「長條圖」看的欄位
ordinal_cols = ['工作滿意度', '環境滿意度', '人際關係滿意度', '工作投入度', '績效評級', '職級']
categorical_cols = ['加班', '商務差旅', '部門', '性別', '婚姻狀況', '教育領域', '職位角色'] + ordinal_cols

# 確保下拉選單只顯示「真正存在」的欄位 (防呆機制)
valid_options = [c for c in (numeric_cols + categorical_cols) if c in df.columns]
if '離職_數值' in valid_options: valid_options.remove('離職_數值')

# 設定預設值 (如果欄位存在才設為預設，避免報錯)
default_opts = [c for c in ['月收入', '年齡', '加班', '工作滿意度'] if c in df.columns]

factors = st.multiselect("選擇分析因子：", valid_options, default=default_opts)

col1, col2 = st.columns([2, 1])

with col1:
    if factors:
        target_factor = st.selectbox("詳細觀察哪一個因子？", factors)
        
        # 智慧判斷圖表類型：如果是類別，或數值種類很少(如1-4分)，都用長條圖
        is_categorical = (target_factor in categorical_cols) or \
                         (df[target_factor].dtype == 'object') or \
                         (df[target_factor].nunique() <= 5)
        
        if is_categorical:
            # === 畫長條圖 (顯示離職率 %) ===
            group_data = df.groupby(target_factor)['離職_數值'].agg(['mean', 'count']).reset_index()
            group_data.columns = [target_factor, '離職率', '人數']
            group_data['離職率%'] = (group_data['離職率'] * 100).round(1)
            
            fig = px.bar(group_data, x=target_factor, y='離職率%', 
                         title=f"【{target_factor}】各組別的離職率",
                         text='離職率%', color='離職率%', color_continuous_scale='Reds')
            fig.update_traces(texttemplate='%{text}%', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        else:
            # === 畫盒鬚圖 (顯示分佈差異) ===
            fig = px.box(df, x="離職", y=target_factor, color="離職", 
                         title=f"離職與在職者的【{target_factor}】差異",
                         color_discrete_map={'是':'#FF4B4B', '否':'#1F77B4'})
            st.plotly_chart(fig, use_container_width=True)
            
            # 顯示平均數差異
            avg_yes = df[df['離職']=='是'][target_factor].mean()
            avg_no = df[df['離職']=='否'][target_factor].mean()
            diff_pct = ((avg_yes - avg_no) / avg_no) * 100
            
            m1, m2, m3 = st.columns(3)
            m1.metric("離職者平均", f"{avg_yes:.1f}")
            m2.metric("在職者平均", f"{avg_no:.1f}")
            m3.metric("差異", f"{diff_pct:+.1f}%", delta_color="inverse")

with col2:
    st.subheader("🔥 相關性熱圖")
    # 只取數值型欄位
    corr_cols = [c for c in factors if c in numeric_cols] + ['離職_數值']
    # 去除重複並確認欄位存在
    corr_cols = list(set([c for c in corr_cols if c in df.columns]))
    
    if len(corr_cols) > 1:
        corr_matrix = df[corr_cols].corr()[['離職_數值']].sort_values(by='離職_數值', ascending=False)
        fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', aspect="auto")
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.write("請選擇更多數值因子")

# ==========================================
# 3. AI 預測模型
# ==========================================
st.divider()
st.header("2. AI 離職預測模型")

c_AI_1, c_AI_2 = st.columns(2)
with c_AI_1:
    n_estimators = st.slider("決策樹數量", 10, 200, 100)
    
    # 簡單特徵工程
    drop_cols = ['離職', '員工數量', '員工編號', '年滿18歲', '標準工時', '離職_數值']
    # 只刪除存在的欄位
    real_drop = [c for c in drop_cols if c in df.columns]
    
    df_ml = pd.get_dummies(df.drop(real_drop, axis=1), drop_first=True)
    
    if st.button("🚀 訓練模型"):
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
        st.subheader("🏆 模型結果")
        st.metric("準確率", f"{res['acc']*100:.1f}%")
        st.metric("召回率 (抓到離職者的比例)", f"{res['recall']*100:.1f}%")
        
        st.write("**關鍵離職因子 (Top 5):**")
        imp = pd.Series(res['model'].feature_importances_, index=res['feat'])
        st.bar_chart(imp.nlargest(5))

# ==========================================
# 4. 策略提案區
# ==========================================
st.divider()
st.header("3. 策略提案")
st.text_area("Q1: 數據顯示哪三個因素是離職主因？")
