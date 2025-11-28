import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="IBM HR 離職預測黑客松", layout="wide")

st.title("📊 IBM HR Analytics：離職數據黑客松")
st.markdown("""
### 競賽任務：
我們使用了 **IBM 真實員工數據集**。請各組利用此分析工具，找出 **「導致員工離職的 3 大關鍵元兇」**，並據此提出改善策略。

**評分標準：**
1.  **數據洞察 (40%)**：是否正確解讀數據？(例如：發現加班對離職的影響)
2.  **商業策略 (40%)**：提出的解決方案是否可行？(例如：針對加班者提供補休或加班費調整)
3.  **預測準度 (20%)**：利用 AI 模型預測誰會離職的準確率。
""")

# ==========================================
# 1. 資料上傳區
# ==========================================
st.sidebar.header("📂 步驟 1：上傳資料集")
uploaded_file = st.sidebar.file_uploader("請上傳 IBM-HR-Employee-Attrition.csv", type=["csv"])

# 預設載入範例資料 (如果老師還沒下載，先產生假資料以免報錯)
@st.cache_data
def load_sample_data():
    # 這裡只是為了演示，實際上請學生上傳 Kaggle 下載的 csv
    return pd.DataFrame() 

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("資料載入成功！")
else:
    st.info("👆 請從側邊欄上傳 Kaggle 的 IBM HR csv 檔案。")
    st.stop()

# ==========================================
# 2. 數據概覽 (Data Overview)
# ==========================================
with st.expander("🔍 點擊檢視原始資料 (Raw Data)", expanded=False):
    st.dataframe(df.head(10))
    st.write(f"總筆數：{df.shape[0]} 位員工 | 欄位數：{df.shape[1]}")

# ==========================================
# 3. 自動化關聯分析 (Correlation Analysis)
# ==========================================
st.header("1. 離職原因探索 (Exploratory Data Analysis)")
st.write("系統自動分析各變數與 **Attrition (離職)** 的關係。")

# 將 Attrition 轉換為數字 (Yes=1, No=0) 以便計算
if 'Attrition' in df.columns:
    df['Attrition_Num'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # 選擇要分析的因子
    factors = st.multiselect("請選擇你們懷疑的影響因子：", 
                             ['Age', 'DailyRate', 'DistanceFromHome', 'EnvironmentSatisfaction', 
                              'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 
                              'MonthlyIncome', 'NumCompaniesWorked', 'OverTime', 
                              'PercentSalaryHike', 'TotalWorkingYears', 'WorkLifeBalance', 
                              'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion'],
                             default=['MonthlyIncome', 'Age', 'DistanceFromHome', 'JobSatisfaction'])
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 視覺化：離職 vs 因子
        target_factor = st.selectbox("詳細觀察哪一個因子？", factors)
        
        # 如果是數值型 (如薪水)
        if df[target_factor].dtype != 'object':
            fig = px.box(df, x="Attrition", y=target_factor, color="Attrition", 
                         title=f"離職者與在職者的 {target_factor} 差異分析",
                         points="all")
            st.plotly_chart(fig, use_container_width=True)
            
            # 統計檢定提示
            avg_yes = df[df['Attrition']=='Yes'][target_factor].mean()
            avg_no = df[df['Attrition']=='No'][target_factor].mean()
            diff_pct = ((avg_yes - avg_no) / avg_no) * 100
            
            st.info(f"💡 數據洞察：離職者的平均 **{target_factor}** 為 {avg_yes:.1f}，比在職者 ({avg_no:.1f}) 差異約 **{diff_pct:.1f}%**。")
            
        else:
            # 如果是類別型 (如 OverTime)
            fig = px.histogram(df, x=target_factor, color="Attrition", barmode="group",
                               title=f"{target_factor} 分佈對離職的影響")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🔥 相關性熱圖")
        st.write("顏色越紅，代表與「離職」相關性越強 (正相關)；越藍代表越能「留任」 (負相關)。")
        
        # 計算相關係數
        # 處理 OverTime 這種文字欄位
        df_corr = df.copy()
        if 'OverTime' in df_corr.columns:
            df_corr['OverTime'] = df_corr['OverTime'].apply(lambda x: 1 if x == 'Yes' else 0)
            
        corr_cols = factors + ['Attrition_Num']
        # 只取存在的欄位
        valid_cols = [c for c in corr_cols if c in df_corr.columns]
        
        corr_matrix = df_corr[valid_cols].corr()[['Attrition_Num']].sort_values(by='Attrition_Num', ascending=False)
        
        fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', aspect="auto")
        st.plotly_chart(fig_corr, use_container_width=True)

# ==========================================
# 4. AI 離職預測模型 (Machine Learning)
# ==========================================
st.divider()
st.header("2. AI 預測模型競賽")
st.write("訓練一個機器學習模型，預測誰會離職。請調整參數以獲得最高準確率。")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

col_model_1, col_model_2 = st.columns(2)

with col_model_1:
    st.subheader("⚙️ 模型參數設定")
    n_estimators = st.slider("決策樹數量 (Trees)", 10, 200, 100)
    max_depth = st.slider("樹的深度 (Max Depth)", 1, 20, 10)
    test_size = st.slider("測試集比例 (Test Size)", 0.1, 0.5, 0.2)
    
    # 特徵工程：將類別轉數字
    df_ml = pd.get_dummies(df.drop(['Attrition', 'EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis=1, errors='ignore'), drop_first=True)
    
    # 執行訓練
    if st.button("🚀 訓練模型並預測"):
        X = df_ml.drop('Attrition_Num', axis=1, errors='ignore')
        y = df_ml['Attrition_Num']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred) # 抓出離職者的能力
        
        st.session_state['model_result'] = {'acc': acc, 'recall': recall, 'model': model, 'features': X.columns}

with col_model_2:
    if 'model_result' in st.session_state:
        res = st.session_state['model_result']
        st.subheader("🏆 模型成績單")
        st.metric("準確率 (Accuracy)", f"{res['acc']*100:.1f}%", help="整體預測對的機率")
        st.metric("召回率 (Recall)", f"{res['recall']*100:.1f}%", help="真正想離職的人，你抓出了多少？(這對HR最重要)")
        
        if res['recall'] < 0.3:
            st.error("⚠️ 警告：你的模型雖然準確率高，但幾乎抓不到離職者 (Recall 低)！這在 HR 領域是不及格的。請嘗試調整參數或處理資料不平衡。")
        else:
            st.success("✅ 模型表現不錯！能夠有效識別潛在離職風險。")
            
        # 顯示特徵重要性
        feat_importances = pd.Series(res['model'].feature_importances_, index=res['features'])
        st.write("**對離職影響最大的前 5 個特徵：**")
        st.bar_chart(feat_importances.nlargest(5))

# ==========================================
# 5. 商業策略提案 (Business Case)
# ==========================================
st.divider()
st.header("3. 策略提案 (請填寫)")
st.write("數據不會告訴你怎麼做，**人**才會。請根據上述分析，寫下各組的策略。")

st.text_area("Q1: 根據熱圖與模型，哪三個因素是導致離職的主因？", placeholder="例如：1. 加班 (OverTime)  2. 月薪 (MonthlyIncome) ...")
st.text_area("Q2: 針對這些主因，你們組建議公司採取什麼具體行動？", placeholder="例如：針對加班超過 10 小時的員工，強制實施週五無會議日...")
