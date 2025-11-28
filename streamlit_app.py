import streamlit as st
import pandas as pd
import numpy as np

# 設定網頁標題
st.set_page_config(page_title="HR 策略選才模擬器", layout="wide")

st.title("🎯 策略性人力資源管理：Moneyball 選才模擬器")
st.markdown("""
### 專題說明
請扮演人資長，根據你們小組分配到的公司策略（創新、成本、或客戶導向），
調整左側的 **「選才權重」** 與 **「篩選門檻」**。
系統將從 1000 位候選人中，挑出最符合你們策略的前 5 名。
""")

# ==========================================
# 1. 系統後端：生成數據 (與之前邏輯相同)
# ==========================================
@st.cache_data
def generate_candidates(num_candidates=1000):
    np.random.seed(42)
    data = {
        'ID': range(1, num_candidates + 1),
        'Resume': np.random.randint(50, 100, num_candidates),       # 履歷分數
        'Interview': np.random.randint(50, 100, num_candidates),    # 面試官評分
        'Tech_Test': np.random.randint(0, 100, num_candidates),     # 技術測驗
        'Culture': np.random.randint(0, 100, num_candidates),       # 文化契合度
        'Comm': np.random.randint(0, 100, num_candidates),          # 溝通能力
        'Uni_Tier': np.random.choice([1, 2, 3], num_candidates, p=[0.2, 0.5, 0.3]) # 學校等級
    }
    df = pd.DataFrame(data)
    
    # 上帝視角：真實績效計算 (學生看不到)
    # 邏輯：技術與溝通最重要，面試分數關聯低
    df['True_Performance'] = (
        df['Tech_Test'] * 0.4 + 
        df['Comm'] * 0.3 + 
        df['Culture'] * 0.2 + 
        np.random.randint(-10, 10, num_candidates)
    )
    df.loc[df['Uni_Tier'] == 1, 'True_Performance'] += 5
    df.loc[df['Uni_Tier'] == 3, 'True_Performance'] -= 5
    
    # 正規化到 0-100
    df['True_Performance'] = ((df['True_Performance'] - df['True_Performance'].min()) / 
                              (df['True_Performance'].max() - df['True_Performance'].min())) * 100
    df['True_Performance'] = df['True_Performance'].round(1)
    
    return df

df = generate_candidates()

# ==========================================
# 2. 左側欄：學生操作區 (控制面板)
# ==========================================
st.sidebar.header("⚙️ 策略參數設定")

st.sidebar.subheader("1. 設定權重 (權重總和建議為 100%)")
w_resume = st.sidebar.slider("履歷分數 (Resume) 權重", 0.0, 1.0, 0.1, 0.05)
w_interview = st.sidebar.slider("面試官評分 (Interview) 權重", 0.0, 1.0, 0.4, 0.05)
w_tech = st.sidebar.slider("技術測驗 (Tech Test) 權重", 0.0, 1.0, 0.2, 0.05)
w_culture = st.sidebar.slider("文化契合 (Culture) 權重", 0.0, 1.0, 0.1, 0.05)
w_comm = st.sidebar.slider("溝通能力 (Comm) 權重", 0.0, 1.0, 0.2, 0.05)

total_weight = w_resume + w_interview + w_tech + w_culture + w_comm
st.sidebar.info(f"目前權重總和: {total_weight:.2f} (建議調整至 1.0)")

st.sidebar.subheader("2. 設定門檻 (Filters)")
min_tech = st.sidebar.number_input("技術分數最低門檻", 0, 100, 60)
min_comm = st.sidebar.number_input("溝通分數最低門檻", 0, 100, 0)

# 按鈕
run_btn = st.sidebar.button("🚀 執行演算法並招募人才", type="primary")

# ==========================================
# 3. 主畫面：顯示結果
# ==========================================

if run_btn:
    # --- 演算法邏輯 ---
    # 1. 門檻篩選
    filtered_df = df[(df['Tech_Test'] >= min_tech) & (df['Comm'] >= min_comm)].copy()
    
    if len(filtered_df) < 5:
        st.error(f"篩選條件太嚴格！只剩下 {len(filtered_df)} 人，不足以招募 5 人。請降低門檻。")
    else:
        # 2. 計算預測分數
        filtered_df['Predicted_Score'] = (
            filtered_df['Resume'] * w_resume +
            filtered_df['Interview'] * w_interview +
            filtered_df['Tech_Test'] * w_tech +
            filtered_df['Culture'] * w_culture +
            filtered_df['Comm'] * w_comm
        )
        
        # 3. 排序並取前 5
        top_picks = filtered_df.sort_values(by='Predicted_Score', ascending=False).head(5)
        
        # --- 顯示結果 ---
        st.subheader("📋 您的 AI 招募結果 (Top 5)")
        st.dataframe(
            top_picks[['ID', 'Predicted_Score', 'Resume', 'Interview', 'Tech_Test', 'Culture', 'Comm', 'True_Performance']],
            use_container_width=True,
            hide_index=True
        )
        
        # --- 績效分析 ---
        avg_perf = top_picks['True_Performance'].mean()
        
        # 計算理論最佳值 (上帝視角)
        best_possible = df.sort_values(by='True_Performance', ascending=False).head(5)['True_Performance'].mean()
        efficiency = (avg_perf / best_possible) * 100
        
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="錄取者平均真實績效 (事后驗證)", value=f"{avg_perf:.1f} 分")
        
        with col2:
            st.metric(label="策略效能 (ROI)", value=f"{efficiency:.1f} %", delta=f"{efficiency-100:.1f}% 與最佳解差距")
            
        # --- 老師的講評建議 (根據結果自動生成) ---
        st.warning("💡 **分析與反思：**")
        if efficiency > 95:
            st.write("太強了！你們的策略幾乎找到了全市場最優秀的人才！你們看重了哪些指標？")
        elif avg_perf < best_possible * 0.8:
            st.write("績效不如預期。可能原因：你們是否過度相信「面試官評分」或「履歷」，而忽略了更能預測績效的「測驗分數」？")
        else:
            st.write("表現不錯，但還有優化空間。試著調整權重，看看能不能更接近 100% 的最佳解。")
            
else:
    st.info("👈 請在左側調整參數，並點擊「執行演算法」開始模擬。")
