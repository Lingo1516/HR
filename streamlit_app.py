import streamlit as st
import pandas as pd
import numpy as np

# 設定網頁配置
st.set_page_config(page_title="HR 策略選才競賽", layout="wide")

# ==========================================
# 1. 側邊欄：老師設定區 (上帝視角)
# ==========================================
st.sidebar.title("👨‍🏫 老師控制台 (God Mode)")
st.sidebar.markdown("這裡設定「什麼樣的人才才是真正好的」。學生看不到這裡的數值，他們必須從個案中去推敲。")

with st.sidebar.expander("🔐 點擊展開/隱藏 真實績效邏輯", expanded=False):
    st.markdown("### 設定「真實績效」權重 (True Performance Model)")
    st.info("請根據您的個案情境調整。例如：如果是業務職缺，溝通的真實權重應該很高。")
    
    # 老師設定權重 (這些是「效標」，決定了誰入職後表現好)
    true_w_tech = st.number_input("技術能力的真實貢獻度", 0.0, 1.0, 0.3, 0.1, key="t_tech")
    true_w_comm = st.number_input("溝通能力的真實貢獻度", 0.0, 1.0, 0.3, 0.1, key="t_comm")
    true_w_culture = st.number_input("文化契合的真實貢獻度", 0.0, 1.0, 0.2, 0.1, key="t_culture")
    true_w_luck = st.number_input("運氣/隨機因素 (誤差)", 0.0, 0.5, 0.1, 0.05, key="t_luck")
    
    st.markdown("---")
    st.write("**面試分數的效度設定：**")
    interview_validity = st.slider("面試官看人準嗎？(面試分數與真實績效的相關性)", 0.0, 1.0, 0.3)
    st.caption("0.0=面試純屬瞎猜, 1.0=面試官完全能看透真實能力")

# ==========================================
# 2. 系統後端：生成 1000 位候選人
# ==========================================
@st.cache_data
def generate_data(t_tech, t_comm, t_culture, t_luck, iv_validity):
    np.random.seed(999) # 固定種子，保證每組面對的候選人庫是一樣的
    n = 1000
    
    # 生成候選人的「真實能力」(這是隱藏屬性)
    # 假設這些是上帝賦予他們的天賦
    true_tech_ability = np.random.randint(40, 100, n)
    true_comm_ability = np.random.randint(40, 100, n)
    true_culture_fit = np.random.randint(40, 100, n)
    
    # 根據老師設定的公式，計算「真實入職後績效」
    true_perf = (
        true_tech_ability * t_tech +
        true_comm_ability * t_comm +
        true_culture_fit * t_culture +
        np.random.randint(-10, 10, n) * t_luck # 隨機誤差
    )
    
    # 生成「甄選指標」 (學生看得到的數據)
    # 1. 測驗分數：通常與真實能力高度相關，但有誤差
    test_tech = true_tech_ability + np.random.randint(-5, 5, n)
    test_comm = true_comm_ability + np.random.randint(-10, 10, n)
    test_culture = true_culture_fit + np.random.randint(-15, 15, n)
    
    # 2. 履歷分數：跟真實能力有相關，但較弱
    resume = (true_tech_ability * 0.3 + true_comm_ability * 0.3 + np.random.randint(0, 40, n))
    
    # 3. 面試分數：這取決於老師設定的「面試效度」
    # 如果效度高，面試分數就接近真實績效；如果效度低，就是隨機亂給
    noise = np.random.randint(40, 100, n)
    interview = (true_perf * iv_validity) + (noise * (1 - iv_validity))
    
    # 建立 DataFrame
    df = pd.DataFrame({
        'ID': range(1, n + 1),
        'Resume': resume.clip(0, 100).astype(int),
        'Interview': interview.clip(0, 100).astype(int),
        'Tech_Test': test_tech.clip(0, 100).astype(int),
        'Comm_Test': test_comm.clip(0, 100).astype(int),
        'Culture_Test': test_culture.clip(0, 100).astype(int),
        'True_Performance': true_perf # 這是最後的答案
    })
    
    # 正規化真實績效到 0-100
    df['True_Performance'] = ((df['True_Performance'] - df['True_Performance'].min()) / 
                              (df['True_Performance'].max() - df['True_Performance'].min())) * 100
    return df

# 生成資料
df = generate_data(true_w_tech, true_w_comm, true_w_culture, true_w_luck, interview_validity)

# ==========================================
# 3. 學生操作區 (主要介面)
# ==========================================

st.title("🏆 HR 策略選才競賽")
st.markdown(f"""
請各組根據個案策略，決定你們的**篩選演算法**。
目標：找出系統中 **真實績效 (True Performance)** 最高的 5 位人才。
""")

# 分組選擇
group_id = st.selectbox("📌 請選擇組別：", ["Group 1", "Group 2", "Group 3", "Group 4", "Group 5", "Group 6"])

st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(f"🛠️ {group_id} 的策略設定")
    st.write("請分配 100% 的權重給以下指標：")
    
    s_resume = st.slider("履歷分數 (Resume)", 0, 100, 10)
    s_interview = st.slider("面試分數 (Interview)", 0, 100, 40)
    s_tech = st.slider("技術測驗 (Tech Test)", 0, 100, 20)
    s_comm = st.slider("溝通測驗 (Comm Test)", 0, 100, 20)
    s_culture = st.slider("文化測驗 (Culture Test)", 0, 100, 10)
    
    total = s_resume + s_interview + s_tech + s_comm + s_culture
    if total != 100:
        st.error(f"目前總和：{total}%。請調整至 100% 才能送出！")
        run = False
    else:
        st.success(f"目前總和：{total}%。設定完成！")
        run = st.button(f"🚀 {group_id} 開始招募", type="primary")

with col2:
    if run:
        # 計算學生預測的分數
        df['Student_Score'] = (
            df['Resume'] * s_resume +
            df['Interview'] * s_interview +
            df['Tech_Test'] * s_tech +
            df['Comm_Test'] * s_comm +
            df['Culture_Test'] * s_culture
        ) / 100
        
        # 挑選前 5 名
        top_5 = df.sort_values(by='Student_Score', ascending=False).head(5)
        
        # 計算成績
        avg_perf = top_5['True_Performance'].mean()
        
        # 計算理論最佳值 (滿分)
        best_possible = df.sort_values(by='True_Performance', ascending=False).head(5)['True_Performance'].mean()
        
        score = (avg_perf / best_possible) * 100
        
        st.subheader("📊 招募結果")
        st.metric(label=f"{group_id} 的最終得分 (ROI)", value=f"{score:.1f} 分")
        
        st.write("你們錄取的 5 位候選人：")
        st.dataframe(top_5[['ID', 'Student_Score', 'True_Performance', 'Interview', 'Tech_Test', 'Comm_Test']], hide_index=True)
        
        if score > 90:
            st.balloons()
            st.success("太厲害了！你們的策略與公司需求的適配度極高！")
        elif score < 70:
            st.warning("分數偏低。原因可能是：你們看重的指標（例如面試或履歷），其實無法預測這個職位的真實績效。")

# ==========================================
# 4. 揭曉答案區 (教學用)
# ==========================================
st.divider()
with st.expander("🕵️ 老師專用：揭曉背後邏輯 (事後檢討用)"):
    st.write("### 為什麼分數是這樣？")
    st.write(f"老師設定的真實績效權重為：技術 {true_w_tech}, 溝通 {true_w_comm}, 文化 {true_w_culture}")
    st.write(f"而該組學生的權重為：技術 {s_tech/100}, 溝通 {s_comm/100}, 文化 {s_culture/100}")
    st.write("分數差異來自於：**學生的選擇策略** 是否 **對齊 (Align)** 了 **真實的職位需求**。")
