import streamlit as st
import pandas as pd
import numpy as np

# 設定頁面
st.set_page_config(page_title="循證選才模擬器 (Evidence-Based)", layout="wide")

# ==========================================
# 1. 知識庫與參數設定 (基於 Schmidt & Hunter 研究)
# ==========================================
# 真實效度係數 (Validity Coefficients) - 這是學術界的「真相」
# 參考資料: Schmidt, F. L., & Hunter, J. E. (1998). The validity and utility of selection methods.
VALIDITY_MAP = {
    "GMA": 0.51,        # 認知能力 (最高效度)
    "Structured": 0.51, # 結構化面試 (高效度)
    "Unstructured": 0.14, # 非結構化面試 (低效度陷阱)
    "Conscientiousness": 0.31, # 盡責性 (人格中預測力最高)
    "Peer_Review": 0.40, # 同儕評估/工作試樣
    "Reference": 0.26   # 推薦信
}

st.title("🧬 Evidence-Based HR：科學選才模擬實驗室")
st.markdown("""
### 專題背景
這不是運氣遊戲，這是基於 **Schmidt & Hunter (1998)** 統合分析的科學模擬。
你們的任務是為一家跨國企業挑選 **「儲備幹部 (MA)」**。
資料庫中有真實的測評數據，請決定你們要採信哪些工具來預測候選人的未來績效。
""")

# ==========================================
# 2. 側邊欄：老師設定 (God Mode)
# ==========================================
with st.sidebar.expander("🔐 老師專用設定 (控制真實權重)"):
    st.write("在此微調該職位的核心需求 (影響真實績效的公式)")
    
    # 預設值是基於一般管理職位 (MA)
    role_w_iq = st.slider("認知能力 (IQ) 的重要性", 0.0, 1.0, 0.5)
    role_w_personality = st.slider("人格特質 (盡責性) 的重要性", 0.0, 1.0, 0.3)
    role_w_social = st.slider("人際互動 (結構化面試) 的重要性", 0.0, 1.0, 0.4)
    
    st.info("提示：非結構化面試(憑感覺)的參數已被系統鎖定為低效度，用來測試學生是否會掉入陷阱。")

# ==========================================
# 3. 數據生成引擎
# ==========================================
@st.cache_data
def generate_real_candidates(n=1000, w_iq=0.5, w_pers=0.3, w_soc=0.4):
    np.random.seed(42)
    
    # 1. 生成潛在變項 (Latent Variables) - 這些是候選人真正的素質
    # 智力 (G)
    true_g = np.random.normal(0, 1, n) 
    # 盡責性 (Conscientiousness)
    true_c = np.random.normal(0, 1, n)
    # 社交技能 (Social Skill)
    true_s = np.random.normal(0, 1, n)
    
    # 2. 計算「真實工作績效」 (True Job Performance)
    # 這是我們最後要驗證的標準，基於老師設定的權重
    performance_score = (true_g * w_iq) + (true_c * w_pers) + (true_s * w_soc) + np.random.normal(0, 0.5, n)
    
    # 正規化績效到 0-100
    performance_score = ((performance_score - performance_score.min()) / 
                         (performance_score.max() - performance_score.min())) * 100
    
    # 3. 生成「測評工具數據」 (Observed Variables)
    # 模擬真實世界的測量誤差 (Measurement Error)
    
    # [工具 A] 認知能力測驗 (如 Wonderlic/SHL) - 信度高，與 G 高度相關
    test_gma = true_g * 0.9 + np.random.normal(0, 0.3, n)
    
    # [工具 B] 五大人格測驗 (如 Hogan) - 測量盡責性
    test_big5 = true_c * 0.8 + np.random.normal(0, 0.4, n)
    
    # [工具 C] 結構化面試 (Structured Interview) - 有效測量社交與職能
    test_structured = true_s * 0.7 + true_g * 0.3 + np.random.normal(0, 0.4, n)
    
    # [工具 D] 非結構化面試 (Unstructured Interview) - 這是陷阱！
    # 往往測到的是「外向性」或「面試官偏見」，跟真實績效關聯低
    bias = np.random.normal(0, 1, n) # 面試官隨機喜好
    test_unstructured = (true_s * 0.2) + (bias * 0.8) # 大部分是雜訊
    
    # [工具 E] 情境判斷測驗 (SJT) - 混合了智力與經驗
    test_sjt = true_g * 0.4 + true_c * 0.3 + true_s * 0.3 + np.random.normal(0, 0.5, n)

    # 轉成 DataFrame 並將分數轉為 T分數或常模 (50-100)
    df = pd.DataFrame({
        'Candidate_ID': range(1, n + 1),
        'GMA_Score': (test_gma * 10 + 70).clip(40, 99).astype(int), # 認知能力
        'Big5_Conscientiousness': (test_big5 * 10 + 70).clip(40, 99).astype(int), # 盡責性
        'Structured_Interview': (test_structured * 10 + 70).clip(40, 99).astype(int), # 結構化
        'Unstructured_Interview': (test_unstructured * 10 + 70).clip(40, 99).astype(int), # 非結構化 (陷阱)
        'SJT_Score': (test_sjt * 10 + 70).clip(40, 99).astype(int), # 情境判斷
        'True_Performance': performance_score
    })
    
    return df

# 載入資料
df = generate_real_candidates(1000, role_w_iq, role_w_personality, role_w_social)

# ==========================================
# 4. 學生操作區：決策儀表板
# ==========================================
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🛠️ 建立甄選模型")
    st.write("請決定各項工具在最終決策中的**權重 (%)**。")
    st.caption("提示：根據研究，並非所有工具都一樣有效。")
    
    w_gma = st.number_input("1. 認知能力測驗 (GMA, 如 Wonderlic)", 0, 100, 0, help="測量學習能力、邏輯推理。成本低，學術效度高。")
    w_big5 = st.number_input("2. 五大人格-盡責性 (Big5, 如 Hogan)", 0, 100, 0, help="測量責任感、成就動機。預測工作表現的穩定指標。")
    w_struct = st.number_input("3. 結構化面試 (Structured Interview)", 0, 100, 0, help="基於職能的標準化提問。成本高，效度高。")
    w_unstruct = st.number_input("4. 傳統非結構化面試 (Unstructured)", 0, 100, 50, help="憑面試官直覺的聊天。成本高，但效度...？")
    w_sjt = st.number_input("5. 情境判斷測驗 (SJT)", 0, 100, 0, help="模擬職場情境的紙筆測驗。")
    
    total_w = w_gma + w_big5 + w_struct + w_unstruct + w_sjt
    
    if total_w != 100:
        st.error(f"目前權重總和: {total_w}% (必須等於 100%)")
        can_run = False
    else:
        st.success("權重配置完成！")
        can_run = True
        
    st.divider()
    group_name = st.text_input("輸入組別名稱 (用於排行榜)", "Group A")
    run_btn = st.button("🚀 送出甄選策略", type="primary", disabled=not can_run)

with col2:
    if run_btn:
        # 計算學生模型的預測總分
        df['Selection_Score'] = (
            df['GMA_Score'] * w_gma +
            df['Big5_Conscientiousness'] * w_big5 +
            df['Structured_Interview'] * w_struct +
            df['Unstructured_Interview'] * w_unstruct +
            df['SJT_Score'] * w_sjt
        ) / 100
        
        # 選出前 5 名
        top_picks = df.sort_values(by='Selection_Score', ascending=False).head(5)
        
        # 計算結果
        avg_perf = top_picks['True_Performance'].mean()
        best_possible = df.sort_values(by='True_Performance', ascending=False).head(5)['True_Performance'].mean()
        efficiency = (avg_perf / best_possible) * 100
        
        st.subheader(f"📊 {group_name} 的甄選結果報告")
        
        # 顯示關鍵指標
        m1, m2, m3 = st.columns(3)
        m1.metric("選入者平均績效", f"{avg_perf:.1f}")
        m2.metric("策略效能 (ROI)", f"{efficiency:.1f}%")
        m3.metric("最佳可能績效", f"{best_possible:.1f}")
        
        st.write("📋 **錄取名單明細：**")
        st.dataframe(top_picks[['Candidate_ID', 'Selection_Score', 'True_Performance', 'GMA_Score', 'Unstructured_Interview', 'Structured_Interview']], hide_index=True)
        
        # === 關鍵教學點：相關係數分析 ===
        st.markdown("---")
        st.subheader("💡 AI 深度分析 (Debriefing)")
        st.write("讓我們來看看你們採用的指標，與真實績效的相關性 (Correlation)：")
        
        # 計算相關係數矩陣
        corr_data = df[['True_Performance', 'GMA_Score', 'Unstructured_Interview', 'Structured_Interview', 'Big5_Conscientiousness']].corr()
        perf_corr = corr_data['True_Performance'].drop('True_Performance')
        
        st.bar_chart(perf_corr)
        
        st.info("""
        **圖表解讀：**
        * 棒狀圖越高，代表該工具越能準確預測績效。
        * 請注意看 **GMA (認知能力)** 與 **Unstructured Interview (非結構化面試)** 的差距。
        * 你們是否過度依賴了「非結構化面試」？這就是許多企業選錯人的主因。
        """)
        
        if w_unstruct > 30:
            st.warning("⚠️ 警告：您的策略高度依賴「非結構化面試」。研究顯示，這種面試容易受到第一印象、月暈效應影響，預測力遠低於智力測驗或結構化面試。")
        elif w_gma > 30 and w_struct > 20:
            st.success("✅ 專家級策略！您結合了預測力最高的「認知能力」與「結構化面試」，這是目前科學上公認最佳的組合。")

    else:
        st.info("👈 請在左側輸入權重並開始模擬。")
        st.markdown("""
        #### 參考工具說明：
        * **GMA Score:** 類似 Wonderlic 人員測驗，測量一般智能。
        * **Big5 Conscientiousness:** 類似 Hogan HPI 中的「審慎性」，測量自律與條理。
        * **Structured Interview:** 類似 DDI 的行為面試法 (STAR)，有固定評分標準。
        * **Unstructured Interview:** 傳統的「聊聊看」，容易受面試官主觀喜好影響。
        """)
