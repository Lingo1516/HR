import streamlit as st
import pandas as pd

st.set_page_config(page_title="10人海選模擬戰 (Talent Selection)", layout="wide")

# ==========================================
# 1. 建立 10 位候選人資料庫 (含誘餌與地雷)
# ==========================================
def get_candidates():
    data = [
        # --- 原始 6 位最佳適配者 (Target Fits) ---
        {"ID": "A", "Name": "Alex (技術怪才)", "Hard_Skills": 98, "Soft_Skills": 20, "Experience": 85, "Innovation": 90, "Stability": 70, "Salary_Exp": 90, "Desc": "頂尖駭客級工程師，極度內向，回答問題不超過三個字。對程式碼有潔癖。"},
        {"ID": "B", "Name": "Bella (社交天后)", "Hard_Skills": 45, "Soft_Skills": 99, "Experience": 75, "Innovation": 60, "Stability": 60, "Salary_Exp": 80, "Desc": "業績女王，能跟任何人在一分鐘內變朋友。但文書報表常出錯，技術理解力低。"},
        {"ID": "C", "Name": "Charlie (守門員)", "Hard_Skills": 75, "Soft_Skills": 65, "Experience": 99, "Innovation": 10, "Stability": 99, "Salary_Exp": 75, "Desc": "20年資深行政，目前為止零失誤紀錄。非常保守，拒絕任何沒被驗證過的新流程。"},
        {"ID": "D", "Name": "Diana (潛力股)", "Hard_Skills": 65, "Soft_Skills": 85, "Experience": 5, "Innovation": 95, "Stability": 80, "Salary_Exp": 45, "Desc": "名校應屆畢業生，反應極快，學習力驚人，但是一張白紙，完全沒進過職場。"},
        {"ID": "E", "Name": "Ethan (連續創業者)", "Hard_Skills": 85, "Soft_Skills": 80, "Experience": 65, "Innovation": 99, "Stability": 20, "Salary_Exp": 85, "Desc": "鬼才型人物，點子多到爆炸。但履歷顯示過去三年換了五份工作，很容易無聊。"},
        {"ID": "F", "Name": "Fiona (完美菁英)", "Hard_Skills": 90, "Soft_Skills": 90, "Experience": 90, "Innovation": 80, "Stability": 85, "Salary_Exp": 120, "Desc": "外商高管出身，幾乎沒有短板的完美人才。唯一的缺點是：她非常、非常貴。"},
        
        # --- 新增 4 位干擾選項 (Distractors) ---
        {"ID": "G", "Name": "Gary (平庸大叔)", "Hard_Skills": 60, "Soft_Skills": 60, "Experience": 60, "Innovation": 40, "Stability": 60, "Salary_Exp": 60, "Desc": "什麼都會一點，但什麼都不精通。個性溫和，但在團隊中常被忽略，缺乏亮點。"},
        {"ID": "H", "Name": "Helen (跳槽女王)", "Hard_Skills": 95, "Soft_Skills": 95, "Experience": 80, "Innovation": 70, "Stability": 10, "Salary_Exp": 95, "Desc": "能力極強，面試表現完美。但注意看履歷：她平均每半年就跳槽一次，且都在試用期後離職。"},
        {"ID": "I", "Name": "Ivan (只想躺平)", "Hard_Skills": 50, "Soft_Skills": 40, "Experience": 30, "Innovation": 30, "Stability": 95, "Salary_Exp": 35, "Desc": "追求「錢多事少離家近」。雖然只要最低薪資，但面試時直言不願意加班，準時下班最重要。"},
        {"ID": "J", "Name": "Jack (空談夢想家)", "Hard_Skills": 30, "Soft_Skills": 90, "Experience": 20, "Innovation": 100, "Stability": 50, "Salary_Exp": 70, "Desc": "口才極佳，滿口區塊鏈與AI趨勢，但被問到具體執行細節時會顧左右而言他。"}
    ]
    return pd.DataFrame(data)

df_candidates = get_candidates()

# ==========================================
# 2. 介面設計：10人履歷牆
# ==========================================
st.title("🧩 10人海選模擬戰 (The Selection Challenge)")
st.markdown("""
### 📢 獵頭任務
市場上有 **10 位候選人**，包含頂尖人才、平庸者，以及隱藏的地雷。
你們 **6 個小組** 代表不同部門，請設定篩選機制，從中找出 **唯一** 最適合你們的那位。
*(注意：有 4 個人最終會無人錄取)*
""")

with st.expander("📂 點擊展開：10 位候選人詳細檔案 (Resumes)", expanded=True):
    # 用兩排顯示，比較整齊
    for i in range(0, 10, 2):
        c1, c2 = st.columns(2)
        row1 = df_candidates.iloc[i]
        row2 = df_candidates.iloc[i+1]
        
        with c1:
            st.info(f"🆔 **{row1['Name']}**")
            st.caption(f"硬實力: {row1['Hard_Skills']} | 軟實力: {row1['Soft_Skills']} | 薪資: {row1['Salary_Exp']}")
            st.write(f"📝 {row1['Desc']}")
            
        with c2:
            st.info(f"🆔 **{row2['Name']}**")
            st.caption(f"硬實力: {row2['Hard_Skills']} | 軟實力: {row2['Soft_Skills']} | 薪資: {row2['Salary_Exp']}")
            st.write(f"📝 {row2['Desc']}")

# ==========================================
# 3. 部門甄選設定
# ==========================================
st.divider()
st.header("⚙️ 制定甄選策略")

department = st.selectbox("請選擇你們代表的部門：", 
    ["Group 1: 研發中心 (R&D)", 
     "Group 2: 業務拓展部 (Sales)", 
     "Group 3: 財務行政部 (Admin)", 
     "Group 4: 儲備幹部計畫 (MA)", 
     "Group 5: 新事業創新小組 (Startup)", 
     "Group 6: 總經理室 (Executive Office)"])

st.subheader(f"設定 {department} 的篩選漏斗")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("#### 1. 門檻篩選 (Knockout)")
    st.caption("低於此標準者直接淘汰")
    min_hard = st.slider("硬實力門檻", 0, 100, 0)
    min_soft = st.slider("軟實力門檻", 0, 100, 0)
    min_exp = st.slider("經驗值門檻", 0, 100, 0)
    min_stab = st.slider("穩定度門檻", 0, 100, 0, help="過濾掉習慣性跳槽的人")
    max_salary = st.number_input("薪資預算上限", 0, 200, 100)

with col2:
    st.markdown("#### 2. 權重排序 (Ranking)")
    st.caption("總分權重分配 (總和須為 100)")
    w_hard = st.number_input("硬實力權重", 0, 100, 20)
    w_soft = st.number_input("軟實力權重", 0, 100, 20)
    w_exp = st.number_input("經驗權重", 0, 100, 20)
    w_inn = st.number_input("創新權重", 0, 100, 20)
    w_stab = st.number_input("穩定權重", 0, 100, 20)
    
    total_w = w_hard + w_soft + w_exp + w_inn + w_stab
    if total_w != 100:
        st.error(f"目前總和：{total_w}% (請調整至 100)")
        run_btn = False
    else:
        run_btn = st.button("🚀 執行篩選", type="primary")

# ==========================================
# 4. 運算與講評
# ==========================================
if run_btn:
    st.divider()
    st.subheader("📊 篩選結果報告")
    
    # 1. 門檻過濾
    passed = df_candidates[
        (df_candidates['Hard_Skills'] >= min_hard) &
        (df_candidates['Soft_Skills'] >= min_soft) &
        (df_candidates['Experience'] >= min_exp) &
        (df_candidates['Stability'] >= min_stab) &
        (df_candidates['Salary_Exp'] <= max_salary)
    ].copy()
    
    if len(passed) == 0:
        st.error("❌ 無人存活！您的門檻設定太高，或是薪資給太低，導致所有人都被篩掉了。")
    else:
        # 2. 計算得分
        passed['Final_Score'] = (
            passed['Hard_Skills'] * w_hard +
            passed['Soft_Skills'] * w_soft +
            passed['Experience'] * w_exp +
            passed['Innovation'] * w_inn +
            passed['Stability'] * w_stab
        ) / 100
        
        # 3. 排序
        ranking = passed.sort_values(by='Final_Score', ascending=False)
        top_pick = ranking.iloc[0]
        
        # 顯示前三名
        st.write(f"通過門檻人數：{len(passed)} 人。您的最佳人選是：")
        st.success(f"🏆 第一名：{top_pick['Name']} (得分: {top_pick['Final_Score']:.1f})")
        st.dataframe(ranking[['Name', 'Final_Score', 'Hard_Skills', 'Soft_Skills', 'Stability', 'Salary_Exp']], hide_index=True)

        # 4. 適配度驗證邏輯
        best_fits = {
            "Group 1: 研發中心 (R&D)": "A",
            "Group 2: 業務拓展部 (Sales)": "B",
            "Group 3: 財務行政部 (Admin)": "C",
            "Group 4: 儲備幹部計畫 (MA)": "D",
            "Group 5: 新事業創新小組 (Startup)": "E",
            "Group 6: 總經理室 (Executive Office)": "F"
        }
        
        target_id = best_fits[department]
        target_name = df_candidates[df_candidates['ID'] == target_id].iloc[0]['Name']
        
        st.markdown("---")
        st.subheader("🕵️ 顧問講評")
        
        if top_pick['ID'] == target_id:
            st.balloons()
            st.success(f"完美適配！{top_pick['Name']} 正是該職位的最佳人選。你們精準地抓住了核心需求。")
        elif top_pick['ID'] == 'H':
            st.error("⚠️ 危險決策！你們選到了 Helen (跳槽女王)。她能力雖然最強，但「穩定度」極低。你們的部門將在三個月後面臨人員流失，且浪費了昂貴的招募成本。")
        elif top_pick['ID'] == 'G':
            st.warning("⚠️ 平庸陷阱。你們選了 Gary。他雖然便宜且過門檻，但無法為部門帶來卓越績效。這通常是因為你們的「門檻設太低」或「權重沒重點」。")
        elif top_pick['ID'] == 'J':
            st.error("⚠️ 詐騙警報！Jack 是空談夢想家。你們可能被「創新」的權重迷惑，卻忽略了「硬實力」或「經驗」的驗證。")
        else:
            st.info(f"尚可接受，但不是最佳解。系統建議的最佳人選其實是：**{target_name}**。試著比較一下兩者的差異？")
