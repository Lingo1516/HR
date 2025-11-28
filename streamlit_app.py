import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="HR 利潤中心大戰 (Tournament)", layout="wide")

# ==========================================
# 1. 初始化模擬環境 (Market Setup)
# ==========================================
# 設定 100 位潛在候選人 (所有組別共用的市場)
@st.cache_data
def generate_market_talent():
    np.random.seed(2024) # 固定種子，確保公平
    n = 100
    data = pd.DataFrame({
        'ID': range(1, n + 1),
        'Ability': np.random.normal(70, 15, n).clip(40, 100), # 能力值 (影響產出)
        'Motivation': np.random.normal(70, 15, n).clip(40, 100), # 動機 (影響產出)
        'Market_Value': np.random.normal(60000, 15000, n).clip(35000, 120000) # 市場行情價
    })
    # 真實潛力 (True Potential) = 能力 x 動機
    data['Potential_Revenue'] = (data['Ability'] * data['Motivation']) * 20 # 預估幫公司賺的錢
    return data

market_data = generate_market_talent()

# ==========================================
# 2. 遊戲標題與規則
# ==========================================
st.title("🏆 HR 策略競賽：誰是全場最賺錢的 HR 團隊？")
st.markdown("""
### 競賽規則
各組皆為一家相同規模的新創公司，需從市場上招募 **20 位員工**。
獲勝標準只有一個：**年度淨利 (Net Profit)**。

$$ \text{年度淨利} = \text{員工總產出 (Revenue)} - \text{總薪資成本 (Cost)} - \text{離職罰款 (Turnover Cost)} $$

**你們需要制定三個策略參數：**
1.  **選才門檻 (Quality)**：你們只要前幾 % 的頂尖人才？(越高越難找，且通常越貴)
2.  **薪資定位 (Pay Strategy)**：你們給薪水是市場行情的多少倍？(給低省錢但會離職，給高留人但傷本)
3.  **績效獎金 (Incentive)**：你們願意撥出多少利潤當獎金？(能提升員工產出)
""")

st.divider()

# ==========================================
# 3. 策略輸入區 (六組大亂鬥)
# ==========================================
st.subheader("⚔️ 各組策略輸入面板")
st.info("請老師根據各組討論結果，輸入以下參數：")

# 建立 6 個 Columns 對應 6 組
cols = st.columns(6)
groups_input = {}

for i in range(6):
    group_name = f"第 {i+1} 組"
    with cols[i]:
        st.markdown(f"#### 🚩 {group_name}")
        
        # 策略 1: 選才標準 (百分位數)
        # 例如 80 代表只錄取市場上前 20% 強的人
        q_threshold = st.number_input(f"選才門檻 (PR值)", 50, 99, 70, key=f"q_{i}", help="數值越高，只錄取能力越強的人")
        
        # 策略 2: 薪資定位 (Compa-Ratio)
        # 1.0 = 符合市場行情, 1.2 = 高於市場 20%
        pay_ratio = st.number_input(f"薪資定位 (倍率)", 0.8, 1.5, 1.0, step=0.05, key=f"p_{i}", help="1.0 為市場均價。低於 1.0 容易離職。")
        
        # 策略 3: 激勵強度
        # 影響員工實際上會發揮多少潛力
        incentive = st.number_input(f"獎金強度 (1-10)", 1, 10, 5, key=f"i_{i}", help="越高員工越賣命，但成本越高")
        
        groups_input[group_name] = {
            "Threshold": q_threshold,
            "Pay_Ratio": pay_ratio,
            "Incentive": incentive
        }

start_battle = st.button("🚀 開始模擬對戰 (Run Simulation)", type="primary", use_container_width=True)

# ==========================================
# 4. 模擬運算核心 (Backend Logic)
# ==========================================
if start_battle:
    results = []
    
    st.divider()
    st.header("📊 戰況即時看板")
    
    for g_name, strategy in groups_input.items():
        # --- A. 招募階段 (Recruitment) ---
        # 根據門檻篩選人才
        # 計算綜合能力分數
        df = market_data.copy()
        df['Score'] = df['Ability'] * 0.6 + df['Motivation'] * 0.4
        
        # 找出該組要求的門檻分數 (例如 PR 80)
        cutoff = np.percentile(df['Score'], strategy['Threshold'])
        
        # 錄取符合條件的人 (取前 20 名)
        hired = df[df['Score'] >= cutoff].sort_values(by='Score', ascending=False).head(20)
        
        if len(hired) < 20:
            # 懲罰：如果門檻設太高導致招不滿，強迫補入平庸員工
            n_short = 20 - len(hired)
            fillers = df[~df['ID'].isin(hired['ID'])].sample(n_short)
            hired = pd.concat([hired, fillers])
            penalty_msg = " (招募不足, 系統強迫補人)"
        else:
            penalty_msg = ""

        # --- B. 薪酬與成本 (Compensation) ---
        # 實際給薪 = 市場價值 * 薪資定位策略
        hired['Actual_Salary'] = hired['Market_Value'] * strategy['Pay_Ratio']
        # 獎金成本 = 基礎薪資 * (獎金強度 * 0.02)
        bonus_cost_per_person = hired['Actual_Salary'] * (strategy['Incentive'] * 0.02)
        hired['Total_Cost'] = hired['Actual_Salary'] + bonus_cost_per_person
        
        total_salary_cost = hired['Total_Cost'].sum()
        
        # --- C. 績效產出 (Performance) ---
        # 實際產出 = 潛力 * 激勵係數
        # 薪資給得越高，激勵越高；獎金越高，激勵越高
        motivation_factor = (strategy['Pay_Ratio'] * 0.5) + (strategy['Incentive'] * 0.05)
        hired['Actual_Revenue'] = hired['Potential_Revenue'] * motivation_factor
        
        total_revenue = hired['Actual_Revenue'].sum()
        
        # --- D. 離職風險 (Turnover) ---
        # 離職機率：薪水越低、能力越高(外面搶著要)，離職率越高
        # 簡單公式：如果 (實際薪資 / 市場價值) < 1.0，風險大增
        hired['Retention_Prob'] = (hired['Actual_Salary'] / hired['Market_Value']) + (strategy['Incentive'] * 0.02)
        
        # 模擬離職 (骰子)
        leavers = 0
        for idx, row in hired.iterrows():
            # 生成一個隨機數，如果大於留任機率，就離職
            if np.random.random() > row['Retention_Prob']:
                leavers += 1
        
        # 離職成本罰款 (每走一個人，損失 30,000 重置成本)
        turnover_cost = leavers * 30000
        
        # --- E. 最終結算 ---
        net_profit = total_revenue - total_salary_cost - turnover_cost
        
        results.append({
            "Team": g_name,
            "Net Profit": int(net_profit),
            "Revenue": int(total_revenue),
            "Cost": int(total_salary_cost),
            "Turnover Cost": int(turnover_cost),
            "Leavers": leavers,
            "Strategy": f"PR{strategy['Threshold']} / x{strategy['Pay_Ratio']} / Lv{strategy['Incentive']}"
        })

    # ==========================================
    # 5. 結果視覺化 (Leaderboard)
    # ==========================================
    res_df = pd.DataFrame(results).sort_values(by="Net Profit", ascending=False).reset_index(drop=True)
    
    # 顯示冠軍
    winner = res_df.iloc[0]
    st.success(f"🎉 冠軍隊伍：**{winner['Team']}**！ 年度淨利：**${winner['Net Profit']:,}**")
    
    # 排行榜圖表
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.markdown("### 📈 各組利潤排行榜")
        chart = alt.Chart(res_df).mark_bar().encode(
            x=alt.X('Net Profit', axis=alt.Axis(title='年度淨利 ($)')),
            y=alt.Y('Team', sort='-x', axis=alt.Axis(title='組別')),
            color=alt.Color('Net Profit', scale=alt.Scale(scheme='greens'), legend=None),
            tooltip=['Team', 'Net Profit', 'Leavers', 'Strategy']
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)
        
    with c2:
        st.markdown("### 📋 詳細數據表")
        st.dataframe(res_df[['Team', 'Net Profit', 'Leavers', 'Strategy']], hide_index=True)

    # ==========================================
    # 6. AI 戰後講評 (Debrief)
    # ==========================================
    st.divider()
    st.subheader("🕵️ 顧問分析報告")
    
    # 分析每一組的死因或勝因
    for i, row in res_df.iterrows():
        team = row['Team']
        profit = row['Net Profit']
        leavers = row['Leavers']
        
        msg = f"**{team} (排名 {i+1})**："
        
        if profit < 0:
            if leavers > 5:
                msg += "❌ **嚴重虧損！** 主因是「離職率太高」。你們薪資給太低，導致人才流失，罰款吃掉了利潤。這叫「省小錢花大錢」。"
            else:
                msg += "❌ **嚴重虧損！** 主因是「人事成本過高」。你們薪水給太高，雖然沒人走，但員工產出的價值無法覆蓋薪水。這叫「被員工吃垮」。"
        else:
            if i == 0:
                msg += "✅ **完美平衡！** 你們找到了「薪資」與「績效」的最佳甜蜜點。既留得住人，成本又控制得當。"
            else:
                if leavers > 3:
                    msg += "⚠️ **還有進步空間。** 雖然賺錢，但離職人數稍多，增加了隱形成本。"
                else:
                    msg += "⚠️ **還有進步空間。** 團隊很穩定，但也許因為門檻設太低，員工產出爆發力不足。"
                    
        st.write(msg)
