import random
import streamlit as st

# --------- 遊戲狀態初始化（使用 session_state） ---------
def init_game():
    st.session_state.rounds = st.session_state.get("rounds", 5)
    st.session_state.current_round = 1
    st.session_state.employees = 50
    st.session_state.productivity = 1.0
    st.session_state.satisfaction = 0.6
    st.session_state.cash = 1_000_000
    st.session_state.revenue = 1_200_000
    st.session_state.salary_per_employee = 30_000
    st.session_state.history = []

# --------- 每回合計算邏輯（沿用前一版規則） ---------
def apply_decisions(hire, training_budget, raise_percent, layoff):
    employees = st.session_state.employees
    satisfaction = st.session_state.satisfaction
    productivity = st.session_state.productivity
    cash = st.session_state.cash
    salary_per_employee = st.session_state.salary_per_employee

    # 成本
    hiring_cost = max(hire, 0) * 20_000
    layoff_cost = max(layoff, 0) * 15_000
    raise_cost = employees * salary_per_employee * (raise_percent / 100)
    total_hr_cost = hiring_cost + layoff_cost + training_budget + raise_cost

    # 員工數更新
    employees = max(employees + hire - layoff, 0)

    # 訓練效果
    train_effect = min(training_budget / 300_000, 1.0)

    # 加薪效果
    if raise_percent > 0:
        raise_effect = min(raise_percent / 10, 0.5)
    else:
        raise_effect = -0.1

    # 裁員效果
    if layoff > 0:
        layoff_effect = - min(layoff / 50, 0.4)
    else:
        layoff_effect = 0

    delta_satisfaction = 0.3 * train_effect + raise_effect + layoff_effect
    external_shock = random.uniform(-0.05, 0.05)
    satisfaction = min(max(satisfaction + delta_satisfaction + external_shock, 0), 1)

    delta_productivity = 0.2 * train_effect + 0.3 * (satisfaction - 0.6)
    productivity = max(productivity + delta_productivity, 0.5)

    turnover_rate = max(0.05, 0.25 - 0.2 * satisfaction)
    turnover = int(employees * turnover_rate)
    employees = max(employees - turnover, 0)

    salary_cost = employees * salary_per_employee * (1 + raise_percent / 100)
    revenue_factor = random.uniform(18_000, 22_000)
    revenue = int(employees * productivity * revenue_factor)

    profit = revenue - salary_cost - total_hr_cost
    cash += profit

    # 更新回 session_state
    st.session_state.employees = employees
    st.session_state.satisfaction = satisfaction
    st.session_state.productivity = productivity
    st.session_state.cash = cash
    st.session_state.revenue = revenue

    record = {
        "round": st.session_state.current_round,
        "hire": hire,
        "training_budget": training_budget,
        "raise_percent": raise_percent,
        "layoff": layoff,
        "turnover": turnover,
        "employees": employees,
        "satisfaction": round(satisfaction, 3),
        "productivity": round(productivity, 3),
        "revenue": revenue,
        "salary_cost": int(salary_cost),
        "total_hr_cost": int(total_hr_cost),
        "profit": int(profit),
        "cash": int(cash),
    }
    st.session_state.history.append(record)

# --------- 最終分數 ---------
def final_score():
    if not st.session_state.history:
        return 0
    last = st.session_state.history[-1]
    score = last["cash"] / 1000 + last["satisfaction"] * 500 + last["employees"] * 5
    return int(score)

# ================= Streamlit 介面 =================
st.title("策略性人力資源模擬遊戲（Streamlit版）")

# 第一次進來先初始化
if "initialized" not in st.session_state:
    init_game()
    st.session_state.initialized = True

# 側邊欄：設定回合數與重置遊戲
with st.sidebar:
    st.header("遊戲設定")
    rounds_input = st.number_input("總回合數", min_value=1, max_value=10, value=st.session_state.rounds, step=1)
    if st.button("重新開始遊戲"):
        init_game()
        st.session_state.rounds = int(rounds_input)
        st.experimental_rerun()

st.write(f"目前為第 **{st.session_state.current_round} / {st.session_state.rounds}** 回合")

# 顯示目前公司狀態
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("員工人數", st.session_state.employees)
    st.metric("員工滿意度 (0~1)", f"{st.session_state.satisfaction:.2f}")
with col2:
    st.metric("平均生產力指數", f"{st.session_state.productivity:.2f}")
    st.metric("現金（預算）", f"{st.session_state.cash:,.0f}")
with col3:
    st.metric("上一回合營收", f"{st.session_state.revenue:,.0f}")

st.markdown("---")
st.subheader("本回合 HR 策略決策")

# 使用輸入元件收集決策
c1, c2 = st.columns(2)
with c1:
    hire = st.number_input("1) 新招募人數", min_value=0, max_value=500, value=0, step=1)
    training_budget = st.number_input("2) 訓練與發展預算", min_value=0, max_value=1_000_000, value=0, step=10_000)
with c2:
    raise_percent = st.number_input("3) 平均調薪百分比", min_value=-10.0, max_value=30.0, value=0.0, step=0.5)
    layoff = st.number_input("4) 裁員人數", min_value=0, max_value=500, value=0, step=1)

# 按下按鈕才進入下一回合
if st.button("提交本回合決策並計算結果"):
    if st.session_state.current_round > st.session_state.rounds:
        st.warning("遊戲已結束，請在側邊欄按『重新開始遊戲』。")
    else:
        apply_decisions(int(hire), int(training_budget), float(raise_percent), int(layoff))
        st.session_state.current_round += 1

# 顯示歷史結果
if st.session_state.history:
    st.markdown("### 歷史回合摘要")
    st.dataframe(st.session_state.history)

# 若遊戲結束，顯示總結與分數
if st.session_state.current_round > st.session_state.rounds:
    st.markdown("---")
    st.subheader("遊戲結束")
    st.write(f"最終綜合分數：**{final_score()}**（可用來比較各小組表現）")
