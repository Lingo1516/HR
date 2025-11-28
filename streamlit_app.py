import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="全方位 HR 決策模擬系統", layout="wide")

# ==========================================
# 0. 初始化數據庫 (Global Data)
# ==========================================
if 'candidates' not in st.session_state:
    # 生成 20 位原始候選人資料
    np.random.seed(42)
    names = [f"Candidate_{i}" for i in range(1, 21)]
    roles = np.random.choice(['Engineer', 'Sales', 'Manager'], 20)
    
    data = {
        'ID': range(1, 21),
        'Name': names,
        'Role': roles,
        'Edu_Level': np.random.choice([1, 2, 3], 20, p=[0.2, 0.5, 0.3]), # 1:HighSchool, 2:Bach, 3:Master
        'Exp_Years': np.random.randint(0, 15, 20),
        'Hard_Skills': np.random.randint(40, 100, 20), # 硬實力
        'Soft_Skills': np.random.randint(40, 100, 20), # 軟實力
        'Teamwork_Score': np.random.randint(30, 90, 20), # 合作潛力
        'Stress_Tolerance': np.random.randint(1, 10, 20), # 抗壓性
        'Exp_Salary': np.random.randint(40, 120, 20) * 1000 # 期望薪資
    }
    st.session_state['candidates'] = pd.DataFrame(data)
    
if 'hired_employees' not in st.session_state:
    st.session_state['hired_employees'] = pd.DataFrame()

# ==========================================
# 介面導航：五大模組
# ==========================================
st.title("🏢 策略性 HRM 全流程決策模擬系統")
st.markdown("請依序完成以下五個關卡，經營您的小組公司。")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1️⃣ 招聘自動化", 
    "2️⃣ 績效評估", 
    "3️⃣ 薪資結構", 
    "4️⃣ 離職預測", 
    "5️⃣ 團隊分析"
])

# ==========================================
# Module 1: 招聘過程自動化 (Recruitment)
# ==========================================
with tab1:
    st.header("1. 招聘篩選自動化")
    st.markdown("設定篩選條件，從 20 位候選人中挑選 **員工**。")
    
    df_c = st.session_state['candidates']
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("設定篩選機器人")
        req_exp = st.slider("最低年資要求 (Years)", 0, 10, 2)
        req_hard = st.slider("硬實力門檻 (Hard Skills)", 0, 100, 60)
        req_soft = st.slider("軟實力門檻 (Soft Skills)", 0, 100, 50)
        
    with col2:
        st.subheader("錄取結果預覽")
        # 篩選邏輯
        filtered = df_c[
            (df_c['Exp_Years'] >= req_exp) & 
            (df_c['Hard_Skills'] >= req_hard) & 
            (df_c['Soft_Skills'] >= req_soft)
        ]
        st.write(f"符合條件人數：{len(filtered)} 人")
        
        if st.button("確認錄取這些人 (Hire)", key="hire_btn"):
            if len(filtered) < 5:
                st.error("錄取人數太少！公司無法運作，請降低標準至少錄取 5 人。")
            else:
                # 模擬入職後的真實工作數據 (為下一關做準備)
                filtered = filtered.copy()
                # 產生工作表現數據 (Manager Rating)
                filtered['Manager_Rating'] = np.random.randint(60, 100, len(filtered))
                # 產生實際產出 (KPI)
                filtered['KPI_Score'] = (filtered['Hard_Skills']*0.6 + filtered['Exp_Years']*2 + np.random.randint(-10, 10, len(filtered))).clip(0, 100)
                
                st.session_state['hired_employees'] = filtered
                st.success(f"已成功錄取 {len(filtered)} 位員工！請前往「績效評估」分頁。")

# ==========================================
# Module 2: 員工績效評估 (Performance)
# ==========================================
with tab2:
    st.header("2. 績效評估模型設計")
    
    employees = st.session_state['hired_employees']
    
    if employees.empty:
        st.warning("請先在第一關錄取員工！")
    else:
        st.markdown("員工已入職一年。請設計績效計算公式，決定誰是 High Performer。")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("設定績效權重")
            w_kpi = st.slider("客觀產出 (KPI) 權重 %", 0, 100, 70)
            w_rating = st.slider("主管評分 (Manager Rating) 權重 %", 0, 100, 30)
            
            if w_kpi + w_rating != 100:
                st.error("權重總和必須為 100%！")
            else:
                # 計算績效
                employees['Final_Perf'] = (employees['KPI_Score'] * w_kpi + employees['Manager_Rating'] * w_rating) / 100
                st.session_state['hired_employees'] = employees
                
                st.info("績效分數已計算完成！")
        
        with col2:
            st.subheader("績效排名 Top 5")
            if 'Final_Perf' in employees.columns:
                st.dataframe(employees[['Name', 'Role', 'KPI_Score', 'Manager_Rating', 'Final_Perf']].sort_values(by='Final_Perf', ascending=False).head(5))

# ==========================================
# Module 3: 薪資結構設計 (Compensation)
# ==========================================
with tab3:
    st.header("3. 薪資結構與獎金計算")
    
    employees = st.session_state['hired_employees']
    
    if 'Final_Perf' not in employees.columns:
        st.warning("請先完成「績效評估」！")
    else:
        st.markdown("請根據職位設定底薪，並根據績效設定獎金倍率。")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            base_eng = st.number_input("工程師 (Engineer) 底薪", 40000, 100000, 60000)
        with c2:
            base_sales = st.number_input("業務 (Sales) 底薪", 30000, 80000, 45000)
        with c3:
            base_mgr = st.number_input("經理 (Manager) 底薪", 50000, 150000, 80000)
            
        bonus_rate = st.slider("績效獎金倍率 (每1分績效 = 多少元獎金)", 0, 1000, 200)
        
        if st.button("計算發薪 (Calculate Payroll)"):
            def calc_salary(row):
                base = 0
                if row['Role'] == 'Engineer': base = base_eng
                elif row['Role'] == 'Sales': base = base_sales
                else: base = base_mgr
                
                bonus = row['Final_Perf'] * bonus_rate
                return base + bonus
            
            employees['Actual_Salary'] = employees.apply(calc_salary, axis=1)
            st.session_state['hired_employees'] = employees
            
            total_cost = employees['Actual_Salary'].sum()
            st.success(f"全公司薪資計算完成！總人事成本：${total_cost:,.0f}")
            st.dataframe(employees[['Name', 'Role', 'Final_Perf', 'Actual_Salary']])

# ==========================================
# Module 4: 員工離職預測 (Retention)
# ==========================================
with tab4:
    st.header("4. 離職風險預測模型")
    
    employees = st.session_state['hired_employees']
    
    if 'Actual_Salary' not in employees.columns:
        st.warning("請先完成「薪資計算」！")
    else:
        st.markdown("設定「離職警示規則」。請思考：什麼樣的人會想走？(薪水太少？壓力太大？)")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("定義高風險群")
            # 讓學生定義規則
            risk_salary_ratio = st.slider("薪資滿意度門檻 (實際薪資 / 期望薪資 < ?%)", 50, 150, 90)
            risk_stress = st.slider("抗壓低標 (Stress Tolerance < ?)", 1, 10, 4)
            
            st.write("---")
            run_pred = st.button("執行預測模型")
            
        with col2:
            if run_pred:
                # 離職邏輯：
                # 1. 薪資低於期望太多
                # 2. 抗壓低且工作難度高 (這裡簡化為抗壓低)
                # 3. 績效高但薪資低 (High Performer Risk)
                
                def predict_turnover(row):
                    is_risk = False
                    reason = []
                    
                    # 規則 1: 錢不夠
                    if row['Actual_Salary'] < (row['Exp_Salary'] * (risk_salary_ratio/100)):
                        is_risk = True
                        reason.append("錢給太少")
                        
                    # 規則 2: 抗壓低
                    if row['Stress_Tolerance'] < risk_stress:
                        is_risk = True
                        reason.append("抗壓不足")
                        
                    return "🔴 離職高風險" if is_risk else "🟢 穩定", ", ".join(reason)

                employees[['Risk_Status', 'Risk_Reason']] = employees.apply(
                    lambda x: pd.Series(predict_turnover(x)), axis=1
                )
                
                risk_count = employees[employees['Risk_Status'] == "🔴 離職高風險"].shape[0]
                turnover_rate = risk_count / len(employees) * 100
                
                st.metric("預測離職率", f"{turnover_rate:.1f}%")
                st.dataframe(employees[['Name', 'Actual_Salary', 'Exp_Salary', 'Risk_Status', 'Risk_Reason']])
                
                st.session_state['hired_employees'] = employees

# ==========================================
# Module 5: 團隊動態分析 (Team Dynamics)
# ==========================================
with tab5:
    st.header("5. 團隊合作與動態分析")
    
    employees = st.session_state['hired_employees']
    
    if 'Risk_Status' not in employees.columns:
        st.warning("請先完成前面所有步驟！")
    else:
        # 只分析留下來的人 (穩定者)
        stable_team = employees[employees['Risk_Status'] == "🟢 穩定"]
        
        st.markdown(f"針對預測**留任的 {len(stable_team)} 位員工**進行團隊分析。")
        
        if len(stable_team) < 2:
            st.error("留任人數過少，無法分析團隊合作！請回到上一關調整薪資或標準，留住更多人。")
        else:
            # 簡單的團隊分析邏輯
            avg_teamwork = stable_team['Teamwork_Score'].mean()
            diversity_score = stable_team['Hard_Skills'].std() # 技能差異越大，互補性越高
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("團隊合作平均分數", f"{avg_teamwork:.1f} / 100")
                if avg_teamwork > 75:
                    st.success("✅ 這是一個高凝聚力的團隊！")
                else:
                    st.warning("⚠️ 團隊合作性偏低，可能會有溝通成本。")
                    
            with col2:
                st.metric("技能互補性 (多樣性)", f"{diversity_score:.1f}")
                if diversity_score > 15:
                    st.success("✅ 技能分佈廣泛，適合解決複雜問題。")
                else:
                    st.info("ℹ️ 團隊技能同質性高，可能缺乏創新。")
            
            st.subheader("最終團隊名單")
            st.dataframe(stable_team[['Name', 'Role', 'Hard_Skills', 'Teamwork_Score', 'Final_Perf']])
            
            st.divider()
            st.success("恭喜您完成所有 HRM 決策流程！")
