import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score

st.set_page_config(page_title="IBM HR 戰情室 (v10.1)", layout="wide")

# ==========================================
# 0. 核心數據處理 (嚴格檢查翻譯與數值)
# ==========================================
@st.cache_data
def load_and_process_data(file):
    try:
        df = pd.read_csv(file)
        
        # 1. 欄位名稱翻譯
        columns_map = {
            'Age': '年齡', 'Attrition': '離職', 'BusinessTravel': '商務差旅', 'DailyRate': '日薪',
            'Department': '部門', 'DistanceFromHome': '通勤距離', 'Education': '教育程度',
            'EducationField': '教育領域', 'EmployeeCount': '員工數量', 'EmployeeNumber': '員工編號',
            'EnvironmentSatisfaction': '環境滿意度', 'Gender': '性別', 'HourlyRate': '時薪',
            'JobInvolvement': '工作投入度', 'JobLevel': '職級', 'JobRole': '職位',
            'JobSatisfaction': '工作滿意度', 'MaritalStatus': '婚姻狀況', 'MonthlyIncome': '月收入',
            'MonthlyRate': '月費率', 'NumCompaniesWorked': '曾工作公司數量', 'Over18': '年滿18歲',
            'OverTime': '加班', 'PercentSalaryHike': '加薪百分比', 'PerformanceRating': '績效評級',
            'RelationshipSatisfaction': '人際關係滿意度', 'StandardHours': '標準工時',
            'StockOptionLevel': '股票期權級別', 'TotalWorkingYears': '年資',
            'TrainingTimesLastYear': '去年培訓次數', 'WorkLifeBalance': '工作生活平衡',
            'YearsAtCompany': '公司年資', 'YearsInCurrentRole': '目前職位年資',
            'YearsSinceLastPromotion': '距離上次晉升年資', 'YearsWithCurrManager': '與目前經理共事年資'
        }

        # 2. 內容翻譯 (確保顯示為 已離職/留任)
        values_map = {
            'Attrition': {'Yes': '已離職', 'No': '留任'},
            'OverTime': {'Yes': '有', 'No': '無'},
            'Gender': {'Female': '女性', 'Male': '男性'},
            'MaritalStatus': {'Single': '單身', 'Married': '已婚', 'Divorced': '離婚'}
        }

        for col, trans_dict in values_map.items():
            if col in df.columns:
                df[col] = df[col].replace(trans_dict)

        df.rename(columns=columns_map, inplace=True)
        
        # 3. 數值化處理 (用於計算)
        if '離職' in df.columns:
            df['離職_數值'] = df['離職'].apply(lambda x: 1 if x == '已離職' else 0)

        # 4. 自動薪資轉換 (USD -> TWD, x30)
        salary_cols = ['月收入', '日薪', '時薪', '月費率']
        for col in salary_cols:
            if col in df.columns:
                df[col] = df[col] * 30
                
        return df
    except Exception as e:
        return pd.DataFrame()

# ==========================================
# 1. 系統初始化
# ==========================================
st.title("🎰 IBM HR 戰情室 (v10.1 修正版)")
st.markdown("本系統已將圖表統一為 **直向顯示**，並標示詳細人數與自動結論。")

uploaded_file = st.sidebar.file_uploader("📂 老師請上傳 CSV", type=["csv"])
if uploaded_file is not None:
    df = load_and_process_data(uploaded_file)
    st.sidebar.success("✅ 資料載入成功")
else:
    st.info("請先上傳資料檔案 (WA_Fn-UseC_-HR-Employee-Attrition.csv)")
    st.stop()

tab1, tab2 = st.tabs(["📊 數據分析教學", "🎡 分組留才大賭桌"])

# ==========================================
# 分頁 1: 數據分析 (EDA)
# ==========================================
with tab1:
    st.header("1. 離職原因探索 (EDA)")
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    ordinal_cols = ['工作滿意度', '環境滿意度', '人際關係滿意度', '工作投入度', '績效評級', '職級']
    categorical_cols = ['加班', '商務差旅', '部門', '性別', '婚姻狀況', '教育領域', '職位'] + ordinal_cols
    
    valid_options = [c for c in (numeric_cols + categorical_cols) if c in df.columns]
    if '離職_數值' in valid_options: valid_options.remove('離職_數值')
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("變數關聯分析 (可多選)")
        
        default_opts = [c for c in ['加班', '月收入'] if c in df.columns]
        selected_factors = st.multiselect("請勾選分析因子：", valid_options, default=default_opts)
        
        for target_factor in selected_factors:
            st.markdown(f"#### 📌 分析項目：{target_factor}")
            
            # 判斷是否為金額
            is_money = target_factor in ['月收入', '日薪', '時薪']
            money_prefix = "NT$ " if is_money else ""
            
            is_categorical = (target_factor in categorical_cols) or \
                             (df[target_factor].dtype == 'object') or \
                             (df[target_factor].nunique() <= 5)
            
            if is_categorical:
                # === A. 類別型分析 (直向長條圖) ===
                
                # 計算數據
                group_data = df.groupby(target_factor)['離職_數值'].agg(['mean', 'sum', 'count']).reset_index()
                group_data.columns = [target_factor, '離職率', '離職人數', '總人數']
                group_data['離職率%'] = (group_data['離職率'] * 100).round(1)
                
                # 建立標籤：30.5% (127/416人)
                group_data['顯示標籤'] = group_data.apply(
                    lambda x: f"{x['離職率%']}%<br>({int(x['離職人數'])}/{int(x['總人數'])}人)", axis=1
                )
                
                # 繪圖 (直向: x=類別, y=數值)
                max_val = group_data['離職率%'].max()
                fig = px.bar(group_data, x=target_factor, y='離職率%', 
                             text='顯示標籤',
                             title=f"【{target_factor}】離職率分析",
                             color='離職率%', color_continuous_scale='Reds')
                
                # 設定文字在柱子上方，並增加 Y 軸高度避免被切掉
                fig.update_traces(textposition='outside', textfont_size=14)
                fig.update_layout(yaxis=dict(range=[0, max_val * 1.35])) 
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 顯示表格
                with st.expander(f"📋 查看【{target_factor}】詳細數據表", expanded=True):
                    st.dataframe(group_data[[target_factor, '總人數', '離職人數', '離職率%']], hide_index=True, use_container_width=True)
                
                # 自動下結論
                max_row = group_data.loc[group_data['離職率%'].idxmax()]
                min_row = group_data.loc[group_data['離職率%'].idxmin()]
                gap = max_row['離職率%'] - min_row['離職率%']
                
                if gap > 10:
                    st.error(f"🔍 **結論：有顯著相關！**\n\n 「{max_row[target_factor]}」 的離職率 ({max_row['離職率%']}%) 遠高於 「{min_row[target_factor]}」 ({min_row['離職率%']}%)。")
                elif gap > 5:
                    st.warning(f"🔍 **結論：有輕微相關。**")
                else:
                    st.info(f"🔍 **結論：無顯著相關。**")

            else:
                # === B. 數值型分析 (直向盒鬚圖) ===
                
                # 繪圖 (直向: x=離職狀態, y=數值)
                fig = px.box(df, x="離職", y=target_factor, color="離職", 
                             title=f"【{target_factor}】分佈比較",
                             color_discrete_map={'已離職':'#FF4B4B', '留任':'#1F77B4'})
                st.plotly_chart(fig, use_container_width=True)
                
                # 計算與顯示表格
                stat_df = df.groupby('離職')[target_factor].agg(['mean', 'median', 'count']).reset_index()
                stat_df.columns = ['狀態', '平均值', '中位數', '人數']
                
                with st.expander(f"📋 查看【{target_factor}】詳細數據表", expanded=True):
                    # 格式化顯示 (不影響計算)
                    show_df = stat_df.copy()
                    if is_money:
                        show_df['平均值'] = show_df['平均值'].apply(lambda x: f"NT$ {x:,.0f}")
                        show_df['中位數'] = show_df['中位數'].apply(lambda x: f"NT$ {x:,.0f}")
                    else:
                        show_df['平均值'] = show_df['平均值'].apply(lambda x: f"{x:.1f}")
                    st.dataframe(show_df, hide_index=True, use_container_width=True)

                # 自動下結論
                try:
                    val_yes = stat_df[stat_df['狀態']=='已離職']['平均值'].values[0]
                    val_no = stat_df[stat_df['狀態']=='留任']['平均值'].values[0]
                    if val_no == 0: val_no = 1
                    diff_pct = ((val_yes - val_no) / val_no) * 100
                    
                    if diff_pct < -10:
                        st.error(f"🔍 **結論：有顯著負相關！(離職者較低)**\n\n 已離職者的平均 {target_factor} ({money_prefix}{val_yes:,.0f}) 比留任者低了 {abs(diff_pct):.1f}%。")
                    elif diff_pct > 10:
                        st.error(f"🔍 **結論：有顯著正相關！(離職者較高)**\n\n 已離職者的平均 {target_factor} ({money_prefix}{val_yes:,.0f}) 比留任者高了 {diff_pct:.1f}%。")
                    else:
                        st.info(f"🔍 **結論：無顯著差異 ({diff_pct:.1f}%)。**")
                except:
                    st.info("數據不足，無法計算差異。")
            
            st.divider()

    with col2:
        st.subheader("相關性熱圖")
        corr_cols = ['離職_數值', '月收入', '年齡', '年資', '通勤距離', '工作滿意度']
        real_corr_cols = [c for c in corr_cols if c in df.columns]
        
        if len(real_corr_cols) > 1:
            corr_matrix = df[real_corr_cols].corr()[['離職_數值']].sort_values(by='離職_數值', ascending=False)
            fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', aspect="auto")
            st.plotly_chart(fig_corr, use_container_width=True)

# ==========================================
# 分頁 2: 綜藝大賭桌 (維持功能)
# ==========================================
with tab2:
    st.header("🎡 HR 留才大賭桌")
    st.markdown("### 規則：\n1. 系統發出 5 張員工牌。\n2. 六個小組同時下注，勾選要挽留的人。\n3. 轉動幸運輪盤，被選中的組別 **本局分數加倍**！")

    if 'scores' not in st.session_state:
        st.session_state['scores'] = {f"第{i}組": 0 for i in range(1, 7)}
    if 'round_data' not in st.session_state:
        st.session_state['round_data'] = None
    if 'lucky_team' not in st.session_state:
        st.session_state['lucky_team'] = None

    c_ctrl_1, c_ctrl_2, c_ctrl_3 = st.columns([1, 1, 3])
    with c_ctrl_1:
        if st.button("🎲 重新發牌 (New Round)", type="primary"):
            st.session_state['round_data'] = df.sample(5).reset_index(drop=True)
            st.session_state['lucky_team'] = None
            st.rerun()
    with c_ctrl_2:
        if st.button("🧹 重置分數"):
            st.session_state['scores'] = {f"第{i}組": 0 for i in range(1, 7)}
            st.session_state['lucky_team'] = None

    if st.session_state['round_data'] is not None:
        round_df = st.session_state['round_data']
        
        st.divider()
        st.subheader("🧐 員工機密檔案")
        cols = st.columns(5)
        for i, row in round_df.iterrows():
            with cols[i]:
                st.info(f"員工 #{i+1}")
                st.write(f"**月薪**: 💰 NT$ {row.get('月收入', 0):,.0f}")
                ot = row.get('加班', '無')
                if ot == '有': st.error(f"加班: {ot}")
                else: st.success(f"加班: {ot}")
                st.write(f"**滿意度**: {row.get('工作滿意度', 0)}")
                st.write(f"**年資**: {row.get('年資', 0)} 年")

        st.divider()
        st.subheader("📝 各組決策看板")
        
        h1, h2, h3, h4, h5, h6 = st.columns([1.5, 1, 1, 1, 1, 1])
        h1.markdown("**組別**")
        h2.markdown("#1")
        h3.markdown("#2")
        h4.markdown("#3")
        h5.markdown("#4")
        h6.markdown("#5")
        
        team_picks = {}
        for team_name in st.session_state['scores'].keys():
            r1, r2, r3, r4, r5, r6 = st.columns([1.5, 1, 1, 1, 1, 1])
            r1.markdown(f"### 🚩 {team_name}")
            p1 = r2.checkbox("", key=f"{team_name}_1")
            p2 = r3.checkbox("", key=f"{team_name}_2")
            p3 = r4.checkbox("", key=f"{team_name}_3")
            p4 = r5.checkbox("", key=f"{team_name}_4")
            p5 = r6.checkbox("", key=f"{team_name}_5")
            team_picks[team_name] = [p1, p2, p3, p4, p5]

        st.divider()

        col_spin, col_submit = st.columns([1, 2])
        with col_spin:
            st.write("#### 🎡 Lucky Time")
            if st.button("轉動幸運輪盤！"):
                placeholder = st.empty()
                teams = list(st.session_state['scores'].keys())
                for _ in range(15):
                    rand_team = np.random.choice(teams)
                    placeholder.markdown(f"### 🎰 {rand_team} ...")
                    time.sleep(0.1)
                lucky = np.random.choice(teams)
                st.session_state['lucky_team'] = lucky
                placeholder.markdown(f"### 🎉 幸運星：{lucky} (分數 x2)！")
            if st.session_state['lucky_team']:
                st.success(f"本局 **{st.session_state['lucky_team']}** 得分將加倍！")

        with col_submit:
            st.write("#### 🚀 結算時刻")
            if st.button("揭曉答案 & 計算總分", type="primary", use_container_width=True):
                st.write("### 📢 本局戰報")
                ans_cols = st.columns(5)
                answers = []
                for i, row in round_df.iterrows():
                    is_leaving = (row['離職'] == '已離職')
                    answers.append(is_leaving)
                    with ans_cols[i]:
                        if is_leaving: st.error(f"#{i+1} 想離職")
                        else: st.success(f"#{i+1} 很忠誠")

                for team, picks in team_picks.items():
                    round_score = 0
                    for i, picked in enumerate(picks):
                        actual_leaving = answers[i]
                        if picked: 
                            if actual_leaving: round_score += 10
                            else: round_score -= 5
                        else:
                            if actual_leaving: round_score -= 10
                            else: round_score += 5
                    
                    if team == st.session_state['lucky_team']:
                        round_score *= 2
                        
                    st.session_state['scores'][team] += round_score
                    luck_icon = "🍀" if team == st.session_state['lucky_team'] else ""
                    st.write(f"**{team}** {luck_icon}: 本局得 **{round_score}** 分")

        st.header("🏆 總積分排行榜")
        lb_df = pd.DataFrame(list(st.session_state['scores'].items()), columns=['組別', '分數'])
        lb_df = lb_df.sort_values(by='分數', ascending=False)
        fig_lb = px.bar(lb_df, y='組別', x='分數', text='分數', orientation='h', color='分數', color_continuous_scale='Greens')
        st.plotly_chart(fig_lb, use_container_width=True)

    else:
        st.info("請點擊左上角「🎲 重新發牌」開始第一回合！")
