# ==========================================
# 3. 自動化關聯分析 (增強版：顯示關鍵數字)
# ==========================================
st.header("1. 離職原因探索 (EDA)")
st.write("系統自動分析各變數與 **離職** 的關係。")

# 將離職轉回數字以便計算 (是=1, 否=0)
if '離職' in df.columns:
    df['離職_數值'] = df['離職'].apply(lambda x: 1 if x == '是' else 0)
    
    # 定義欄位類型
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    # 這些雖然是數字，但其實是類別 (1-4分)，用長條圖看比較清楚
    ordinal_cols = ['工作滿意度', '環境滿意度', '人際關係滿意度', '工作投入度', '績效評級', '職級']
    
    # 讓這些欄位也可以被當作類別分析
    categorical_cols = ['加班', '商務差旅', '部門', '性別', '婚姻狀況', '教育領域', '職位角色'] + ordinal_cols
    
    # 下拉選單
    factors = st.multiselect("請選擇你們懷疑的影響因子：", 
                             numeric_cols + categorical_cols,
                             default=['月收入', '年齡', '加班', '工作滿意度'])
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        target_factor = st.selectbox("詳細觀察哪一個因子？", factors)
        
        # 判斷分析模式：如果它在我們定義的「類別/等級清單」中，就用長條圖看離職率
        is_categorical = (target_factor in categorical_cols) or (df[target_factor].dtype == 'object')
        
        if is_categorical:
            # === 模式 A：類別分析 (看離職率 %) ===
            # 計算每一組的離職率
            group_data = df.groupby(target_factor)['離職_數值'].agg(['mean', 'count']).reset_index()
            group_data.columns = [target_factor, '離職率', '人數']
            group_data['離職率%'] = (group_data['離職率'] * 100).round(1)
            
            # 畫圖
            fig = px.bar(group_data, x=target_factor, y='離職率%', 
                         title=f"【{target_factor}】各組別的離職率分析",
                         text='離職率%', # 這行讓數字直接顯示在柱子上
                         color='離職率%', 
                         color_continuous_scale='Reds')
            fig.update_traces(texttemplate='%{text}%', textposition='outside') # 強制顯示 % 符號
            st.plotly_chart(fig, use_container_width=True)
            
            # 顯示洞察文字
            max_row = group_data.loc[group_data['離職率%'].idxmax()]
            min_row = group_data.loc[group_data['離職率%'].idxmin()]
            st.info(f"💡 數據洞察：**{max_row[target_factor]}** 的群體離職率最高 (達 {max_row['離職率%']}%)；而 **{min_row[target_factor]}** 的群體最穩定。")

        else:
            # === 模式 B：數值分析 (看平均數差異) ===
            # 畫盒鬚圖
            fig = px.box(df, x="離職", y=target_factor, color="離職", 
                         title=f"離職者與在職者的【{target_factor}】差異",
                         color_discrete_map={'是':'#FF4B4B', '否':'#1F77B4'})
            st.plotly_chart(fig, use_container_width=True)
            
            # === 關鍵修改：在這裡直接計算並顯示數字 ===
            avg_yes = df[df['離職']=='是'][target_factor].mean()
            avg_no = df[df['離職']=='否'][target_factor].mean()
            diff_pct = ((avg_yes - avg_no) / avg_no) * 100
            
            # 使用 st.metric 顯示大大的數字
            m1, m2, m3 = st.columns(3)
            m1.metric("離職者平均", f"{avg_yes:.1f}")
            m2.metric("在職者平均", f"{avg_no:.1f}")
            m3.metric("差異幅度", f"{diff_pct:+.1f}%", delta_color="inverse")
            
            st.caption(f"解讀：離職者的 {target_factor} 平均比在職者{'高' if diff_pct > 0 else '低'}了 {abs(diff_pct):.1f}%。")

    with col2:
        st.subheader("🔥 相關性熱圖")
        # 只取純數字欄位做熱圖
        corr_cols = [c for c in factors if c in numeric_cols] + ['離職_數值']
        # 去除重複
        corr_cols = list(set(corr_cols))
        
        if len(corr_cols) > 1:
            corr_matrix = df[corr_cols].corr()[['離職_數值']].sort_values(by='離職_數值', ascending=False)
            fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', aspect="auto")
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.write("請選擇更多數值型因子(如月收入、年齡)以顯示熱圖")
