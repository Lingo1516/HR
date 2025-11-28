import pandas as pd
import numpy as np

# 假設的候選人資料（簡歷）
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Education': ['Bachelors', 'Masters', 'Bachelors', 'PhD', 'Masters'],
    'Experience': [3, 5, 2, 10, 4],
    'Skills': ['Python, SQL', 'Python, Java', 'SQL, Excel', 'C++, Python', 'Java, SQL'],
    'Job Applied': ['Data Scientist', 'Software Engineer', 'Data Analyst', 'Researcher', 'Software Engineer']
}

# 轉換成DataFrame
df = pd.DataFrame(data)

# 假設的招聘要求
requirements = {
    'Data Scientist': {
        'Education': 'Bachelors',
        'Min Experience': 3,
        'Skills': ['Python', 'SQL']
    },
    'Software Engineer': {
        'Education': 'Masters',
        'Min Experience': 4,
        'Skills': ['Python', 'Java']
    },
    'Data Analyst': {
        'Education': 'Bachelors',
        'Min Experience': 2,
        'Skills': ['SQL', 'Excel']
    },
    'Researcher': {
        'Education': 'PhD',
        'Min Experience': 5,
        'Skills': ['C++', 'Python']
    }
}

# 招聘篩選邏輯
def filter_candidates(job, df, requirements):
    filtered_candidates = []
    job_req = requirements[job]

    for index, row in df.iterrows():
        # 檢查學歷是否符合要求
        if row['Education'] != job_req['Education']:
            continue
        # 檢查經驗是否符合要求
        if row['Experience'] < job_req['Min Experience']:
            continue
        # 檢查技能是否符合要求
        required_skills = job_req['Skills']
        if not all(skill in row['Skills'] for skill in required_skills):
            continue
        filtered_candidates.append(row['Name'])

    return filtered_candidates

# 測試招聘過程
job_to_fill = 'Software Engineer'
qualified_candidates = filter_candidates(job_to_fill, df, requirements)

print(f"Qualified candidates for {job_to_fill}: {qualified_candidates}")
