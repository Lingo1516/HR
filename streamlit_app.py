import random

class HRGame:
    def __init__(self, rounds=5, seed=None):
        self.rounds = rounds
        self.current_round = 1
        self.history = []
        if seed is not None:
            random.seed(seed)
        # 初始狀態（可以依需要調整）
        self.employees = 50          # 員工人數
        self.productivity = 1.0      # 每位員工生產力指數
        self.satisfaction = 0.6      # 員工滿意度 (0~1)
        self.cash = 1_000_000        # 公司現金（預算）
        self.revenue = 1_200_000     # 年營收
        self.salary_per_employee = 30_000

    def print_state(self):
        print(f"\n========== 第 {self.current_round} 回合 / 共 {self.rounds} 回合 ==========")
        print(f"員工人數：{self.employees}")
        print(f"平均生產力指數：{self.productivity:.2f}")
        print(f"員工滿意度：{self.satisfaction:.2f} (0~1)")
        print(f"現金（可用預算）：{self.cash:,.0f}")
        print(f"上一回合營收：{self.revenue:,.0f}")

    def get_decisions(self):
        print("\n請輸入本回合 HR 策略決策（直接輸入數字）：")
        hire = int(input("1) 本年度新招募人數（可為 0）："))
        training_budget = int(input("2) 訓練與發展預算（建議 0~300000）："))
        raise_percent = float(input("3) 平均調薪百分比（例如輸入 3 代表 3%）："))
        layoff = int(input("4) 裁員人數（沒有就輸入 0）："))
        return {
            "hire": hire,
            "training_budget": training_budget,
            "raise_percent": raise_percent,
            "layoff": layoff,
        }

    def apply_decisions(self, d):
        # 基本成本與人數變動
        # 招募成本：每人 20,000
        hiring_cost = max(d["hire"], 0) * 20_000
        # 裁員補償金：每人 15,000
        layoff_cost = max(d["layoff"], 0) * 15_000

        # 調薪成本：以現有人數 * 平均薪資 * 百分比 粗略估算
        raise_cost = self.employees * self.salary_per_employee * (d["raise_percent"] / 100)

        total_hr_cost = hiring_cost + layoff_cost + d["training_budget"] + raise_cost

        # 員工人數更新
        self.employees = max(self.employees + d["hire"] - d["layoff"], 0)

        # 滿意度與生產力變化（非常簡化的規則）
        # 訓練預算：提高生產力與滿意度，但有遞減效果
        train_effect = min(d["training_budget"] / 300_000, 1.0)  # 0~1

        # 調薪：提高滿意度，但過高會壓力大（成本高），稍微影響現金
        if d["raise_percent"] > 0:
            raise_effect = min(d["raise_percent"] / 10, 0.5)  # 最多 +0.5
        else:
            raise_effect = -0.1  # 完全不加薪，員工有點不爽

        # 裁員：滿意度下降，短期成本上升
        if d["layoff"] > 0:
            layoff_effect = - min(d["layoff"] / 50, 0.4)  # 最多 -0.4
        else:
            layoff_effect = 0

        # 綜合對滿意度的影響
        delta_satisfaction = 0.3 * train_effect + raise_effect + layoff_effect

        # 隨機外部環境影響（市場、景氣等）
        external_shock = random.uniform(-0.05, 0.05)

        # 更新滿意度（保持在 0~1）
        self.satisfaction = min(max(self.satisfaction + delta_satisfaction + external_shock, 0), 1)

        # 生產力受訓練與滿意度影響
        delta_productivity = 0.2 * train_effect + 0.3 * (self.satisfaction - 0.6)
        self.productivity = max(self.productivity + delta_productivity, 0.5)  # 不低於 0.5

        # 離職率：滿意度低時較高
        turnover_rate = max(0.05, 0.25 - 0.2 * self.satisfaction)  # 大約 5%~25%
        turnover = int(self.employees * turnover_rate)
        self.employees = max(self.employees - turnover, 0)

        # 計算人力成本與營收
        salary_cost = self.employees * self.salary_per_employee * (1 + d["raise_percent"] / 100)
        # 營收 = 員工數 * 生產力 * 一個係數（簡化）
        revenue_factor = random.uniform(18_000, 22_000)
        self.revenue = int(self.employees * self.productivity * revenue_factor)

        # 更新現金
        profit = self.revenue - salary_cost - total_hr_cost
        self.cash += profit

        # 記錄本回合結果
        record = {
            "round": self.current_round,
            "hire": d["hire"],
            "training_budget": d["training_budget"],
            "raise_percent": d["raise_percent"],
            "layoff": d["layoff"],
            "turnover": turnover,
            "employees": self.employees,
            "satisfaction": round(self.satisfaction, 3),
            "productivity": round(self.productivity, 3),
            "revenue": self.revenue,
            "salary_cost": int(salary_cost),
            "total_hr_cost": int(total_hr_cost),
            "profit": int(profit),
            "cash": int(self.cash),
        }
        self.history.append(record)

        print("\n--- 本回合結果摘要 ---")
        print(f"自然離職人數：{turnover}")
        print(f"本回合營收：{self.revenue:,.0f}")
        print(f"人事成本（含加薪後）：{salary_cost:,.0f}")
        print(f"HR 額外成本（招募+訓練+裁員+加薪）：{total_hr_cost:,.0f}")
        print(f"本回合盈餘（可能為負）：{profit:,.0f}")
        print(f"期末現金餘額：{self.cash:,.0f}")

    def final_score(self):
        # 綜合指標：最後現金 + 最後滿意度 * 權重 + 最後人數 * 權重
        if not self.history:
            return 0
        last = self.history[-1]
        score = last["cash"] / 1000 + last["satisfaction"] * 500 + last["employees"] * 5
        return int(score)

    def print_summary(self):
        print("\n========== 遊戲結束：總結 ==========")
        for r in self.history:
            print(
                f"第{r['round']}回合 | 員工:{r['employees']} | 滿意度:{r['satisfaction']} | "
                f"生產力:{r['productivity']} | 營收:{r['revenue']:,.0f} | 盈餘:{r['profit']:,.0f} | 現金:{r['cash']:,.0f}"
            )
        print("\n*** 最終綜合分數（可用來與其他小組比較）：", self.final_score(), "***")


def main():
    print("歡迎來到『策略性人力資源模擬遊戲』！")
    rounds = int(input("請輸入總回合數（建議 3~5）："))
    game = HRGame(rounds=rounds)

    while game.current_round <= game.rounds:
        game.print_state()
        decisions = game.get_decisions()
        game.apply_decisions(decisions)
        game.current_round += 1

    game.print_summary()


if __name__ == "__main__":
    main()
