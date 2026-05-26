import logging
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger("finance_app.analytics_service")

class AnalyticsService:
    """Computes finance metrics, balance states, and formats charts data."""

    def compute_summary_kpis(self, expenses: List[Dict[str, Any]], income: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute top-level key financial indicators."""
        total_exp = sum(e.get("amount", 0.0) for e in expenses)
        total_inc = sum(i.get("amount", 0.0) for i in income)
        balance = total_inc - total_exp
        
        # Savings rate = (Income - Expense) / Income * 100
        savings_rate = 0.0
        if total_inc > 0:
            savings_rate = round((balance / total_inc) * 100, 1)

        return {
            "total_expenses": round(total_exp, 2),
            "total_income": round(total_inc, 2),
            "net_savings": round(balance, 2),
            "savings_rate": max(0.0, savings_rate)
        }

    def compute_category_breakdown(self, expenses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate aggregate totals and percentage distribution per category."""
        if not expenses:
            return []
            
        df = pd.DataFrame(expenses)
        if "category" not in df.columns or "amount" not in df.columns:
            return []

        # Standardize category text
        df["category"] = df["category"].fillna("Other").str.strip().str.title()
        group = df.groupby("category")["amount"].sum().reset_index()
        total = group["amount"].sum()
        
        if total > 0:
            group["percentage"] = (group["amount"] / total * 100).round(1)
        else:
            group["percentage"] = 0.0
            
        # Sort descending by amount
        group = group.sort_values(by="amount", ascending=False)
        return group.to_dict(orient="records")

    def compute_monthly_comparisons(self, expenses: List[Dict[str, Any]], income: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate month-over-month differences for income and expenses."""
        if not expenses and not income:
            return {"exp_change_pct": 0.0, "inc_change_pct": 0.0, "message": "No transaction history."}

        df_exp = pd.DataFrame(expenses) if expenses else pd.DataFrame(columns=["date", "amount"])
        df_inc = pd.DataFrame(income) if income else pd.DataFrame(columns=["date", "amount"])

        # Local parsing helper
        def parse_month(d_str):
            try:
                return datetime.strptime(d_str, "%Y-%m-%d").strftime("%Y-%m")
            except Exception:
                return "Unknown"

        current_month = datetime.utcnow().strftime("%Y-%m")
        # Subtract ~30 days for last month
        last_month = (datetime.utcnow() - pd.Timedelta(days=30)).strftime("%Y-%m")

        # Expense calculations
        curr_exp_sum = 0.0
        last_exp_sum = 0.0
        if not df_exp.empty and "date" in df_exp.columns and "amount" in df_exp.columns:
            df_exp["month"] = df_exp["date"].apply(parse_month)
            curr_exp_sum = df_exp[df_exp["month"] == current_month]["amount"].sum()
            last_exp_sum = df_exp[df_exp["month"] == last_month]["amount"].sum()

        # Income calculations
        curr_inc_sum = 0.0
        last_inc_sum = 0.0
        if not df_inc.empty and "date" in df_inc.columns and "amount" in df_inc.columns:
            df_inc["month"] = df_inc["date"].apply(parse_month)
            curr_inc_sum = df_inc[df_inc["month"] == current_month]["amount"].sum()
            last_inc_sum = df_inc[df_inc["month"] == last_month]["amount"].sum()

        # Percentage updates
        exp_change_pct = 0.0
        if last_exp_sum > 0:
            exp_change_pct = round(((curr_exp_sum - last_exp_sum) / last_exp_sum) * 100, 1)

        inc_change_pct = 0.0
        if last_inc_sum > 0:
            inc_change_pct = round(((curr_inc_sum - last_inc_sum) / last_inc_sum) * 100, 1)

        return {
            "current_month_expenses": round(curr_exp_sum, 2),
            "last_month_expenses": round(last_exp_sum, 2),
            "exp_change_pct": exp_change_pct,
            
            "current_month_income": round(curr_inc_sum, 2),
            "last_month_income": round(last_inc_sum, 2),
            "inc_change_pct": inc_change_pct
        }

    def compute_daily_spending_trend(self, expenses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group expenses by date to produce chronologically ordered daily balance outputs."""
        if not expenses:
            return []
            
        df = pd.DataFrame(expenses)
        if "date" not in df.columns or "amount" not in df.columns:
            return []

        df["date"] = pd.to_datetime(df["date"])
        # Group and sort chronologically
        daily = df.groupby("date")["amount"].sum().reset_index()
        daily = daily.sort_values(by="date")
        daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")
        
        return daily.to_dict(orient="records")

    def get_highest_spending_merchants(self, expenses: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
        """Identify top merchant endpoints driving user expenditure."""
        if not expenses:
            return []
            
        df = pd.DataFrame(expenses)
        if "merchant" not in df.columns or "amount" not in df.columns:
            return []

        df["merchant"] = df["merchant"].fillna("Unknown").str.strip()
        merchants = df.groupby("merchant")["amount"].sum().reset_index()
        merchants = merchants.sort_values(by="amount", ascending=False).head(limit)
        
        return merchants.to_dict(orient="records")

# Singleton Instance
analytics_service = AnalyticsService()
