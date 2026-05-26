import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta

logger = logging.getLogger("finance_app.fraud_agent")

class FraudAgent:
    """Agent running rule-based and statistics-driven double-charge and anomaly checks."""
    
    def scan_for_anomalies(self, expenses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify potential duplicate bills, spikes in spending, or abnormal transaction sizes."""
        logger.info("FraudAgent: Running financial sanity scanning...")
        anomalies = []
        if not expenses:
            return anomalies

        # --- 1. Scan for Double Billing ---
        # Same amount, merchant, and close dates (within 2 days)
        seen_transactions = []
        for e in expenses:
            e_id = e.get("_id") or e.get("id")
            amount = e.get("amount", 0.0)
            merchant = e.get("merchant", "").strip().lower()
            date_str = e.get("date", "")
            
            try:
                e_date = datetime.strptime(date_str, "%Y-%m-%d")
            except Exception:
                continue

            for prev in seen_transactions:
                prev_id = prev.get("_id") or prev.get("id")
                prev_amount = prev.get("amount", 0.0)
                prev_merchant = prev.get("merchant", "").strip().lower()
                prev_date_str = prev.get("date", "")
                
                try:
                    prev_date = datetime.strptime(prev_date_str, "%Y-%m-%d")
                except Exception:
                    continue

                if e_id != prev_id and merchant == prev_merchant and amount == prev_amount:
                    # Check if transaction dates are within 2 days of each other
                    diff = abs((e_date - prev_date).days)
                    if diff <= 2:
                        anomalies.append({
                            "type": "Double Billing Warning",
                            "severity": "High",
                            "description": f"Potential duplicate charge of ${amount} detected at '{e.get('merchant')}' on {date_str} (similar to transaction on {prev_date_str}).",
                            "affected_ids": [e_id, prev_id]
                        })
                        
            seen_transactions.append(e)

        # --- 2. Scan for Large Transactions (Outliers) ---
        # Group by category to find median and standard deviations
        cat_amounts = {}
        for e in expenses:
            cat = e.get("category", "Other")
            amount = e.get("amount", 0.0)
            if cat not in cat_amounts:
                cat_amounts[cat] = []
            cat_amounts[cat].append(amount)

        for e in expenses:
            e_id = e.get("_id") or e.get("id")
            cat = e.get("category", "Other")
            amount = e.get("amount", 0.0)
            amounts_in_cat = cat_amounts[cat]
            
            if len(amounts_in_cat) >= 3:
                median = sorted(amounts_in_cat)[len(amounts_in_cat) // 2]
                # If transaction is more than 4x the median of that category and over $100
                if amount > 4 * median and amount > 100.0:
                    anomalies.append({
                        "type": "Category Outlier Warning",
                        "severity": "Medium",
                        "description": f"Abnormally high expense of ${amount} in category '{cat}' at '{e.get('merchant')}'. This is {round(amount/median, 1)}x the typical category median (${median}).",
                        "affected_ids": [e_id]
                    })

        # --- 3. Scan for Sudden Spikes in Daily Velocity ---
        # Calculate daily spending totals
        daily_totals = {}
        for e in expenses:
            date_str = e.get("date", "")
            amount = e.get("amount", 0.0)
            daily_totals[date_str] = daily_totals.get(date_str, 0.0) + amount

        if len(daily_totals) >= 3:
            median_daily = sorted(list(daily_totals.values()))[len(daily_totals) // 2]
            for day, total in daily_totals.items():
                if total > 3 * median_daily and total > 200.0:
                    # Find merchants involved on this day
                    merchants_today = [e.get("merchant") for e in expenses if e.get("date") == day]
                    anomalies.append({
                        "type": "Spending velocity alert",
                        "severity": "Low",
                        "description": f"Daily spending surge on {day}: Total spent was ${round(total, 2)}, which is {round(total/median_daily, 1)}x the daily historical median (${round(median_daily, 2)}). Activities: {', '.join(merchants_today[:3])}.",
                        "affected_ids": []
                    })

        # Deduplicate warnings to avoid double-printing
        unique_anomalies = []
        seen_descriptions = set()
        for a in anomalies:
            if a["description"] not in seen_descriptions:
                seen_descriptions.add(a["description"])
                unique_anomalies.append(a)

        return unique_anomalies

# Singleton Instance
fraud_agent = FraudAgent()
