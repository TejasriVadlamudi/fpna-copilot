import pandas as pd
from langchain.tools import tool

DF = pd.read_csv("data/fpna_data.csv")

@tool
def query_budget_data(department: str = "", quarter: str = "") -> str:
    """Retrieve budget vs actual data. Filter by department and/or quarter (both optional)."""
    df = DF.copy()
    if department:
        df = df[df['Department'].str.lower() == department.lower()]
    if quarter:
        df = df[df['Quarter'] == quarter]
    if df.empty:
        return "No data found for those filters."
    return df.to_string(index=False)

@tool
def calculate_variance(department: str, quarter: str) -> str:
    """Calculate budget variance for a specific department and quarter."""
    row = DF[(DF['Department'].str.lower() == department.lower()) &
             (DF['Quarter'] == quarter)]
    if row.empty:
        return f"No data found for {department} in {quarter}."
    r = row.iloc[0]
    direction = "unfavorable (overspend)" if r['Variance_USD'] > 0 else "favorable (underspend)"
    return (
        f"{department} | {quarter}\n"
        f"Budget: ${r['Budget_USD']:,.0f}\n"
        f"Actual: ${r['Actual_USD']:,.0f}\n"
        f"Variance: ${r['Variance_USD']:,.0f} ({r['Variance_Pct']:.1f}%) - {direction}"
    )

@tool
def top_variances(quarter: str, n: int = 3) -> str:
    """Return the top N departments with the largest absolute variance for a quarter."""
    df = DF[DF['Quarter'] == quarter].copy()
    if df.empty:
        return f"No data for {quarter}."
    df['Abs_Variance'] = df['Variance_USD'].abs()
    top = df.nlargest(n, 'Abs_Variance')[['Department', 'Variance_USD', 'Variance_Pct']]
    return top.to_string(index=False)

@tool
def summarize_all_data() -> str:
    """Provide a high-level summary of the entire FP&A dataset: all departments, all quarters, key totals and patterns. Use this when the user asks for an overview or general analysis."""
    total_budget = DF['Budget_USD'].sum()
    total_actual = DF['Actual_USD'].sum()
    total_variance = total_actual - total_budget
    pct_variance = (total_variance / total_budget) * 100

    dept_summary = DF.groupby('Department').agg(
        Total_Budget=('Budget_USD', 'sum'),
        Total_Actual=('Actual_USD', 'sum'),
        Avg_Variance_Pct=('Variance_Pct', 'mean')
    ).round(2)

    biggest_overspender = DF.groupby('Department')['Variance_USD'].sum().idxmax()
    biggest_underspender = DF.groupby('Department')['Variance_USD'].sum().idxmin()
    worst_quarter = DF.groupby('Quarter')['Variance_USD'].sum().idxmax()

    summary = f"""DATASET SUMMARY (8 quarters, 5 departments):
Total Budget: ${total_budget:,.0f}
Total Actual: ${total_actual:,.0f}
Total Variance: ${total_variance:,.0f} ({pct_variance:.2f}%)

Biggest Overspender Department (all-time): {biggest_overspender}
Biggest Underspender Department (all-time): {biggest_underspender}
Worst Variance Quarter: {worst_quarter}

BY DEPARTMENT:
{dept_summary.to_string()}
"""
    return summary

@tool
def compare_departments(quarter: str = "") -> str:
    """Compare all departments side-by-side. Optional: filter to a single quarter. Use this when the user asks 'which department' or 'compare departments'."""
    df = DF.copy()
    if quarter:
        df = df[df['Quarter'] == quarter]
        if df.empty:
            return f"No data for {quarter}."
        result = df[['Department', 'Budget_USD', 'Actual_USD', 'Variance_USD', 'Variance_Pct']].sort_values('Variance_Pct', ascending=False)
    else:
        result = df.groupby('Department').agg(
            Total_Budget=('Budget_USD', 'sum'),
            Total_Actual=('Actual_USD', 'sum'),
            Total_Variance=('Variance_USD', 'sum'),
            Avg_Variance_Pct=('Variance_Pct', 'mean')
        ).round(2).sort_values('Avg_Variance_Pct', ascending=False)
    return result.to_string()
