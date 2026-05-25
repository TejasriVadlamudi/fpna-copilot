import pandas as pd
from langchain.tools import tool

DF = pd.read_csv("data/fpna_data.csv")

@tool
def query_budget_data(department: str = "", quarter: str = "") -> str:
    """
    Retrieve budget vs actual data. Filter by department and/or quarter.
    Examples: query_budget_data(department='Marketing', quarter='Q3-2025')
    """
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
