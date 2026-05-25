import pandas as pd
import numpy as np

np.random.seed(42)

departments = ['Marketing', 'R&D', 'Sales', 'Operations', 'G&A']
quarters = ['Q1-2024', 'Q2-2024', 'Q3-2024', 'Q4-2024',
            'Q1-2025', 'Q2-2025', 'Q3-2025', 'Q4-2025']

rows = []
for dept in departments:
    base_budget = np.random.randint(500, 2000) * 1000
    for q in quarters:
        budget = base_budget * np.random.uniform(0.95, 1.10)
        # Inject realistic variance patterns
        if dept == 'Marketing' and q == 'Q3-2025':
            actual = budget * 1.28  # Big overspend story
        elif dept == 'R&D' and 'Q4' in q:
            actual = budget * 1.15  # Year-end push
        else:
            actual = budget * np.random.uniform(0.88, 1.12)

        rows.append({
            'Quarter': q,
            'Department': dept,
            'Budget_USD': round(budget, 2),
            'Actual_USD': round(actual, 2),
            'Variance_USD': round(actual - budget, 2),
            'Variance_Pct': round((actual - budget) / budget * 100, 2)
        })

df = pd.DataFrame(rows)
df.to_csv('data/fpna_data.csv', index=False)
print(f"Generated {len(df)} rows")
print(df.head(10))
