import pandas as pd
import numpy as np

print("=== ODOO DATASET PROCESSOR - CSV EXPORT ===")
print("Loading dataset...")

# Load the original dataset
df = pd.read_csv('odoo.apps.csv')
print(f"Original dataset loaded: {df.shape}")

# Create a copy for processing
df_processed = df.copy()

print("\nProcessing features...")

# 1. Clean basic columns
df_processed['_id'] = df_processed['_id'].astype(str)
df_processed['Price'] = pd.to_numeric(df_processed['Price'], errors='coerce').fillna(0)
df_processed['Purchases'] = pd.to_numeric(df_processed['Purchases'], errors='coerce').fillna(0)

# 2. Extract rating votes from Rating column
df_processed['Rating_Votes'] = df_processed['Rating'].str.extract(r'(\d+)').astype(float).fillna(0)

# 3. Create text-based features
df_processed['Description_Length'] = df_processed['Description'].fillna('').str.len()
df_processed['Description_Word_Count'] = df_processed['Description'].fillna('').str.split().str.len()
df_processed['App_Name_Length'] = df_processed['App'].str.len()

# 4. Vendor analysis - THIS IS THE KEY PART
print("Calculating vendor statistics...")
vendor_stats = df_processed.groupby('Vendor').agg({
    'Purchases': ['mean', 'count', 'sum'],
    'Price': 'mean'
}).round(2)

# Flatten column names
vendor_stats.columns = ['Vendor_Avg_Purchases', 'Vendor_App_Count', 'Vendor_Total_Purchases', 'Vendor_Avg_Price']

# Map vendor statistics back to main dataset
df_processed['Vendor_Avg_Purchases'] = df_processed['Vendor'].map(vendor_stats['Vendor_Avg_Purchases'])
df_processed['Vendor_App_Count'] = df_processed['Vendor'].map(vendor_stats['Vendor_App_Count'])
df_processed['Vendor_Total_Purchases'] = df_processed['Vendor'].map(vendor_stats['Vendor_Total_Purchases'])
df_processed['Vendor_Avg_Price'] = df_processed['Vendor'].map(vendor_stats['Vendor_Avg_Price'])

# 5. Calculate vendor success rate (normalized)
max_avg_purchases = df_processed['Vendor_Avg_Purchases'].max()
df_processed['Vendor_Success_Rate'] = df_processed['Vendor_Avg_Purchases'] / max_avg_purchases

# 6. Price categories
df_processed['Price_Category'] = pd.cut(df_processed['Price'], 
                                       bins=[-0.01, 0.01, 100, 500, float('inf')],
                                       labels=['Free', 'Budget', 'Premium', 'Enterprise'])

# 7. Price vs vendor average
df_processed['Price_vs_Vendor_Avg'] = df_processed['Price'] / (df_processed['Vendor_Avg_Price'] + 0.01)

# 8. Success definition (top 20% of purchases)
success_threshold = df_processed['Purchases'].quantile(0.8)
df_processed['Success'] = (df_processed['Purchases'] > success_threshold).astype(int)

# 9. Percentile rankings
df_processed['Price_Percentile'] = df_processed['Price'].rank(pct=True)
df_processed['Purchases_Percentile'] = df_processed['Purchases'].rank(pct=True)
df_processed['Rating_Votes_Percentile'] = df_processed['Rating_Votes'].rank(pct=True)

# 10. Success levels
df_processed['Success_Level'] = pd.cut(df_processed['Purchases'], 
                                      bins=[df_processed['Purchases'].min()-1, 
                                            df_processed['Purchases'].quantile(0.5),
                                            df_processed['Purchases'].quantile(0.8),
                                            df_processed['Purchases'].quantile(0.95),
                                            df_processed['Purchases'].max()+1],
                                      labels=['Low', 'Medium', 'High', 'Very_High'])

print("Feature engineering complete!")

# SAVE THE PROCESSED DATASET
output_filename = 'odoo_apps_with_new_features.csv'
df_processed.to_csv(output_filename, index=False)

print(f"\n✅ SUCCESS! Dataset saved as: {output_filename}")
print(f"📊 Dataset shape: {df_processed.shape}")
print(f"📈 New columns added: {df_processed.shape[1] - df.shape[1]}")

# Show what's in the new dataset
print(f"\n📋 COLUMN LIST:")
for i, col in enumerate(df_processed.columns, 1):
    print(f"{i:2d}. {col}")

print(f"\n🔍 SAMPLE OF NEW FEATURES:")
new_features = ['Rating_Votes', 'Description_Length', 'Vendor_Success_Rate', 
                'Price_Category', 'Success', 'Success_Level']
print(df_processed[new_features].head())

print(f"\n📈 QUICK STATS:")
print(f"Success rate: {df_processed['Success'].mean():.1%}")
print(f"Unique vendors: {df_processed['Vendor'].nunique()}")
print(f"Average vendor success rate: {df_processed['Vendor_Success_Rate'].mean():.3f}")

# Create a smaller version with just the key columns for modeling
modeling_columns = [
    '_id', 'App', 'Vendor', 'Price', 'Purchases', 
    'Rating_Votes', 'Description_Length', 'Description_Word_Count', 'App_Name_Length',
    'Vendor_Avg_Purchases', 'Vendor_App_Count', 'Vendor_Success_Rate', 
    'Price_vs_Vendor_Avg', 'Success'
]

df_modeling = df_processed[modeling_columns].copy()
modeling_filename = 'odoo_apps_modeling_dataset.csv'
df_modeling.to_csv(modeling_filename, index=False)

print(f"\n✅ BONUS! Modeling dataset saved as: {modeling_filename}")
print(f"📊 Modeling dataset shape: {df_modeling.shape}")

print(f"\n🎉 DONE! You now have:")
print(f"   1. {output_filename} - Full dataset with all features")
print(f"   2. {modeling_filename} - Clean dataset for machine learning")
print(f"\nBoth files are ready to use!")