# Data manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

print(" Libraries imported successfully!")
 # Load GRACE data (2003-2008)
grace_2003_2008 = pd.read_csv('/kaggle/input/india-drought-analysis-data-2000-2023/groundwater_data/grace_2003_2008.csv')

# Load GRACE data (2009-2017)
grace_2009_2017 = pd.read_csv('/kaggle/input/india-drought-analysis-data-2000-2023/groundwater_data/grace_2009_2017.csv')

# Combine GRACE datasets
grace_data = pd.concat([grace_2003_2008, grace_2009_2017], ignore_index=True)

print(f" GRACE Data Shape: {grace_data.shape}")
print(f" Date Range: {grace_data['date'].min()} to {grace_data['date'].max()}")
print(f"\n GRACE groundwater data loaded!")
grace_data.head()
 # Load GLDAS data (2018-2023)
gldas_data = pd.read_csv('/kaggle/input/india-drought-analysis-data-2000-2023/groundwater_data/gldas_2018_2023.csv')

print(f"GLDAS Data Shape: {gldas_data.shape}")
print(f" Date Range: {gldas_data['date'].min()} to {gldas_data['date'].max()}")
print(f"\n GLDAS soil moisture data loaded!")
gldas_data.head()
# Load ICRISAT agricultural data
agri_data = pd.read_csv('/kaggle/input/india-drought-analysis-data-2000-2023/agricultural_data/icrisat_district_data.csv')

print(f"📊 Agricultural Data Shape: {agri_data.shape}")
print(f"📊 Columns: {list(agri_data.columns)}")
print(f"\n✅ Agricultural data loaded!")
# Preview agricultural data
agri_data.head()
 # Summary statistics for GRACE data
print("📊 GRACE Data Summary:")
print(grace_data.describe())
#Check for missing values
print("🔍 Missing Values in GRACE Data:")
print(grace_data.isnull().sum())

print("\n🔍 Missing Values in GLDAS Data:")
print(gldas_data.isnull().sum())

print("\n🔍 Missing Values in Agricultural Data:")
print(agri_data.isnull().sum())
# Unique regions and districts
if 'region' in grace_data.columns:
    print("🗺 Regions in Dataset:")
    print(grace_data['region'].unique())
    print(f"\n📍 Total Districts: {grace_data['district'].nunique()}")
    print("\nDistricts by Region:")
    print(grace_data.groupby('region')['district'].nunique())
 # Convert date column to datetime
grace_data['date'] = pd.to_datetime(grace_data['date'])

# Extract year and month
grace_data['year'] = grace_data['date'].dt.year
grace_data['month'] = grace_data['date'].dt.month
 # Plot groundwater anomalies by region (if region column exists)
if 'region' in grace_data.columns and 'lwe_thickness' in grace_data.columns:
    plt.figure(figsize=(14, 6))
    
    for region in grace_data['region'].unique():
        region_data = grace_data[grace_data['region'] == region]
        monthly_avg = region_data.groupby('date')['lwe_thickness'].mean()
        plt.plot(monthly_avg.index, monthly_avg.values, label=region, linewidth=2)
    
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Normal Level')
    plt.title('Groundwater Storage Anomalies by Region (GRACE)', fontsize=16, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Liquid Water Equivalent (cm)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("📊 Negative values indicate groundwater depletion")
    print("📊 Positive values indicate groundwater recharge")
# Convert date column to datetime
gldas_data['date'] = pd.to_datetime(gldas_data['date'])

# Plot soil moisture trends
if 'region' in gldas_data.columns and 'soil_moisture' in gldas_data.columns:
    plt.figure(figsize=(14, 6))
    
    for region in gldas_data['region'].unique():
        region_data = gldas_data[gldas_data['region'] == region]
        monthly_avg = region_data.groupby('date')['soil_moisture'].mean()
        plt.plot(monthly_avg.index, monthly_avg.values, label=region, linewidth=2)
    
    plt.title('Soil Moisture Trends by Region (GLDAS)', fontsize=16, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Soil Moisture (kg/m²)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
# Average groundwater anomaly by district
if 'district' in grace_data.columns and 'lwe_thickness' in grace_data.columns:
    district_avg = grace_data.groupby('district')['lwe_thickness'].mean().sort_values()
    
    plt.figure(figsize=(12, 8))
    colors = ['red' if x < 0 else 'green' for x in district_avg.values]
    district_avg.plot(kind='barh', color=colors, edgecolor='black', linewidth=0.5)
    plt.title('Average Groundwater Anomaly by District (2003-2017)', fontsize=16, fontweight='bold')
    plt.xlabel('Average LWE Thickness (cm)', fontsize=12)
    plt.ylabel('District', fontsize=12)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()
    print(f"\n🔴 Districts with groundwater depletion: {(district_avg < 0).sum()}")
    print(f"🟢 Districts with groundwater recharge: {(district_avg > 0).sum()}")
# Analyze seasonal patterns in groundwater
if 'month' in grace_data.columns and 'lwe_thickness' in grace_data.columns:
    monthly_pattern = grace_data.groupby('month')['lwe_thickness'].mean()
    
    plt.figure(figsize=(10, 6))
    plt.plot(monthly_pattern.index, monthly_pattern.values, marker='o', linewidth=2, markersize=8)
    plt.title('Average Groundwater Anomaly by Month', fontsize=16, fontweight='bold')
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Average LWE Thickness (cm)', fontsize=12)
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
# Explore crop data structure
print("🌾 Agricultural Data Structure:")
print(agri_data.info())

print("\n📊 Sample Data:")
print(agri_data.head(10))
