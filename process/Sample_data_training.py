import pandas as pd

file_path = 'C:/Users/凡/Desktop/project/durham_mds/data/PDNTBA pKa scans.xlsx'
sheet_name = 'pdntba20mmscan2'
df_new = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')

# Check the headers and data to ensure correct column access
print("Column Headers and Data Sample:")
print(df_new.head())

# Remove any completely empty columns
df_new = df_new.dropna(axis=1, how='all')

# Naming columns based on the header content directly
new_columns = []
for i in range(0, len(df_new.columns), 2):
    base_name = df_new.columns[i].strip()  # Use the base name directly from headers
    new_columns.extend([
        f"{base_name}_Wavelength",
        f"{base_name}_Absorbance"
    ])
df_new.columns = new_columns

# Optionally, you can clean the data by converting all to numeric and handling NaNs
for col in df_new.columns:
    df_new[col] = pd.to_numeric(df_new[col], errors='coerce')

# Display the renamed DataFrame to verify correct column naming
print("\nRenamed DataFrame:")
print(df_new.head())
