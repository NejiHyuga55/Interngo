import pandas as pd
import numpy as np
import ast

# Load the data
df = pd.read_csv('internship_data.csv')

# Check missing values
print("Missing values before handling:")
print(df.isnull().sum())

# Handle missing values
def handle_missing_values(df):


     # Convert to numeric, handling any non-numeric values
    df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')
    
    # Count non-null values
    non_null_count = df['Duration'].notna().sum()
    
    if non_null_count > 0:
        # Try median first
        try:
            median_val = df['Duration'].median()
            if not pd.isna(median_val):
                df['Duration'] = df['Duration'].fillna(median_val)
            else:
                # If median is NaN, try mean
                mean_val = df['Duration'].mean()
                df['Duration'] = df['Duration'].fillna(mean_val)
        except:
            # If any error occurs, use mean
            mean_val = df['Duration'].mean()
            df['Duration'] = df['Duration'].fillna(mean_val)
    else:
        # If all values are missing, fill with 0 or appropriate default
        df['Duration'] = df['Duration'].fillna(0)
    
    return df

# Main execution
if __name__ == "__main__":
    try:
        # Load your data here
        # df = pd.read_csv('your_data.csv')
        
        # Test data
        df = pd.DataFrame({
            'Duration': [10, 20, None, 'invalid', 40, None],
            'Other_Column': [1, 2, 3, 4, 5, 6]
        })
        
        print("Before processing:")
        print(df)
        print(f"Data types: {df['Duration'].dtype}")
        
        # Process the data
        df = handle_missing_values(df)
        
        print("\nAfter processing:")
        print(df)
        print(f"Data types: {df['Duration'].dtype}")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Please check your data and try again")


    
    # Fill missing Skills with empty lists
    df['Skills'] = df['Skills'].fillna('[]')
    
    # Fill missing Perks with empty lists
    df['Perks'] = df['Perks'].fillna('[]')
    
    # Fill missing Location with 'Not specified'
    df['Location'] = df['Location'].fillna('Not specified')
    
    # Fill missing Stipend with 'Unspecified'
    df['Stipend'] = df['Stipend'].fillna('Unspecified')
    
    # # Fill numeric columns with appropriate defaults
    # df['Duration'] = df['Duration'].fillna(df['Duration'].median())
    
    # Fill applications with 0 for "Be an early applicant" or missing
    df['Number of Applications'] = df['Number of Applications'].replace('Be an early applicant', 0)
    df['Number of Applications'] = pd.to_numeric(df['Number of Applications'], errors='coerce').fillna(0)
    
    

df = handle_missing_values(df)
print("\nMissing values after handling:")
print(df.isnull().sum())

import re

def clean_and_standardize(df):
    # Clean Location column
    def extract_locations(loc_str):
        if pd.isna(loc_str) or loc_str == 'Not specified':
            return []
        # Remove parentheses and quotes
        loc_str = re.sub(r"[()']", "", str(loc_str))
        # Split by commas and strip whitespace
        locations = [loc.strip() for loc in loc_str.split(',')]
        # Remove empty strings
        return [loc for loc in locations if loc]
    
    df['Location'] = df['Location'].apply(extract_locations)
    
    # Clean Skills column (convert string representation to list)
    def parse_list_string(list_str):
        try:
            if pd.isna(list_str):
                return []
            # Handle both string representations and actual strings
            if isinstance(list_str, str) and list_str.startswith('['):
                return ast.literal_eval(list_str)
            elif isinstance(list_str, str):
                return [skill.strip() for skill in list_str.split(',')]
            else:
                return []
        except:
            return []
    
    df['Skills'] = df['Skills'].apply(parse_list_string)
    
    # Clean Perks column
    df['Perks'] = df['Perks'].apply(parse_list_string)
    
    # Clean and convert Stipend to numeric values
    def parse_stipend(stipend_str):
        if pd.isna(stipend_str) or stipend_str == 'Unspecified':
            return np.nan, np.nan
        
        # Handle different formats
        if 'Unpaid' in str(stipend_str):
            return 0, 0
        if 'Performance Based' in str(stipend_str) or 'Incentives' in str(stipend_str):
            return np.nan, np.nan  # Or handle separately
        
        # Extract numbers using regex
        numbers = re.findall(r'[\d,]+', str(stipend_str))
        numbers = [float(num.replace(',', '')) for num in numbers]
        
        if len(numbers) == 1:
            return numbers[0], numbers[0]  # Single value
        elif len(numbers) >= 2:
            return min(numbers), max(numbers)  # Range
        else:
            return np.nan, np.nan
    
    df[['Stipend_Min', 'Stipend_Max']] = df['Stipend'].apply(
        lambda x: pd.Series(parse_stipend(x))
    )
    
    # Clean Duration (extract numeric part)
    def parse_duration(duration_str):
        if pd.isna(duration_str):
            return np.nan
        # Extract numbers
        numbers = re.findall(r'\d+', str(duration_str))
        if numbers:
            return int(numbers[0])
        return np.nan
    
    df['Duration_Months'] = df['Duration'].apply(parse_duration)
    
    # Clean Intern Type
    def clean_intern_type(intern_type):
        if pd.isna(intern_type):
            return 'Not specified'
        if isinstance(intern_type, str) and intern_type.startswith('['):
            try:
                types = ast.literal_eval(intern_type)
                return types[0] if types else 'Not specified'
            except:
                return 'Not specified'
        return str(intern_type)
    
    df['Intern_Type_Clean'] = df['Intern Type'].apply(clean_intern_type)
    
    return df

df = clean_and_standardize(df)

# Display cleaned data sample
print("\nCleaned data sample:")
print(df[['Role', 'Location', 'Skills', 'Stipend_Min', 'Stipend_Max', 'Duration_Months']].head())