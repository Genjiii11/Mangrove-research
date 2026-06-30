import pandas as pd
import numpy as np
from skyborn.calc import mann_kendall_multidim

def calculate_slope(df, value_col, year_col='year', grid_col='GridID'):
    """
    Calculates the Theil-Sen slope for a given variable in a long-format DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame in long format.
        value_col (str): The name of the column containing the values to analyze.
        year_col (str): The name of the column containing the year.
        grid_col (str): The name of the column containing the grid cell identifier.

    Returns:
        pd.DataFrame: A DataFrame with grid identifiers and their corresponding slopes.
    """
    print(f"Calculating slope for {value_col}...")
    
    # Pivot the data to wide format
    df_pivot = df.pivot_table(index=year_col, columns=grid_col, values=value_col)
    
    # Get the grid IDs in the correct order
    grid_ids = df_pivot.columns
    
    # Convert to numpy array for processing
    data_array = df_pivot.values
    
    # Perform Mann-Kendall trend test
    # The 'trend' key in the result corresponds to the Theil-Sen slope
    mk_result = mann_kendall_multidim(data_array, axis=0)
    slopes = mk_result['trend']
    
    # Create a results DataFrame
    slope_df = pd.DataFrame({
        grid_col: grid_ids,
        f'{value_col}_slope': slopes
    })
    
    print("Calculation complete.")
    return slope_df

def main():
    """
    Main function to load data, calculate slopes for MHI and impervious surface,
    and save the results to a CSV file.
    """
    # Load the dataset
    file_path = 'final_environment_with_MHI.csv'
    print(f"Loading data from {file_path}...")
    try:
        # Use specific columns to reduce memory usage
        columns_to_load = ['GridID', 'year', 'MHI', 'impervious_surface_percent']
        df = pd.read_csv(file_path, usecols=columns_to_load)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return
    except ValueError as e:
        print(f"Error loading columns: {e}. Please ensure the CSV contains {columns_to_load}.")
        return

    # Filter data for the specified year range
    print("Filtering data for years 2000-2024...")
    df = df[df['year'].between(2000, 2024)]
    
    # Calculate slope for MHI
    mhi_slopes = calculate_slope(df, 'MHI')
    
    # Calculate slope for impervious surface
    impervious_slopes = calculate_slope(df, 'impervious_surface_percent')
    
    # Merge the results
    print("Merging results...")
    merged_slopes = pd.merge(mhi_slopes, impervious_slopes, on='GridID')
    
    # Rename columns for clarity
    merged_slopes.rename(columns={
        'MHI_slope': 'MHI_slope',
        'impervious_surface_percent_slope': 'Impervious_Slope'
    }, inplace=True)
    
    # Save the final results
    output_path = 'slopes.csv'
    merged_slopes.to_csv(output_path, index=False)
    print(f"Slopes successfully calculated and saved to {output_path}")
    print("\nFinal DataFrame head:")
    print(merged_slopes.head())

if __name__ == '__main__':
    main()