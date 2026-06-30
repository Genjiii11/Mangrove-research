import pandas as pd
import plotly.graph_objects as go
import numpy as np

# --- Constants and Configuration ---
DATA_FILE = 'final_environmental_vars_with_MHI.csv'
YEARS_MHI = [2000, 2010, 2024]
YEARS_LAND_USE = [2000, 2010, 2024]
AREA_PER_GRID = 1  # Area in km^2 per GridID

# Plotting style configuration
FONT_FAMILY = "Times New Roman"
FONT_SIZE = 14
MHI_GRADE_LABELS = ['Severely Degraded', 'Mildly Degraded', 'Sub-healthy', 'Healthy']
MHI_COLORS = ['#d7191c', '#fdae61', '#abdda4', '#2b83ba'] # Diverging color scale: Red -> Yellow -> Green -> Blue

LAND_USE_LABELS = ["Mangrove", "Water", "Cropland", "Impervious Surface"]
LAND_USE_COLORS = {
    "Mangrove": "#2b83ba",
    "Water": "#4575b4",
    "Cropland": "#a6611a",
    "Impervious Surface": "#d7191c",
}


def load_and_preprocess_data(file_path, years):
    """
    Loads and filters the dataset for the specified years.
    """
    print(f"Loading data for years: {years}...")
    try:
        # Use low_memory=False to handle mixed dtypes more efficiently
        df = pd.read_csv(file_path, low_memory=False)
        df_filtered = df[df['year'].isin(years)].copy()
        print("Data loaded and filtered successfully.")
        return df_filtered
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None

def assign_mhi_grades(df):
    """
    Calculates global MHI quantiles and assigns grades to the dataframe.
    """
    print("Assigning MHI grades...")
    mhi_all = df['MHI'].dropna()
    if mhi_all.empty:
        print("Warning: MHI column is empty or all NaN. Skipping MHI grading.")
        return df
    quantiles = mhi_all.quantile([0, 0.25, 0.5, 0.75, 1.0]).values
    quantiles[0] = -np.inf # Ensure the lowest values are included
    quantiles[-1] = np.inf # Ensure the highest values are included

    df['mhi_grade'] = pd.cut(df['MHI'], bins=quantiles, labels=MHI_GRADE_LABELS, right=False)
    print("MHI grades assigned.")
    return df

def plot_mhi_transition_sankey(df):
    """
    Generates and saves a Sankey diagram for MHI grade transitions over time.
    """
    print("Generating MHI transition Sankey diagram...")
    if df is None or df.empty or 'mhi_grade' not in df.columns or df['mhi_grade'].isnull().all():
        print("Dataframe is empty or 'mhi_grade' column is missing/all NaN. Skipping MHI plot.")
        return

    # Pivot data to track transitions
    df_pivot = df.pivot_table(index='GridID', columns='year', values='mhi_grade', aggfunc='first')
    df_pivot = df_pivot.dropna()

    # --- Prepare Sankey Data ---
    all_nodes_labels = []
    node_dict = {}
    node_x = {}
    
    # Create unique nodes for each grade and year
    for i, year in enumerate(YEARS_MHI):
        for grade in MHI_GRADE_LABELS:
            node_name = f"{grade}_{year}"
            if node_name not in node_dict:
                node_dict[node_name] = len(all_nodes_labels)
                all_nodes_labels.append(grade)
                node_x[node_dict[node_name]] = i / (len(YEARS_MHI) - 1) if len(YEARS_MHI) > 1 else 0.5


    sources, targets, values = [], [], []

    # Calculate flows for each period
    for i in range(len(YEARS_MHI) - 1):
        year_start, year_end = YEARS_MHI[i], YEARS_MHI[i+1]
        
        flows = df_pivot.groupby([year_start, year_end]).size().reset_index(name='count')
        
        for _, row in flows.iterrows():
            source_name = f"{row[year_start]}_{year_start}"
            target_name = f"{row[year_end]}_{year_end}"
            
            if source_name in node_dict and target_name in node_dict:
                sources.append(node_dict[source_name])
                targets.append(node_dict[target_name])
                values.append(row['count'] * AREA_PER_GRID)

    # --- Create Plot ---
    fig = go.Figure(data=[go.Sankey(
        arrangement='snap',
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_nodes_labels,
            color=[MHI_COLORS[i % len(MHI_GRADE_LABELS)] for i in range(len(all_nodes_labels))],
            x=[node_x[i] for i in range(len(all_nodes_labels))],
            y=[0.1] * len(all_nodes_labels) # Placeholder y, will be adjusted by layout
        ),
        link=dict(source=sources, target=targets, value=values)
    )])

    annotations = []
    for i, year in enumerate(YEARS_MHI):
        x_pos = i / (len(YEARS_MHI) - 1) if len(YEARS_MHI) > 1 else 0.5
        annotations.append(
            dict(
                x=x_pos,
                y=-0.1,
                xref="paper",
                yref="paper",
                text=str(year),
                showarrow=False,
                font=dict(family=FONT_FAMILY, size=FONT_SIZE + 2)
            )
        )

    fig.update_layout(
        font_family=FONT_FAMILY, 
        font_size=FONT_SIZE,
        annotations=annotations,
        margin=dict(b=100) # Add bottom margin for annotations
    )

    print("Saving MHI transition Sankey diagram to 'mhi_health_transition.png'...")
    fig.write_image("mhi_health_transition.png", scale=3)
    print("Done.")


def plot_land_use_transition_sankey(df):
    """
    Generates and saves a Sankey diagram for land use transitions over time.
    """
    print("Generating Land Use transition Sankey diagram...")
    if df is None or df.empty:
        print("Dataframe is empty. Skipping Land Use plot.")
        return

    # --- Determine Dominant Land Use ---
    df['mangrove_percent'] = df['proportion_of_landscape'] * 100
    land_use_cols = ['mangrove_percent', 'cropland_percent', 'water_percent', 'impervious_surface_percent']
    df['dominant_land_use'] = df[land_use_cols].idxmax(axis=1)
    df['dominant_land_use'] = df['dominant_land_use'].str.replace('_percent', '').str.capitalize()
    df['dominant_land_use'] = df['dominant_land_use'].str.replace('Impervious_surface', 'Impervious Surface')


    # --- Pivot and Prepare Data for Sankey ---
    df_pivot = df.pivot_table(index='GridID', columns='year', values='dominant_land_use', aggfunc='first')
    df_pivot = df_pivot.dropna()

    # --- Prepare Sankey Data ---
    all_unique_labels = set()
    for year in YEARS_LAND_USE:
        all_unique_labels.update(df_pivot[year].unique())
    all_labels = sorted(list(all_unique_labels))
    
    all_nodes_labels = []
    node_dict = {}
    node_x = {}

    for i, year in enumerate(YEARS_LAND_USE):
        for label in all_labels:
            node_name = f"{label}_{year}"
            if node_name not in node_dict:
                node_dict[node_name] = len(all_nodes_labels)
                all_nodes_labels.append(label)
                node_x[node_dict[node_name]] = i / (len(YEARS_LAND_USE) - 1) if len(YEARS_LAND_USE) > 1 else 0.5

    sources, targets, values = [], [], []

    # Calculate flows for each period
    for i in range(len(YEARS_LAND_USE) - 1):
        year_start, year_end = YEARS_LAND_USE[i], YEARS_LAND_USE[i+1]
        
        flows = df_pivot.groupby([year_start, year_end]).size().reset_index(name='count')
        
        for _, row in flows.iterrows():
            source_name = f"{row[year_start]}_{year_start}"
            target_name = f"{row[year_end]}_{year_end}"
            
            if source_name in node_dict and target_name in node_dict:
                sources.append(node_dict[source_name])
                targets.append(node_dict[target_name])
                values.append(row['count'] * AREA_PER_GRID)

    # --- Create Plot ---
    node_colors = [LAND_USE_COLORS.get(label, '#808080') for label in all_labels] * len(YEARS_LAND_USE)

    fig = go.Figure(data=[go.Sankey(
        arrangement='snap',
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_nodes_labels,
            color=node_colors,
            x=[node_x[i] for i in range(len(all_nodes_labels))],
            y=[0.1] * len(all_nodes_labels)
        ),
        link=dict(source=sources, target=targets, value=values)
    )])

    annotations = []
    for i, year in enumerate(YEARS_LAND_USE):
        x_pos = i / (len(YEARS_LAND_USE) - 1) if len(YEARS_LAND_USE) > 1 else 0.5
        annotations.append(
            dict(
                x=x_pos,
                y=-0.1,
                xref="paper",
                yref="paper",
                text=str(year),
                showarrow=False,
                font=dict(family=FONT_FAMILY, size=FONT_SIZE + 2)
            )
        )

    fig.update_layout(
        font_family=FONT_FAMILY, 
        font_size=FONT_SIZE,
        annotations=annotations,
        margin=dict(b=100) # Add bottom margin for annotations
    )

    print(f"Saving Land Use transition Sankey diagram to 'land_use_transition.png'...")
    fig.write_image("land_use_transition.png", scale=3)
    print("Done.")


def main():
    """
    Main function to run the analysis and generate plots.
    """
    # --- Plot 1: MHI Health Transition ---
    df_mhi = load_and_preprocess_data(DATA_FILE, YEARS_MHI)
    if df_mhi is not None:
        df_mhi_graded = assign_mhi_grades(df_mhi)
        plot_mhi_transition_sankey(df_mhi_graded)
    else:
        print("Could not load data for MHI plot. Skipping.")

    # --- Plot 2: Land Use Transition ---
    df_land_use = load_and_preprocess_data(DATA_FILE, YEARS_LAND_USE)
    if df_land_use is not None:
        plot_land_use_transition_sankey(df_land_use)
    else:
        print("Could not load data for Land Use plot. Skipping.")


if __name__ == '__main__':
    main()
