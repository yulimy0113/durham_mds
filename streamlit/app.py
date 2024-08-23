import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from google.oauth2 import service_account
from googleapiclient.discovery import build
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from sklearn.cluster import DBSCAN
import pyarrow
print(pyarrow.__version__)

# Configuration
SHEET_ID = "1RnuDFi45hl-rlQjXpXA40nUh36nUqI2b_FZr_G_8uJM"
SERVICE_ACCOUNT_FILE = Path.joinpath(Path.cwd(),"streamlit/cohesive-scope-428317-j2-c2ffabd1a9fd.json")


# Hyperparameters
HEIGHT = 0.3
DISTANCE = 20
WIDTH = 1
MAX_SIGMOID = 1.5

# Functions
def edit_df(df, i, j):
    """
    function to drop empty cells and separate the type of additive compound and its volume.
    df: pd.DataFrame
        headers include information about the amount of HCl,
        the first row has wavelength and absorbance like header row,
        other cells have numerical values,
        there are some unnecessary & missing values as well.
    i: int, start column
    j: int, end column
    temp: pd.DataFrame, the edited dataframe.
    """
    # Remove null values of each HCl column.
    temp = df.iloc[1:, i:j].dropna()
    
    # Take the information about the additive & the amount of it from the header.
    cols = temp.columns
    add = (cols[0]).split(' ')
    if add[-1] != "hcl":
        additive = 'hcl'
        vol = 0
    else:
        additive = add[-1]
        vol = eval(add[0])
    temp = temp.rename(columns = {cols[0] : "wavelength",
                                 cols[1] : "absorbance"})
    temp['additive'] = additive
    temp['volume(mm)'] = vol
    return temp

def data_transformer(df):
    """
    function to transform the shape of dataframe.
    df: pd.DataFrame
    transformed_df: pd.DataFrame, the dataframe after dropping unnecessary & empty cells.
    """
    # Remove unnecessary cells by detecting an empty row.
    df = df.dropna(axis = 1,
                  how = "all")
    idx = df.index[df.isnull().all(axis=1)].tolist()
    if len(idx) > 0:
        df = df.iloc[:idx[0], :]
    
    # Create the transformed data frame.
    transformed_df = pd.DataFrame()
    iter = int(len(df.columns) // 2)
    
    for i in range(iter):
        temp = edit_df(df, (2 * i), (2 * i + 2))
        transformed_df = pd.concat([transformed_df, temp])

    return transformed_df.reset_index(drop = True)

def create_sheet(title):
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE
                                                                        , scopes = ["https://www.googleapis.com/auth/spreadsheets"])
    
    service = build("sheets", "v4", credentials = credentials)
    
    data = {"requests": [
        {
            "addSheet": {
                "properties": {"title": title}
            }
        }
    ]} 
    try:
        res = service.spreadsheets().batchUpdate(spreadsheetId = SHEET_ID,
                                                 body = data).execute()
    except HTTPError as error:
        print(f"An error occurred: {error}")
        return error

def upload_df(df, title):
    value = [[' '] + df.columns.tolist()] + (df.reset_index().values.tolist())

    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE
                                                                        , scopes = ["https://www.googleapis.com/auth/spreadsheets"])
    
    service = build("sheets", "v4", credentials = credentials)

    try:
        request = service.spreadsheets().values().update(
            spreadsheetId = SHEET_ID,
            valueInputOption = "RAW",
            range = f"{title}!A1",
            body = dict(majorDimension = "ROWS",
            values = value)
        ).execute()
    except HTTPError as error:
        print(f"An error occurred: {error}")
        return error

def get_sheets_list():
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE
                                                                        , scopes = ["https://www.googleapis.com/auth/spreadsheets"])
    
    service = build("sheets", "v4", credentials = credentials)
    
    sheet_metadata = service.spreadsheets().get(spreadsheetId = SHEET_ID).execute()
    sheet_info = sheet_metadata.get("sheets", "")
    sheet_names = tuple(titles['properties']['title'] for titles in sheet_info)
    return sheet_names

def get_df_from_sheet(title):
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE
                                                                        , scopes = ["https://www.googleapis.com/auth/spreadsheets"])
    
    service = build("sheets", "v4", credentials = credentials)
    results = service.spreadsheets().values().get(spreadsheetId = SHEET_ID,
                                                  range = title
                                                  ).execute()
    raw_data = results.get("values", [])
    raw_df = (pd.DataFrame(raw_data[1:], columns=raw_data[0])).iloc[:, 1:]

    for c in range(len(raw_df.columns)):
        try:
            raw_df.iloc[:, c] = raw_df.iloc[:, c].apply(eval)
        except:
            continue

    return raw_df
def file_loader(option):
    """
    A function to upload a new file to Google Sheet after creating a new sheet 
    or to load an existing file from Google Sheet.
    
    option: str, either "New file" or "Existing file".
    """
    sheet_names = get_sheets_list()
    
    # Log the available sheets to ensure they are being retrieved correctly
    st.write("Available sheets:", sheet_names)
    
    if option == "New file":
        uploaded_file = st.file_uploader(label="Upload a file in excel format.",
                                         type=['xlsx'])  
        if uploaded_file is not None:
            raw_data = pd.read_excel(uploaded_file)
            raw_df = data_transformer(raw_data)

            sheet_title = st.text_input("Type the name of new dataset.", "TITLE HERE")
            
            if sheet_title not in sheet_names:
                if sheet_title != "TITLE HERE":
                    create_sheet(sheet_title)
                    upload_df(raw_df, sheet_title)
                    st.write(sheet_title, "is successfully uploaded.")
                    return raw_df
            else:
                st.write(f"Sheet [{sheet_title}] already exists.")
            
    elif option == "Existing file":
        sheet_title = st.selectbox("Choose the dataset.", sheet_names, index=0, placeholder="Select a dataset.")
        
        st.write("Selected sheet:", sheet_title)
        
        try:
            raw_df = get_df_from_sheet(sheet_title)
            if raw_df is not None and not raw_df.empty:
                st.write(f"Sheet '{sheet_title}' is selected and data is loaded successfully.")
                return raw_df
            else:
                st.write("The selected sheet has no data.")
                return None
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return None

def check_the_plot(df, title):
    """
    A function to plot a pivoted table in a scattered plot format.
    df: pd.Dataframe, 
        table with indices as a group, columns as x-axis values. The values will be plotted on y-axis.
    title: str, 
           title of the plot.
    """
    fig = plt.figure(figsize = (10, 6))
    x_data = df.columns

    for i in df.index:
        y_data = df.loc[i]
        plt.plot(x_data, y_data, label = f"{i} mm")

    plt.xlabel("Wavelength (nm)", fontsize = 14)
    plt.ylabel("Absorbance", fontsize = 14)
    plt.title(title,
             fontsize = 18,
             weight = "bold")
    plt.legend(loc = "upper right",
              title = "HCl")
    return fig

def get_wl_range(pivot_df):
    """
    df: pd.DataFrame, raw dataset in pivotted shape.
    """
    min_range = int(round(pivot_df.columns.min(),0))
    max_range = int(round(pivot_df.columns.max(),0))

    wl_range = [str(i) for i in range(min_range, max_range, 10)]
    wl_range.append(str(max_range))

    start_wl, end_wl = st.select_slider(
        "Select a range of wavelength.",
        options = wl_range,
        value = (wl_range[0], wl_range[-1])
        )
    
    if type(start_wl) == str:
        start_wl = eval(start_wl)
    if type(end_wl) == str:
        end_wl = eval(end_wl)
        
    return start_wl, end_wl

def preprocessor(pivot_df, start_wavelength, end_wavelength):
    """
    """
    # Trim the dataset based on selected wavelength range.
    pivot_df = pivot_df.loc[:, pivot_df.columns[(pivot_df.columns > start_wavelength) & (pivot_df.columns < end_wavelength)]]
    
    # Fill the missing values. - interpolation
    pivot_df = pivot_df.apply(pd.to_numeric,
                             errors = "coerce").interpolate(method = "linear",
                                                           axis = 1)
    
    # Normalisation based on the absorbance at the longest wavelength.
    norm_df = pivot_df.dropna(axis = 1).copy()
    for hcl in norm_df.index:
        norm_abs = (norm_df.loc[hcl]).values[-1] 
        norm_df.loc[hcl] = (norm_df.loc[hcl]).sub(norm_abs)
        
    # Calculate spectral differences based on the acid.
    spec_diff = norm_df.subtract(norm_df.iloc[-1], axis = 1) # or norm_df.loc[500]

    return spec_diff

def getting_peaks(array:pd.Series,
                  height, 
                  distance, 
                  width, 
                  negative = False):
    """
    A function to find peaks and return the pd.Series shaped data points of the peaks.
    array: pd.Series, array of x-axis & y-axis data points.
    height: float, required hight of peaks. 
    distance: int, required minimal horizontal distance (>= 1) in samples between neighbouring peaks.
    width: int, required width of peaks in samples.
    negative: boolean, find negative side peaks if True, find positive peaks if False.
    """
    if negative:
        array = - array
        
    peaks, _ = find_peaks(array, 
                          height = height,
                          distance = distance,
                          width = width)
    if negative:
        return -array.iloc[peaks]
    else:
        return array.iloc[peaks]

def peaks_n_plot(df):
    """
    A function to get coordinate data of every detected peaks and plot them.
    df: pd.DataFrame, preprocessed dataset.
    """
    fig = plt.figure(figsize = (10, 6))
    pos_peaks = pd.Series()
    neg_peaks = pd.Series()

    for idx in df.index:
        series = df.loc[idx]
        plt.plot(series, label = f"{idx} mm")
        pos_peak = getting_peaks(series,
                                height = HEIGHT,
                                distance = DISTANCE,
                                width = WIDTH)
        pos_peaks = pd.concat([pos_peaks if not pos_peaks.empty else None,
                              pos_peak if not pos_peak.empty else None])

        neg_peak = getting_peaks(series,
                                height = HEIGHT,
                                distance = DISTANCE,
                                width = WIDTH,
                                negative = True)
        neg_peaks = pd.concat([neg_peaks if not neg_peaks.empty else None,
                              neg_peak if not neg_peak.empty else None])

        plt.plot(pos_peaks, "x", color = "red")
        plt.plot(neg_peaks, "x", color = "blue")

    plt.title("Peaks in the Spectral Difference Plot", fontsize = 18, weight = 
            "bold")
    plt.legend(loc = "upper right",
              title = "HCl")
    plt.xlabel("Wavelength (nm)", fontsize = 14)
    plt.ylabel("Spectral Difference", fontsize = 14)
    
    return pos_peaks, neg_peaks, fig



def peaks_n_plot_with_clustering(df, height=HEIGHT, distance=DISTANCE, width=WIDTH, eps=10, min_samples=1):
    """
    A function to get coordinate data of every detected peaks, cluster them, and plot them.
    df: pd.DataFrame, preprocessed dataset.
    height, distance, width: parameters for peak detection.
    eps, min_samples: parameters for DBSCAN clustering.
    """
    fig = plt.figure(figsize=(10, 6))
    pos_peaks = pd.Series(dtype=float)
    neg_peaks = pd.Series(dtype=float)

    for idx in df.index:
        series = df.loc[idx]
        plt.plot(series, label=f"{idx} mm")
        
        # Finding positive and negative peaks
        pos_peak = getting_peaks(series, height=height, distance=distance, width=width)
        pos_peaks = pd.concat([pos_peaks, pos_peak]) if not pos_peaks.empty else pos_peak

        neg_peak = getting_peaks(series, height=height, distance=distance, width=width, negative=True)
        neg_peaks = pd.concat([neg_peaks, neg_peak]) if not neg_peaks.empty else neg_peak

        # Plot peaks
        plt.plot(pos_peaks, "x", color="red")
        plt.plot(neg_peaks, "x", color="blue")

    # Clustering peaks with DBSCAN
    all_peaks = pd.concat([pos_peaks, neg_peaks])
    peak_wavelengths = all_peaks.index.values.reshape(-1, 1)
    
    # Apply DBSCAN clustering to the detected peaks
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(peak_wavelengths)
    labels = clustering.labels_

    # Select the most significant peak within each cluster
    final_wavelengths = []
    for label in np.unique(labels):
        cluster_indices = np.where(labels == label)[0]
        cluster_peaks = all_peaks.iloc[cluster_indices]
        most_significant_peak = cluster_peaks.idxmax()  # Get the wavelength with the highest peak value
        final_wavelengths.append(most_significant_peak)
    
    # Plot the vertical lines for the most significant peaks in each cluster
    for wavelength in final_wavelengths:
        plt.axvline(x=wavelength, color='green', linestyle='--', label=f'{wavelength} nm')

    plt.title("Peaks in the Spectral Difference Plot with Clustering", fontsize=18, weight="bold")
    plt.legend(loc="upper right", title="HCl")
    plt.xlabel("Wavelength (nm)", fontsize=14)
    plt.ylabel("Spectral Difference", fontsize=14)

    plt.grid(True)
    plt.show()

    print("Final significant wavelengths:", final_wavelengths)
    return pos_peaks, neg_peaks, final_wavelengths, fig


def plot_sigmoidal_fit_for_wavelengths(spec_diff, final_wavelengths):
    """
    Function to generate and plot sigmoidal fits for total absorbance differences
    across various pH levels at significant wavelengths.

    Parameters:
    spec_diff : pd.DataFrame
        A DataFrame where each row represents different volumes (indexed by volume),
        and each column represents different wavelengths. The DataFrame contains
        absorbance data.
    final_wavelengths : list or np.array
        A list or array of significant wavelengths for which the sigmoidal fits
        should be generated.

    Returns:
    None. The function generates and displays a plot for each wavelength.
    """

    # Define the pH mapping for the volumes
    pH_mapping = {
        '0': 2,
        '10': 1.885,
        '15': 1.707,
        '20': 1.611,
        '25': 1.506,
        '30': 1.404,
        '40': 1.264,
        '50': 1.184,
        '60': 1.105,
        '100': 0.878,
        '500': 0.3
    }

    # Map the volumes in spec_diff.index to corresponding pH values
    pH_values = np.array([pH_mapping[str(volume)] for volume in spec_diff.index])

    # Loop over each significant wavelength and generate a separate plot
    for wavelength in final_wavelengths:
        # Find the nearest wavelength in the DataFrame columns
        closest_wavelength_index = (np.abs(spec_diff.columns.astype(float) - wavelength)).argmin()
        closest_wavelength = spec_diff.columns[closest_wavelength_index]
        
        # Calculate the total absorbance difference for this wavelength
        total_abs_diff = spec_diff.iloc[:, closest_wavelength_index].abs()
        
        # Define the sigmoidal function for fitting
        def sigmoid(pH, pKa, a, b):
            return a / (1 + np.exp(-(pH - pKa) * 2)) + b
        
        # Fit the combined data using the pH values and total absorbance differences
        p0 = [6.5, max(total_abs_diff) - min(total_abs_diff), min(total_abs_diff)]  # Initial guess
        popt, _ = curve_fit(sigmoid, pH_values, total_abs_diff, p0, maxfev=10000)
        
        # Extract the pKa value from the fitted parameters
        pKa_value = round(popt[0], 2)
        
        # Generate a pH range for plotting the fit
        pH_range = np.linspace(min(pH_values), max(pH_values), 500)
        fit = sigmoid(pH_range, *popt)
        
        # Plot the fit against the combined data
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(pH_values, total_abs_diff, color='black', label='Experimental Data')
        ax.plot(pH_range, fit, label=f'Fit (pKa = {pKa_value:.2f})', color='red')
        
        # Add a vertical line at the determined pKa value
        ax.axvline(x=pKa_value, color='blue', linestyle='--', label=f'pKa = {pKa_value:.2f}')
        
        # Label the axes and title
        ax.set_xlabel('pH')
        ax.set_ylabel('Total Absorbance Difference')
        ax.set_title(f'Total Absorbance Difference vs pH at {closest_wavelength:.2f} nm')
        ax.legend(loc='upper left')
        ax.grid(True)
        
        # Display the plot using Streamlit
        st.pyplot(fig)
        
        # Display the determined pKa value
        st.write(f'Wavelength: {closest_wavelength:.2f} nm, Determined pKa value: {pKa_value:.2f}')

def quantify_total_fluctuation(df, selected_wavelength, window=3):
    """
    Quantifies the total fluctuation in absorbance values around a selected wavelength.
    
    Parameters:
    df : pd.DataFrame
        The DataFrame containing absorbance data with wavelengths as columns.
    selected_wavelength : float
        The wavelength around which fluctuation is to be quantified.
    window : int, optional, default=5
        The window size (in nm) to consider on either side of the selected wavelength.
    
    Returns:
    total_fluctuation : float
        The total fluctuation (sum of standard deviations) across all volumes.
    """
    # Select the range of wavelengths within ±window nm of the selected wavelength
    wavelength_range = df.loc[:, (df.columns >= selected_wavelength - window) & 
                                   (df.columns <= selected_wavelength + window)]
    
    # Calculate the standard deviation across the wavelength range for each volume (row)
    fluctuation_df = wavelength_range.std(axis=1)
    
    # Sum up all the fluctuation values to return a single number
    total_fluctuation = fluctuation_df.sum()
    
    return 1/total_fluctuation

def main():
    # Landing Page
    st.markdown("<h1 style='color:blue;'>pKa Determination Model (Demo)</h1>", unsafe_allow_html=True)

    st.write("**Indicator**  *m*-Cresol Purple (*m*-Cresolsulfonphthalein)")
    st.write("**pH range**   1.2  :red[== red ==]  2.8  :orange[== yellow ==]  7.4  :violet[== purple ==]  9.0")

    # Select: upload a new file or analyze an existing file?
    option = st.selectbox(
        "What file do you want to analyse?",
        ("New file", "Existing file"),
        placeholder="Select the type of file."
    )

    try:
        # Load the file
        raw_df = file_loader(option)
        
        if raw_df is None or raw_df.empty:
            st.error("Dataframe is None or empty.")
            return  # Exit if no data is loaded

        # Process the data if loaded correctly
        pivot_df = pd.pivot_table(raw_df, columns="wavelength", index="volume(mm)", values="absorbance")
        st.pyplot(check_the_plot(pivot_df, "Raw Data"))
        
        start_wavelength, end_wavelength = get_wl_range(pivot_df)
        
        if start_wavelength and end_wavelength:
            spec_diff = preprocessor(pivot_df, start_wavelength, end_wavelength)
            pos_peaks, neg_peaks, final_wavelengths, peak_fig = peaks_n_plot_with_clustering(spec_diff)
            st.pyplot(peak_fig)

            # Checkbox for each final wavelength with consistency score
            st.write("### Select wavelengths to generate sigmoidal fit plots:")
            selected_wavelengths = []
            for wavelength in final_wavelengths:
                consistency_score = quantify_total_fluctuation(spec_diff, wavelength)
                checkbox_label = f"Wavelength {wavelength:.1f} nm (Consistency Score: {consistency_score:.2f})"
                if st.checkbox(checkbox_label):
                    selected_wavelengths.append(wavelength)
            
            # Custom wavelength checkbox and input
            add_custom_wavelength = st.checkbox("Add a custom wavelength")

            if add_custom_wavelength:
                custom_wavelength = st.number_input(
                    "Enter a custom wavelength (nm)", 
                    min_value=float(start_wavelength), 
                    max_value=float(end_wavelength), 
                    step=0.1
                )
                if custom_wavelength:
                    selected_wavelengths.append(custom_wavelength)

            # Plot sigmoidal fit for selected wavelengths
            if selected_wavelengths:
                st.write("### Sigmoidal Fit Plots for Selected Wavelengths")
                plot_sigmoidal_fit_for_wavelengths(spec_diff, selected_wavelengths)
            else:
                st.warning("No wavelengths selected.")
        else:
            st.warning("Wavelength range was not selected properly.")

    except Exception as e:
        # Capture and display the error
        st.error(f"An error occurred: {e}")



if __name__ == "__main__":
    main()