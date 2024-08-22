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





def main():
    # Landing Page
    st.markdown("<h1 style='color:blue;'>pKa Determination Model (Demo)</h1>", unsafe_allow_html=True)

    st.write("**Indicator**  *m*-Cresol Purple (*m*-Cresolsulfonphthalein)")
    st.write("**pH range**   1.2  == red == 2.8  == yellow == 7.4  == purple == 9.0")

    # Select: upload a new file or analyze an existing file?
    option = st.selectbox(
        "What file do you want to analyse?",
        ("New file", "Existing file"),
        index=None,
        placeholder="Select the type of file."
    )

    # Debug point: checking the selected option
    st.write(f"Selected option: {option}")
    
    try:
        # Forcing the existing file option for now
        raw_df = file_loader(option)
        
        # Debug point: checking if the raw_df is loaded
        if raw_df is not None:
            st.write("Dataframe loaded successfully.")
            st.write(raw_df.head())  # Show a small portion of the dataframe for debugging
        else:
            st.write("Dataframe is None or empty.")
            return  # Exit if no data is loaded

        # Continue with the rest of the pipeline if the dataframe is valid
        if not raw_df.empty:
            st.write("Processing data...")
            pivot_df = pd.pivot_table(raw_df, columns="wavelength", index="volume(mm)", values="absorbance")
            st.pyplot(check_the_plot(pivot_df, "Raw Data"))
            
            start_wavelength, end_wavelength = get_wl_range(pivot_df)
            
            # Debug point: checking wavelength range selection
            st.write(f"Start Wavelength: {start_wavelength}, End Wavelength: {end_wavelength}")
            
            if start_wavelength and end_wavelength:
                spec_diff = preprocessor(pivot_df, start_wavelength, end_wavelength)
                pos_peaks, neg_peaks, peak_fig = peaks_n_plot(spec_diff)
                st.pyplot(peak_fig)
            else:
                st.write("Wavelength range was not selected properly.")
        else:
            st.write("Loaded dataframe is empty.")
    except Exception as e:
        # Capture and display the error
        st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()