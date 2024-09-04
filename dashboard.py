import pandas as pd
import numpy as np
import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from sklearn.cluster import DBSCAN

# Configuration
SHEET_ID = "1RnuDFi45hl-rlQjXpXA40nUh36nUqI2b_FZr_G_8uJM"
SERVICE_ACCOUNT_FILE = "credentials.json"

# Hyperparameters
HEIGHT = 0.1
DISTANCE = 20
WIDTH = 3
EPS = 10
MIN_SAMPLES = 1
WINDOW = 5


# Functions
def edit_df(df: pd.DataFrame, i: int, j: int):
    """
    Function to drop empty cells and separate the type of additive compound and its volume.
    df: Dataframe to edit its contents. Its headers include information about the amount of HCl, the first row has wavelength and absorbance like header row, other cells have numerical values, there are some unnecessary & missing values as well.
    i: Start column
    j: End column
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


def data_transformer(df: pd.DataFrame):
    """
    Function to transform the shape of dataframe after dropping unnecessary & empty cells.
    df: Dataframe to transform.
    """
    # Remove unnecessary cells by detecting an empty row.
    df = df.dropna(axis = 1,
                  how = "all")
    idx = df.index[df.isnull().all(axis=1)].tolist()
    if len(idx) > 0:
        df = df.iloc[:idx[0], :]
    
    # Create the transformed dataframe.
    transformed_df = pd.DataFrame()
    iter = int(len(df.columns) // 2)
    for i in range(iter):
        temp = edit_df(df, (2 * i), (2 * i + 2))
        transformed_df = pd.concat([transformed_df, temp])

    return transformed_df.reset_index(drop = True)


def create_sheet(title: str):
    """
    Function to create a new sheet to the existing Google Sheet file.
    title: Name of the new sheet. 
    """
    # Access to the Google Sheet.
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE
                                                                        , scopes = ["https://www.googleapis.com/auth/spreadsheets"])
    
    service = build("sheets", "v4", credentials = credentials)

    # Create a new sheet to the Google Sheet.
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
    except:
        # Show the warning message if not working.
        st.warning("Something went wrong. Try again.", icon="⚠️") # ":material/warning:"


def upload_df(df: pd.DataFrame, title: str):
    """
    Function to upload a dataframe to Google Sheet.
    df: Dataframe to upload.
    title: Name of the sheet to save the data.
    """
    # Get the values in the input Dataframe.
    value = [[' '] + df.columns.tolist()] + (df.reset_index().values.tolist())
    
    # Access to the Google Sheet.
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE
                                                                        , scopes = ["https://www.googleapis.com/auth/spreadsheets"])
    
    service = build("sheets", "v4", credentials = credentials)

    # Update the contents of the input df to the sheet with the name as the input title.
    try:
        request = service.spreadsheets().values().update(
            spreadsheetId = SHEET_ID,
            valueInputOption = "RAW",
            range = f"{title}!A1",
            body = dict(majorDimension = "ROWS",
            values = value)
        ).execute()
    except:
        # Show the warning message if not working.
        st.warning("Something went wrong. Try again.", icon="⚠️") # ":material/warning:"


def get_sheets_list():
    """
    Function to get the list of names of existing sheets on the Google Sheet.
    """
    # Access to the Google Sheet.
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE
                                                                        , scopes = ["https://www.googleapis.com/auth/spreadsheets"])
    
    service = build("sheets", "v4", credentials = credentials)

    # Get the list of existing sheets.
    sheet_metadata = service.spreadsheets().get(spreadsheetId = SHEET_ID).execute()
    sheet_info = sheet_metadata.get("sheets", "")
    sheet_names = tuple(titles['properties']['title'] for titles in sheet_info)
    return sheet_names


def get_df_from_sheet(title: str):
    """
    Function to get data from an existng sheet and load it in pd.DataFrame format.
    """
    # Access to the Google Sheet.
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE
                                                                        , scopes = ["https://www.googleapis.com/auth/spreadsheets"])
    
    service = build("sheets", "v4", credentials = credentials)
    
    # Get the data saved in the sheet named as the input title.
    results = service.spreadsheets().values().get(spreadsheetId = SHEET_ID,
                                                  range = title
                                                  ).execute()
    raw_data = results.get("values", [])
    # Transform the loaded data into Pandas DataFrame format.
    raw_df = (pd.DataFrame(raw_data[1:], columns=raw_data[0])).iloc[:, 1:]

    for c in range(len(raw_df.columns)):
        try:
            raw_df.iloc[:, c] = raw_df.iloc[:, c].apply(eval)
        except:
            continue

    return raw_df


def file_loader(option: str):
    """
    Function to upload a new file to Google Sheets after creating a new sheet or to load an existing file from Google Sheets.
    option: To upload a new file or to read an existing file?
    """
    sheet_names = get_sheets_list()
    # Upload and save the new file after creating a new sheet in the Google Sheet.
    if option == "New file":
        uploaded_file = st.file_uploader(label = "Upload a file in excel format.",
                                        type = ['xlsx'],
                                        )  
        if uploaded_file is not None:
            raw_data = pd.read_excel(uploaded_file)
            raw_df = data_transformer(raw_data)

            sheet_title = st.text_input("Type the name of new dataset.", "TITLE HERE")
            if sheet_title not in sheet_names: # To avoid the same name of the sheet.
                if sheet_title != "TITLE HERE":
                    create_sheet(sheet_title)
                    upload_df(raw_df, sheet_title)
                    st.write(sheet_title, "is successfully uploaded.")
                    st.session_state.raw_df = raw_df
                    return raw_df
            else:
                st.warning(f"Sheet [{sheet_title}] already exists.", icon="⚠️") # ":material/warning:"
            
    elif option == "Existing file":
        # Choose one sheet among the given list of existing sheets.
        sheet_title = st.selectbox(
                                 "Choose the dataset.",
                                 sheet_names,
                                 index = None,
                                 placeholder = "Select a dataset."
                                 )
        # Get data in the selected sheet.
        try:
            raw_df = get_df_from_sheet(sheet_title)
            st.write(sheet_title, "is selected.")
            st.session_state.raw_df = raw_df
            return raw_df
        except:
            st.warning("There is no data selected.", icon="⚠️") # ":material/warning:"


def check_the_plot(df: pd.DataFrame, title: str):
    """
    Function to plot a pivoted table, absorbance versus wavelength as lines.
    df: Table with indices as a group, columns as x-axis values. The values will be plotted on y-axis.
    title: Title of the plot.
    """
    fig = plt.figure(figsize = (10, 6))
    # Set x-axis with the column data.
    x_data = df.columns

    # Draw line graphs of each index(sample).
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
    plt.grid(True)
    return fig


def get_wl_range(pivot_df: pd.DataFrame):
    """
    Function to get the range of wavelength of the dataset.
    df: Raw dataset in pivotted shape.
    """
    # Get the minimum/maximum wavelength values in the dataset and round them up to make them integer.
    min_range = int(round(pivot_df.columns.min(),0))
    max_range = int(round(pivot_df.columns.max(),0))

    # Make a list of string-typed numbers for wavelength intervals.
    wl_range = [str(i) for i in range(min_range, max_range, 10)]
    wl_range.append(str(max_range))

    # Create a selection slider to choose the wavelength range to analyse.
    start_wl, end_wl = st.select_slider(
        "Select a range of wavelength.",
        options = wl_range,
        value = (wl_range[0], wl_range[-1])
        )

    # Convert the string-typed numbers into integers.
    if type(start_wl) == str:
        start_wl = eval(start_wl)
    if type(end_wl) == str:
        end_wl = eval(end_wl)
        
    return start_wl, end_wl


def preprocessor(pivot_df: pd.DataFrame, start_wavelength: int, end_wavelength: int):
    """
    Function to preprocess the dataset - cut the dataset within the selected wavelength range, fill the missing values using the interpolation algorithm, and normalise the values.
    pivot_df: Dataset to preprocess.
    start_wavelength: the minimum wavelength to analyse.
    end_wavelength: the maximum wavelength to analyse.
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


def getting_peaks(array:pd.Series, height: float, distance: float, width: float, negative = False):
    """
    Function to find peaks and return the pd.Series shaped data points of the peaks.
    array: Array of x-axis & y-axis data points.
    height: Required hight of peaks.
    distance: Required minimal horizontal distance (>= 1) in samples between neighbouring peaks.
    width: Required width of peaks in samples.
    negative: Find negative side peaks if True, find positive peaks if False.
    """
    if negative:
        array = - array # Convert the direction if finding negative peaks.

    # Detect peaks with the given parameters settings.
    peaks, _ = find_peaks(array, 
                          height = height,
                          distance = distance,
                          width = width)
    if negative:
        return -array.iloc[peaks]
    else:
        return array.iloc[peaks]


def peaks_n_plot_with_clustering(df: pd.DataFrame, height=HEIGHT, distance=DISTANCE, width=WIDTH, eps=EPS, min_samples=MIN_SAMPLES):
    """
    Function to get coordinate data of every detected peaks, create clusters, and visualise them.
    df: Dataset to find peaks.
    height: Required hight of peaks.
    distance: Required minimal horizontal distance (>= 1) in samples between neighbouring peaks.
    width: Required width of peaks in samples.
    eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
    """
    fig = plt.figure(figsize = (10, 6))
    pos_peaks = pd.Series(dtype = float)
    neg_peaks = pd.Series(dtype = float)

    # Find positive & negative peak points from each sample.
    for idx in df.index:
        series = df.loc[idx]
        plt.plot(series, label = f"{idx} mm")

        pos_peak = getting_peaks(series, height = height, distance = distance, width = width)
        pos_peaks = pd.concat([pos_peaks, pos_peak]) if not pos_peaks.empty else pos_peak

        neg_peak = getting_peaks(series, height = height, distance = distance, width = width, negative = True)
        neg_peaks = pd.concat([neg_peaks, neg_peak]) if not neg_peaks.empty else neg_peak

    # Create clusters of peak points with DBSCAN
    all_peaks = pd.concat([pos_peaks, neg_peaks])
    peak_points = np.array(list(zip(all_peaks.index.values, all_peaks.values)))
    dbscan = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit(peak_points)
    labels = dbscan.labels_

    final_wavelengths = []
    for label in np.unique(labels):
        cluster_idx = np.where(labels == label)[0]
        cluster_peaks = all_peaks.iloc[cluster_idx]
        most_significant_peak = abs(cluster_peaks).idxmax() # Get the wavelength with the highest absolute absorbance value.
        # Visualise the detected peak clusters with annotations.
        plt.plot(cluster_peaks, "x") 
        plt.axvline(x = most_significant_peak, linestyle = "--")
        plt.text(most_significant_peak + 1, 
                 plt.ylim()[1] * 0.95, 
                 f'{most_significant_peak:.2f} nm', 
                 verticalalignment='bottom',
                 fontsize=12)
        
        final_wavelengths.append(most_significant_peak)

    plt.title("Detected Peaks", fontsize = 18, weight = "bold")
    plt.legend(loc = "best", title = "HCl")
    plt.xlabel("Wavelength (nm)", fontsize = 14)
    plt.ylabel("Absorbance Difference", fontsize = 14)
    plt.grid(True)
 
    return fig, final_wavelengths


def sigmoid(pH: float, pKa: float, a: float, b: float):
    """
    Function to generate values fitted the sigmoid curve with the given values.
    pH: The input value for which you want to calculate the sigmoid function.
    pKa: The value at which the sigmoid function reaches its midpoint.
    a: The scaling factor that determines the maximum value of the function.
    b: The baseline or offset of the function, shifting the sigmoid curve vertically.
    """
    return a / (1 + np.exp(-(pH - pKa) * 2)) + b


def plot_sigmoidal_fit_for_wavelengths(df: pd.DataFrame, final_wavelengths: list):
    """
    Function to generate and plot sigmoidal fits for total absorbance differences across various pH levels at significant wavelengths. It estimates values fitted the curve and shows the results pictorially.
    df: Dataframe, where each row represents different volumes (indexed by volume), and each column represents different wavelengths. The DataFrame contains the absorbance data.
    final_wavelengths: a list of significant wavelengths at which the highest absorbance values appear in each cluster.
    """
    # Map the amount of HCl in the input dataset to corresponding pH values.
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
    pH_values = np.array([pH_mapping[str(volume)] for volume in df.index])
    pH_range = np.linspace(min(pH_values), max(pH_values), 500)
    
    # Generate a sigmoid plot at each wavelength.
    for wavelength in final_wavelengths:
        wl_idx = np.where(df.columns == wavelength)[0][0]
        total_abs_diff = df.iloc[:, wl_idx].abs()

        # Initial guess for the sigmoid function.
        p0 = [6.5, max(total_abs_diff) - min(total_abs_diff), min(total_abs_diff)]  
        # Estimate the pKa value from the fitted parameters.
        popt, _ = curve_fit(sigmoid, pH_values, total_abs_diff, p0, maxfev=10000)
        pKa_value = round(popt[0], 2)
        fit = sigmoid(pH_range, *popt)
    
        # Initial guess for the sigmoid function.
        p0 = [6.5, max(total_abs_diff) - min(total_abs_diff), min(total_abs_diff)]  
        
        # Estimate the pKa value from the fitted parameters.
        popt, _ = curve_fit(sigmoid, pH_values, total_abs_diff, p0, maxfev=10000)
        pKa_value = round(popt[0], 3)
        fit = sigmoid(pH_range, *popt)

        # Visualis each result.
        fig = plt.figure(figsize = (10, 6))
        plt.scatter(pH_values, total_abs_diff, color = "black", label = "Experimental Data")
        plt.plot(pH_range, fit, label = "Fitted curve", color = "red", )
        plt.axvline(x = pKa_value, color = "blue", linestyle = "--")
        plt.annotate(f"     pKa: {pKa_value:.3f}", xy = (pKa_value, sigmoid(pKa_value, *popt)))
        
        plt.xlabel("pH")
        plt.ylabel("Total Absorbance Difference")
        plt.title(f"pKa Determination at {wavelength:.2f} nm", fontsize = 18, weight = "bold")
        plt.legend(loc = "best")
        plt.grid(True)

        st.pyplot(fig)
        st.subheader(f"Estimated pKa is :blue[{pKa_value}]", divider = "gray")
        st.write()


def quantify_total_fluctuation(df: pd.DataFrame, selected_wavelength: float, window = WINDOW):
    """
    Function to quantify the total fluctuation in absorbance values around a selected wavelength using standard deviation.
    df : The DataFrame containing absorbance data with wavelengths as columns.
    selected_wavelength : The wavelength around which fluctuation is to be quantified.
    window : The window size (in nm) to consider on either side of the selected wavelength.
    """
    # Select the range of wavelengths within ±window nm.
    wavelength_range = df.loc[:, (df.columns >= selected_wavelength - window) & (df.columns <= selected_wavelength + window)]

    # Calculate the standard deviation across the wavelength range for each sample.
    flunctuation_df = wavelength_range.std(axis = 1)
    # Sum up all the fluctuation values to return a single number as a consistency score.
    total_fluctuation = flunctuation_df.sum()
    eps = 0.0000001 # Add a small number to prevent the ZeroDivisionError.
    
    return 1/(total_fluctuation + eps)


def main():
    # Landing Page
    st.markdown("<h1 style='color:black;'>pKa Determination Model</h1>", unsafe_allow_html=True)
    st.divider()
    st.subheader("[pH Indicator]")
    st.image("mCresolPurple.png")
    st.divider()

    if "selected_option" not in st.session_state:
        st.session_state.selected_option = None

    if "selected_wavelength" not in st.session_state:
        st.session_state.selected_wavelength = None


    # Select: upload a new file or analyze an existing file?
    option = st.selectbox(
        "What file do you want to analyse?",
        ("New file", "Existing file"),
        index = None,
        placeholder="Select the type of file."
    )

    if st.session_state.selected_option is None:
        st.session_state.selected_option = option


    if st.session_state.selected_option is not None and 'raw_df' not in st.session_state:
        # Load the file as selected.
        raw_df = file_loader(st.session_state.selected_option)

    # Used the same file once uploaded.
    if st.session_state.selected_option is not None and 'raw_df' in st.session_state:
        raw_df = st.session_state.raw_df
        try:
            pivot_df = pd.pivot_table(raw_df, columns="wavelength", index="volume(mm)", values="absorbance")
            st.pyplot(check_the_plot(pivot_df, "Raw Data"))
            st.session_state.pivot_df = pivot_df
            # Select the range of wavelength to analyse.
            st.session_state.selected_wavelength = get_wl_range(st.session_state.pivot_df)
            
            if 'pivot_df' in st.session_state and st.session_state.selected_wavelength is not None:
                # Get the spectral differences & Visualise the peak detection results.
                start_wavelength = st.session_state.selected_wavelength[0]
                end_wavelength = st.session_state.selected_wavelength[1]
                spec_diff = preprocessor(st.session_state.pivot_df, start_wavelength, end_wavelength)
                peak_fig, final_wavelengths = peaks_n_plot_with_clustering(spec_diff)
                st.pyplot(peak_fig)

                # Calculate the consistency score of each peak.
                wavelength_scores = [(wavelength, quantify_total_fluctuation(spec_diff, wavelength)) for wavelength in final_wavelengths]
                wavelength_scores = sorted(wavelength_scores, key=lambda x: x[1], reverse=True)

                # Select the wavelengths to analyse.
                st.write("Select the wavelengths to analyse.")
                selected_wavelengths = []
                for wl, c_score in wavelength_scores:
                    checkbox_label = f"wavelength: {wl:.2f}  --- Consistency Score: {c_score:.2f}"
                    if st.checkbox(checkbox_label):
                        selected_wavelengths.append(wl)

                # Input wavelength if the given values are not appropriate.
                add_custom_wavelength = st.checkbox("Add wavelength to analyse.")
                if add_custom_wavelength:
                    custom_wavelength = st.number_input(
                        "Enter wavelength (nm)",
                        min_value = float(start_wavelength),
                        max_value = float(end_wavelength),
                        step = 0.1
                    )
                    if custom_wavelength:
                        selected_wavelengths.append(custom_wavelength)
                
                # Plot the sigmoid graphs for each wavelength.
                if len(selected_wavelengths) > 0:
                    plot_sigmoidal_fit_for_wavelengths(spec_diff, selected_wavelengths)
                else:
                    st.error(f"There is no wavelength selected.", icon="⚠️")

        except:
            st.error(f"There is no file selected.", icon="⚠️") 


    # Restart the analysis if clicking "Restart" button.
    if st.button("Restart"):
        for key in st.session_state.keys():
            st.session_state[key] = None
        st.rerun()


if __name__ == "__main__":
    main()