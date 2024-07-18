import pandas as pd
import numpy as np
import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
import matplotlib.pyplot as plt

st.subheader(body = ":blue[pKa] Determination Model (Demo)",
             divider = "grey")

data_input = st.selectbox(
                        "How would you like to input the dataset?",
                        ("Select from Google Sheet", "Upload a file")
                        )

if data_input == "Select from Google Sheet":
    # Credentials & Connection
    SERVICE_ACCOUNT_FILE = "cohesive-scope-428317-j2-c2ffabd1a9fd.json"
    CREDENTIALS = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE,
                                                                        scopes = ["https://www.googleapis.com/auth/spreadsheets"])
    service = build("sheets",
                "v4",
                credentials = CREDENTIALS)

    SPREADSHEET_ID = "1uyqxcWlHmQZlwjqdPmAnrduxPP_V5IG1E13GjQl0nyI"
    
    sheet_metadata = service.spreadsheets().get(spreadsheetId = SPREADSHEET_ID).execute()
    sheet_info = sheet_metadata.get("sheets", "")
    sheet_names = tuple(titles['properties']['title'] for titles in sheet_info)
               
    SHEET_NAME = st.selectbox(
                             "Choose the data sheet.",
                             sheet_names
                             )
    st.write("Dataset:", SHEET_NAME)
    
    results = service.spreadsheets().values().get(spreadsheetId = SPREADSHEET_ID,
                                                range = SHEET_NAME).execute()

    raw_data = results.get("values", [])
    df = pd.DataFrame(np.asarray(raw_data[1:], dtype = np.float64), columns = raw_data[0])
    df = df.set_index('pH')


else:
    uploaded_file = st.file_uploader("Upload a file in excel format.")
    df = pd.read_csv(uploaded_file, index_col=0)

pH_values = df.index.tolist()
wavelengths = [eval(w) for w in df.columns]
data = np.asarray(df)

# Plot the toy data
fig = plt.figure(figsize = (10, 7))
for i, pH in enumerate(pH_values):
    plt.plot(wavelengths, data[i], label = f"pH {pH:.2f}")

plt.xlabel("Wavelength (nm)")
plt.ylabel("Absorbance")
plt.title("UV Spectrum of 4-Nitrophenol in Different pH Buffer Solutions")
plt.legend(title = "pH",
          bbox_to_anchor = (0, 1),
          loc = "upper left",
          ncol = 2)
plt.grid = True
st.pyplot(fig)