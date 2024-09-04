
# 23/24 Durham University MDS Research Project
  
### Project Name
Ionisation Conatant Determination Algorithm for Flow System

### Students
Qifan Lin
Yurim Yang

### Supervisor  
Ian Baxendale

### Project Contents
This project built a data analysis algorithm to determine an ionisation constant(pKa) from a UV-vis spectrometry dataset generated by a flow system.
The flow system generates continuous changes in the ratio of each solution to be analysed by the spectrometer, causing gradient pH changes in each solution. 
Continuous data can be tricky to analyse due to its enormous size and complexity. 
The algorithm here aims to tackle the inconvenient and laborious analysis process of the vast result data by using peak detection algorithms and visualising each step.

### Language
Python 3.11

### Library Version
 - Pandas 2.2.2
 - Numpy 1.26.4
 - matplotlib 3.8.2
 - Scipy 1.12.0
 - Sklearn 1.4.0
 - Streamlit 1.36.0
 - googleapiclient 2.136.0
 
 
### How to
1. Download all the files in the same directory.
2. Run the "dashboard.py" using Termonal.
3. Run code: streamlit run FILE_PATH/dashboard.py
