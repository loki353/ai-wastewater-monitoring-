WATERWISE - PROJECT HANDOFF
===========================

This is the complete handoff package for the AI-Assisted Wastewater
Monitoring and Anomaly Detection project. It includes the Python analysis,
interactive Streamlit dashboard, original UCI dataset file, setup launcher,
variable dictionary, workflow diagram, result previews and presenter guide.

EASIEST WAY TO START (WINDOWS)
------------------------------
1. Extract the ZIP into Downloads (do not run files from the ZIP preview).
2. Open the extracted WATERWISE_Project_Handoff folder.
3. Double-click START_WATERWISE.bat.
4. The launcher installs/checks Python libraries, runs the analysis from the
   included dataset if outputs are not present, and starts the dashboard.
5. Keep the black terminal window open while using the dashboard.

IMPORTANT: DO NOT launch the dashboard with `python app.py`. That runs the
file as an ordinary Python script, not as a Streamlit app. The launcher uses
`python -m streamlit run app.py` correctly. If starting manually, use that
command from this folder after running `python main.py` once.

REQUIREMENTS
------------
- Windows computer with Python 3.10 or newer installed.
- Internet is required on first launch to install Python libraries.
- The UCI dataset source file is included in data\water-treatment.data, so
  a separate dataset download is not needed.

WHAT THE APP DOES
-----------------
- Cleans 527 observations and 38 numeric wastewater variables.
- Handles missing values using median imputation.
- Uses Isolation Forest for unsupervised anomaly screening.
- Shows BOD, COD and suspended-solids measurements on separate scales.
- Lets users inspect flagged observations and explore cleaned measurements.
- Provides dataset-relative, rule-based investigation suggestions.

FILES
-----
main.py                 Data cleaning, analysis, ML and output generation
app.py                  Interactive Streamlit dashboard
START_WATERWISE.bat     One-click Windows setup and launch
requirements.txt        Python dependencies
data\water-treatment.data  Included original UCI dataset file
data_dictionary.csv    Descriptions of the 38 variables
workflow_diagram.png    Project workflow
results_preview\       Preview graphs from the project's run
outputs\               Analysis CSVs and charts generated on first launch
PRESENTER_GUIDE.txt     Demo sequence and likely faculty questions
RESULTS_SUMMARY.txt     Results recorded from Saketh's local run

INTERPRETATION LIMITATIONS
--------------------------
An anomaly means an unusual combination of measurements, not proof of a
pollution incident, equipment fault, or regulatory violation. Percentile
thresholds are relative to this dataset, not legal discharge limits. The
recommendations are rule-based investigation prompts, not direct ML outputs.
The data is historical from one urban wastewater-treatment plant, not live
sensor data.

DATASET SOURCE AND CREDIT
-------------------------
Poch, M. (1993). Water Treatment Plant [Dataset]. UCI Machine Learning
Repository. https://doi.org/10.24432/C5FS4C
Dataset page: https://archive.ics.uci.edu/dataset/106/water+treatment+plant
License: Creative Commons Attribution 4.0 International (CC BY 4.0).

Keep the dataset citation in any report or presentation.
