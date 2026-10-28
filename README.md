<img width="186" height="238" alt="taylors_logo png" src="https://github.com/user-attachments/assets/9a47d3e8-ef3f-445c-8e37-2e0ed434878b" />

### **“Taylor’s University Smart Campus – EV Analytics & Machine Learning Dashboard”**

### Project Overview
This project is an interactive Streamlit web application developed for analysing Electric Vehicle (EV) charging activity on a smart campus. It combines data analytics, visualisation, and a machine-learning/rule-based risk analysis interface to help understand EV charging behaviour, energy usage, revenue, charger performance, and charging-session risks.

The application contains three main sections:

1. **EV Taylor's Core Metrics**
   - Executive KPI dashboard
   - Total energy delivered
   - Average session duration
   - Peak revenue
   - Total charging sessions
   - Energy delivered by day
   - Charging sessions by charger type
   - Peak vs non-peak revenue
   - Filtered charging-session data table
   - Search and filtering functionality

2. **Smart Grid Analytics**
   - Visual analytics for EV charging data
   - Energy delivered vs temperature
   - Peak vs non-peak session distribution
   - Charger type distribution
   - Sessions by hour of day
   - Energy distribution by charger type
   - Feature correlation heatmap
   - Interactive filters for charger type, grid status, and peak/non-peak sessions

3. **EV Charge ML Engine**
   - Interactive charging-session risk analysis
   - Scenario selection
   - Custom charging inputs
   - Charger type
   - Charging duration
   - Energy
   - Rated power
   - Initial state of charge
   - Battery temperature
   - Overstay duration
   - Grid status
   - Risk score
   - Risk category
   - Cost and penalty estimation
   - CO₂-related information
   - Explanation of factors contributing to risk
   - Recommended operator actions

### Dataset
The project uses a cleaned EV charging dataset stored as:

`cleaned_ev_dataset.csv`

The dataset contains EV charging-session information used for analysis, visualisations, and the application's prediction/risk-analysis functionality.

### Machine Learning
The trained model is stored as:

`ev_model.pkl`

The project also contains `app.py`, which implements the Streamlit application and integrates the available model/risk-analysis functionality.

Important: Do not claim that the model uses a specific algorithm unless it can be confirmed from the project files. If the README needs to mention the model, describe it simply as the trained EV charging risk model.

### Technologies
Mention the technologies actually relevant to this project:

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Data visualisation libraries used by the application
- Pickle/model file (`ev_model.pkl`)
- GitHub

Do not add technologies that are not required by the project.

### Assessment Tasks
The README should briefly explain that the project was developed as part of the assessment and covers the required tasks, including:

- Data preparation and cleaning
- Data analysis
- Data visualisation
- Machine-learning/risk-analysis functionality
- Streamlit web application development
- UI/UX improvements
- GitHub repository organisation
- Deployment/online demonstration preparation

Do not write a long academic explanation of each task. Keep this section concise but clear.

### Repository Structure
Show the repository structure exactly like this:

```text
POP-Group-Assessment--2/
│
├── app.py
├── cleaned_ev_dataset.csv
├── ev_model.pkl
└── requirements.txt
```

Explain the purpose of each file:

- `app.py` — main Streamlit application
- `cleaned_ev_dataset.csv` — cleaned EV charging dataset
- `ev_model.pkl` — trained model used by the application
- `requirements.txt` — Python dependencies required to run the application

### How to Run
Provide simple Windows-friendly instructions:

1. Clone the repository.
2. Open the project folder in a terminal.
3. Create a virtual environment if required.
4. Install dependencies using:

```bash
pip install -r requirements.txt
```

5. Run the Streamlit application using:

```bash
streamlit run app.py
```

6. Open the local Streamlit URL shown in the terminal.


### Project Objective
Explaining that the application aims to turn EV charging-session data into understandable insights that can support monitoring of campus charging operations, energy usage, charger behaviour, revenue, and potential charging risks.

### Conclusion
Explaining that the project demonstrates how Python, data analytics, visualisation, Streamlit, and machine-learning/risk-analysis techniques can be combined into an interactive EV charging analytics application for a smart-campus environment.
