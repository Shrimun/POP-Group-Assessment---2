<img width="1024" height="434" alt="image" src="https://github.com/user-attachments/assets/1558dc63-70be-4a83-85bd-239941969fcd" />

# Taylor’s University Smart Campus – EV Analytics & Machine Learning Dashboard

## Project Overview

This project is an interactive **Streamlit web application** developed for analysing Electric Vehicle (EV) charging activity in a smart-campus environment.

The application combines **data analytics, visualisation, and machine-learning/risk-analysis functionality** to provide insights into EV charging behaviour, energy usage, charger performance, revenue, and charging-session risks.

The project was developed as part of the **POP Group Assessment – 2**.

---

## Main Sections

### 1. EV Taylor's Core Metrics

This section provides an overview of the EV charging data through an interactive dashboard.

It includes:

- Total energy delivered
- Average session duration
- Peak revenue
- Total charging sessions
- Energy delivered by day
- Charging sessions by charger type
- Peak vs non-peak revenue
- Filtered charging-session data
- Search and filtering functionality

### 2. Smart Grid Analytics

This section provides visual analysis of EV charging activity and grid-related information.

It includes:

- Energy delivered vs temperature
- Peak vs non-peak session distribution
- Charger type distribution
- Sessions by hour of day
- Energy distribution by charger type
- Feature correlation analysis
- Interactive filters for charger type, grid status, and peak/non-peak sessions

### 3. EV Charge ML Engine

This section provides an interactive charging-session risk-analysis interface.

Users can adjust charging-session inputs such as:

- Charger type
- Charging duration
- Energy
- Rated power
- Initial state of charge
- Battery temperature
- Overstay duration
- Grid status

The application provides:

- Risk score
- Risk category
- Charging-session assessment
- Cost and penalty estimation
- CO₂-related information
- Factors contributing to the risk
- Recommended operator actions

---

## Dataset

The project uses the cleaned EV charging dataset:

`cleaned_ev_dataset.csv`

The dataset is used for the application's data analysis, visualisations, and charging-session risk-analysis functionality.

---

## Machine Learning

The trained model is stored in:

`ev_model.pkl`

The application loads this model where required for the EV charging risk-analysis functionality.

The project also contains `app.py`, which implements the Streamlit application and connects the dataset, model, interface, and analysis functionality.

> The README does not specify a particular machine-learning algorithm unless it is confirmed directly from the project implementation.

---

## Technologies

The project uses the following technologies:

- **Python**
- **Streamlit**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Data visualisation libraries used by the application**
- **Pickle** for the saved model
- **GitHub** for version control and project submission

---

## Assessment Tasks

This project covers the required assessment work, including:

- Data preparation and cleaning
- Data analysis
- Data visualisation
- Machine-learning/risk-analysis functionality
- Streamlit application development
- User interface improvements
- GitHub repository organisation
- Application deployment and online demonstration

The **Task 2 Machine Learning Workflow & Model Evaluation** work is also included in the repository.

---

## Repository Structure

```text
POP-Group-Assessment--2/
│
├── .devcontainer/
│   └── Development container configuration
│
├── assets/
│   ├── Taylor's University logo
│   ├── Application images
│   └── Other visual assets used by the Streamlit interface
│
├── README.md
│   └── Project documentation
│
├── Task - 2 Machine Learning Workflow & Model Evaluation...
│   └── Task 2 machine-learning workflow and evaluation work (**Use Google Colab**)
│
├── app.py
│   └── Main Streamlit application
│
├── cleaned_ev_dataset.csv
│   └── Cleaned EV charging dataset
│
├── ev_model.pkl
│   └── Trained EV charging model
│
└── requirements.txt
    └── Python dependencies required to run the application
```

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Shrimun/POP-Group-Assessment--2.git
```

### 2. Open the project folder

```bash
cd POP-Group-Assessment--2
```

### 3. Install the required dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

### 5. Open the application

Streamlit will provide a local URL in the terminal. Open that URL in your browser to access the application.

---

## Project Objective

The objective of this project is to transform EV charging-session data into understandable and interactive insights for a smart-campus environment.

The application helps users explore:

- EV charging activity
- Energy usage
- Charger behaviour
- Revenue
- Grid-related patterns
- Charging-session risks
- Recommended operational actions

---

## Conclusion

This project demonstrates how **Python, data analytics, data visualisation, Streamlit, and machine-learning/risk-analysis techniques** can be combined to create an interactive EV charging analytics application.

The final application provides a practical smart-campus interface for exploring EV charging data and evaluating potential charging-session risks.
