 🧠 Gradient Descent Visual App

A user-friendly Streamlit web app to explore and visualize **Gradient Descent** for both regression and classification tasks.

---

## 🚀 Key Features

- 📊 Upload your dataset (`.csv`)
- 🧹 Preprocess your data (handle missing values, encoding, scaling)
- ⚙️ Tune hyperparameters (learning rate, number of iterations)
- 📈 Visualize training progress (learning curve)
- 🤖 Train regression or classification models
- 🔍 Make predictions on new inputs
- 💾 Export results (trained model, preprocessing steps, performance metrics)

---

## 🛠️ Built With

- **Python 3.11+**
- [Streamlit](https://streamlit.io/) — Web app framework
- [Pandas](https://pandas.pydata.org/) — Data manipulation
- [NumPy](https://numpy.org/) — Numerical operations
- [Plotly](https://plotly.com/python/) — Interactive visualizations
- [Seaborn](https://seaborn.pydata.org/) / Matplotlib — Charts & plots
- [Scikit-learn](https://scikit-learn.org/) — ML preprocessing & metrics

---

## 📋 Requirements

```txt
streamlit>=1.28.0
streamlit-option-menu>=0.3.6
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.15.0
scikit-learn>=1.3.0
seaborn>=0.12.0

---

## 💻 Installation

To run this application locally:

```bash
# 1. Clone the repository
git clone https://github.com/wijdanehr08/gradient-descent-app.git
cd gradient-descent-app

# 2. Create and activate a virtual environment
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py

# The app will open in your browser at:
# http://localhost:8501

