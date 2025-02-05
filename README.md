# Customer Feedback Sentiment Analysis

## Project Overview
This project aims to analyze customer feedback and classify it as either **positive (1)** or **negative (0)** using **LightGBM**. The model is trained on textual customer reviews and predicts sentiment based on the given comments.

## Features
- **Machine Learning Model:** Utilizes **LightGBM** for high-performance classification.
- **Accuracy:** Achieved **90% test accuracy** and **93% training accuracy**.
- **Variance:** Maintains a variance of **90%**, ensuring balanced generalization.
- **Streamlit Frontend:** A user-friendly interface to enter customer feedback and get sentiment predictions.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/customer-feedback-analysis.git
   cd customer-feedback-analysis
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Enter a customer feedback comment in the text box.
3. Click "Predict" to see the sentiment classification.

## Model Performance
- **Train Accuracy:** 93%
- **Test Accuracy:** 90%
- **Variance:** 90%

## Technologies Used
- **Python**
- **LightGBM**
- **Streamlit** (for frontend UI)
- **Pandas & NumPy** (for data preprocessing)
- **Sklearn** (for model evaluation)

## Future Enhancements
- Improve model performance using more advanced NLP techniques.
- Deploy the app on a cloud platform.
- Add real-time feedback analysis.


