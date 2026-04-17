# SMS Spam Detection & Analysis App

A comprehensive Streamlit web application for analyzing SMS spam data and predicting whether new messages are spam or ham (legitimate) using machine learning models.

## Features

### 📊 Data Analysis
- Dataset overview with key statistics
- Interactive visualizations including:
  - Category distribution (pie chart)
  - Message length distribution
  - Word count analysis
  - Average statistics comparison
  - Word clouds for spam and ham messages

### 🤖 ML Prediction
- Three trained machine learning models:
  - Naive Bayes
  - Logistic Regression
  - Random Forest
- Real-time spam detection for user input
- Prediction confidence scores
- Example messages to test

### 📈 Model Performance
- Model accuracy comparison
- Detailed classification reports
- Confusion matrices
- Performance insights

## Installation

1. Make sure you have Python 3.7+ installed
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Navigate to the project directory:
   ```bash
   cd "e:/Study material/MCA/sem 2/python/project/Nikhil"
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run spam_analysis_app.py
   ```

3. Open your web browser and go to the URL shown in the terminal (usually `http://localhost:8501`)

## Dataset

The app uses the `spam.csv` file which contains:
- **Category**: Label indicating 'ham' (legitimate) or 'spam'
- **Message**: The actual SMS text content
- **Total Messages**: 5,574 SMS messages

## Machine Learning Models

The app implements three different algorithms:

1. **Naive Bayes**: Fast and effective for text classification
2. **Logistic Regression**: Linear model with good interpretability
3. **Random Forest**: Ensemble method for robust predictions

All models use TF-IDF (Term Frequency-Inverse Document Frequency) vectorization for feature extraction.

## App Structure

- `spam_analysis_app.py`: Main Streamlit application
- `spam.csv`: Dataset file
- `requirements.txt`: Python dependencies
- `README.md`: This documentation file

## Navigation

Use the sidebar to navigate between different sections:
- **📊 Data Analysis**: Explore the dataset with various visualizations
- **🤖 ML Prediction**: Test spam detection on custom messages
- **📈 Model Performance**: Compare model accuracies and view detailed metrics

## Example Usage

1. **Data Analysis**: View comprehensive statistics and visualizations of the spam dataset
2. **Prediction**: Enter a message like "URGENT! You have won £1000!" to see it classified as spam
3. **Performance**: Compare different models and see which performs best on the test data

## Technical Details

- **Text Preprocessing**: Converts to lowercase, removes punctuation and numbers
- **Feature Extraction**: TF-IDF with 5000 maximum features
- **Train/Test Split**: 80/20 split with stratification
- **Evaluation Metrics**: Accuracy, precision, recall, F1-score

## Requirements

- Python 3.7+
- Streamlit 1.28.1
- Pandas 2.0.3
- Scikit-learn 1.3.0
- Plotly 5.15.0
- WordCloud 1.9.2
- Other dependencies listed in requirements.txt

---
> 🛡️ **Automated Security Scan:** 27-March-2026 | **Status:** Warning: 1 potential issues detected ⚠️


---
> 🛡️ **Security Status:** Scan Completed ✅ | **Last Audit:** 17-April-2026
