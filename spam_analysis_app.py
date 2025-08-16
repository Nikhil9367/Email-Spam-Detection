import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import re
import string
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="SMS Spam Detection & Analysis",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #ff7f0e;
    margin-top: 2rem;
    margin-bottom: 1rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# Load and cache data
@st.cache_data
def load_data():
    """Load the spam dataset"""
    try:
        df = pd.read_csv('spam.csv')
        return df
    except FileNotFoundError:
        st.error("Dataset file not found. Please ensure spam.csv is in the correct directory.")
        return None

# Text preprocessing function
def preprocess_text(text):
    """Clean and preprocess text data"""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

# Train ML models
@st.cache_data
def train_models(df):
    """Train multiple ML models for spam detection"""
    # Preprocess text
    df['processed_message'] = df['Message'].apply(preprocess_text)
    
    # Prepare features and target
    X = df['processed_message']
    y = df['Category']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create pipelines for different models
    models = {
        'Naive Bayes': Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
            ('classifier', MultinomialNB())
        ]),
        'Logistic Regression': Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
            ('classifier', LogisticRegression(random_state=42))
        ]),
        'Random Forest': Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
    }
    
    # Train models and store results
    trained_models = {}
    model_scores = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        trained_models[name] = model
        model_scores[name] = {
            'accuracy': accuracy,
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    return trained_models, model_scores, X_test, y_test

# Main app
def main():
    st.markdown('<h1 class="main-header">📱 SMS Spam Detection & Analysis</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["📊 Data Analysis", "🤖 ML Prediction", "📈 Model Performance"])
    
    if page == "📊 Data Analysis":
        show_data_analysis(df)
    elif page == "🤖 ML Prediction":
        show_prediction_page(df)
    elif page == "📈 Model Performance":
        show_model_performance(df)

def show_data_analysis(df):
    st.markdown('<h2 class="sub-header">📊 Dataset Overview</h2>', unsafe_allow_html=True)
    
    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Messages", len(df))
    with col2:
        spam_count = len(df[df['Category'] == 'spam'])
        st.metric("Spam Messages", spam_count)
    with col3:
        ham_count = len(df[df['Category'] == 'ham'])
        st.metric("Ham Messages", ham_count)
    with col4:
        spam_percentage = (spam_count / len(df)) * 100
        st.metric("Spam Percentage", f"{spam_percentage:.1f}%")
    
    # Display sample data
    st.markdown('<h3 class="sub-header">Sample Data</h3>', unsafe_allow_html=True)
    st.dataframe(df.head(10))
    
    # Visualizations
    st.markdown('<h2 class="sub-header">📈 Data Visualizations</h2>', unsafe_allow_html=True)
    
    # 1. Distribution of Categories (Pie Chart)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Category Distribution")
        category_counts = df['Category'].value_counts()
        fig_pie = px.pie(values=category_counts.values, names=category_counts.index, 
                        title="Ham vs Spam Distribution",
                        color_discrete_map={'ham': '#2E8B57', 'spam': '#DC143C'})
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("2. Message Length Distribution")
        df['message_length'] = df['Message'].str.len()
        fig_hist = px.histogram(df, x='message_length', color='Category', 
                               title="Message Length Distribution",
                               color_discrete_map={'ham': '#2E8B57', 'spam': '#DC143C'},
                               nbins=50)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # 3. Word Count Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("3. Word Count Analysis")
        df['word_count'] = df['Message'].str.split().str.len()
        fig_box = px.box(df, x='Category', y='word_count', 
                        title="Word Count by Category",
                        color='Category',
                        color_discrete_map={'ham': '#2E8B57', 'spam': '#DC143C'})
        st.plotly_chart(fig_box, use_container_width=True)
    
    with col2:
        st.subheader("4. Average Message Statistics")
        stats_df = df.groupby('Category').agg({
            'message_length': 'mean',
            'word_count': 'mean'
        }).round(2)
        
        fig_bar = go.Figure(data=[
            go.Bar(name='Avg Message Length', x=stats_df.index, y=stats_df['message_length'], 
                  marker_color=['#2E8B57', '#DC143C']),
            go.Bar(name='Avg Word Count', x=stats_df.index, y=stats_df['word_count'], 
                  marker_color=['#90EE90', '#FFB6C1'])
        ])
        fig_bar.update_layout(title="Average Statistics by Category", barmode='group')
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # 5. Word Clouds
    st.subheader("5. Word Clouds")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Ham Messages Word Cloud**")
        ham_text = ' '.join(df[df['Category'] == 'ham']['Message'].astype(str))
        ham_text = preprocess_text(ham_text)
        
        if ham_text:
            wordcloud_ham = WordCloud(width=400, height=300, background_color='white', 
                                     colormap='Greens').generate(ham_text)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(wordcloud_ham, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
    
    with col2:
        st.write("**Spam Messages Word Cloud**")
        spam_text = ' '.join(df[df['Category'] == 'spam']['Message'].astype(str))
        spam_text = preprocess_text(spam_text)
        
        if spam_text:
            wordcloud_spam = WordCloud(width=400, height=300, background_color='white', 
                                      colormap='Reds').generate(spam_text)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(wordcloud_spam, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

def show_prediction_page(df):
    st.markdown('<h2 class="sub-header">🤖 Spam Detection Prediction</h2>', unsafe_allow_html=True)
    
    # Train models
    with st.spinner("Training ML models..."):
        trained_models, model_scores, X_test, y_test = train_models(df)
    
    # Model selection
    st.subheader("Select Model")
    selected_model = st.selectbox("Choose a model for prediction:", list(trained_models.keys()))
    
    # Display model accuracy
    accuracy = model_scores[selected_model]['accuracy']
    st.success(f"Selected Model: **{selected_model}** | Accuracy: **{accuracy:.3f}**")
    
    # User input
    st.subheader("Enter Message for Prediction")
    user_input = st.text_area("Type your message here:", 
                             placeholder="Enter a text message to check if it's spam or ham...")
    
    if st.button("🔍 Predict", type="primary"):
        if user_input.strip():
            # Preprocess and predict
            processed_input = preprocess_text(user_input)
            model = trained_models[selected_model]
            prediction = model.predict([processed_input])[0]
            prediction_proba = model.predict_proba([processed_input])[0]
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 'spam':
                    st.error(f"🚨 **SPAM DETECTED**")
                    st.write(f"Confidence: {prediction_proba[1]:.3f}")
                else:
                    st.success(f"✅ **HAM (Not Spam)**")
                    st.write(f"Confidence: {prediction_proba[0]:.3f}")
            
            with col2:
                # Probability chart
                fig = go.Figure(data=[
                    go.Bar(x=['Ham', 'Spam'], y=prediction_proba, 
                          marker_color=['#2E8B57', '#DC143C'])
                ])
                fig.update_layout(title="Prediction Probabilities", 
                                yaxis_title="Probability",
                                showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please enter a message to predict.")
    
    # Example messages
    st.subheader("Try These Examples")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Example Ham Message"):
            st.text_area("Example:", "Hey, are we still meeting for lunch tomorrow?", key="ham_example")
    
    with col2:
        if st.button("Example Spam Message"):
            st.text_area("Example:", "URGENT! You have won £1000! Call now to claim your prize!", key="spam_example")

def show_model_performance(df):
    st.markdown('<h2 class="sub-header">📈 Model Performance Analysis</h2>', unsafe_allow_html=True)
    
    # Train models
    with st.spinner("Training and evaluating models..."):
        trained_models, model_scores, X_test, y_test = train_models(df)
    
    # Model comparison
    st.subheader("Model Accuracy Comparison")
    
    # Create accuracy comparison chart
    model_names = list(model_scores.keys())
    accuracies = [model_scores[name]['accuracy'] for name in model_names]
    
    fig_comparison = px.bar(x=model_names, y=accuracies, 
                           title="Model Accuracy Comparison",
                           color=accuracies,
                           color_continuous_scale='viridis')
    fig_comparison.update_layout(xaxis_title="Models", yaxis_title="Accuracy")
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Detailed performance metrics
    st.subheader("Detailed Performance Metrics")
    
    selected_model_perf = st.selectbox("Select model for detailed analysis:", model_names)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Classification report
        y_test_model = model_scores[selected_model_perf]['y_test']
        y_pred_model = model_scores[selected_model_perf]['y_pred']
        
        st.write("**Classification Report:**")
        report = classification_report(y_test_model, y_pred_model, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(3))
    
    with col2:
        # Confusion Matrix
        st.write("**Confusion Matrix:**")
        cm = confusion_matrix(y_test_model, y_pred_model)
        
        fig_cm = px.imshow(cm, 
                          text_auto=True, 
                          aspect="auto",
                          title="Confusion Matrix",
                          labels=dict(x="Predicted", y="Actual"),
                          x=['Ham', 'Spam'],
                          y=['Ham', 'Spam'])
        st.plotly_chart(fig_cm, use_container_width=True)
    
    # Model insights
    st.subheader("Key Insights")
    
    best_model = max(model_scores.keys(), key=lambda x: model_scores[x]['accuracy'])
    best_accuracy = model_scores[best_model]['accuracy']
    
    insights = f"""
    - **Best Performing Model**: {best_model} with {best_accuracy:.3f} accuracy
    - **Dataset Size**: {len(df)} messages
    - **Spam Ratio**: {(len(df[df['Category'] == 'spam']) / len(df) * 100):.1f}%
    - **Feature Engineering**: TF-IDF vectorization with 5000 features
    """
    
    st.markdown(insights)

if __name__ == "__main__":
    main()