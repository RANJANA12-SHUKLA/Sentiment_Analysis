import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from pathlib import Path
import os
import time
import pandas as pd # Import pandas here for use in the app

# --- Configuration (Must be consistent with your local setup) ---
# NOTE: Replace 'fine_tuned_distilbert_lora' with your actual local model path
MODEL_DIR = Path("./fine_tuned_distilbert_lora") 
MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 2
LABEL_MAP = {0: "Negative", 1: "Positive"}
COLOR_MAP = {"Positive": "#10b981", "Negative": "#ef4444"} # Green and Red colors
# REMOVED: TEMPERATURE = 1.2 # Temperature for confidence smoothing

# --- 1. Model Loading (Uses Streamlit's caching for speed) ---

@st.cache_resource
def load_sentiment_model(model_path: Path):
    """Loads the base model and the LoRA adapters."""
    
    # st.info(f"Attempting to load LoRA model from {model_path}. This runs only once.")
    
    # 1. Load the tokenizer from the local path
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        st.error(f"Error loading tokenizer. Ensure config files are in {model_path}. Error: {e}")
        return None, None
        
    # 2. Load the base model (DistilBERT)
    try:
        # Check for CUDA availability for faster inference if deploying outside of Streamlit Cloud
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=NUM_LABELS
        ).to(device)
    except Exception as e:
        st.error(f"Error loading base model {MODEL_NAME}. Check connectivity. Error: {e}")
        return None, None

    # 3. Load the LoRA adapters onto the base model
    try:
        model = PeftModel.from_pretrained(base_model, str(model_path)).to(device)
        model.eval()
        # st.success("✨ Model loaded successfully! Ready for inference.")
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Error loading LoRA adapters. Did you unzip the adapter files (.safetensors, .config) into the '{model_path.name}' folder? Error: {e}")
        return None, None, None

# Load the model and tokenizer globally
model, tokenizer, device = load_sentiment_model(MODEL_DIR)

# --- 2. Inference Function ---

def predict_sentiment(text: str, model, tokenizer, device):
    """Tokenizes, predicts, and returns probability scores."""
    if not model or not tokenizer:
        return {"error": "Model not loaded."}, 0, 0, 0

    # Tokenize input and move to the correct device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256).to(device)

    # Get raw logits
    with torch.no_grad():
        outputs = model(**inputs)
        
    logits = outputs.logits
    
    # MODIFIED: Use standard Softmax (no temperature scaling)
    probabilities = F.softmax(logits, dim=-1)
    
    # Extract confidence for the predicted class
    confidence, predicted_index = torch.max(probabilities, dim=1)
    confidence_score = confidence.item()
    predicted_label = LABEL_MAP[predicted_index.item()]
    
    # Extract both positive and negative probabilities for the chart
    prob_list = probabilities.squeeze().tolist()
    
    # Ensure list order matches LABEL_MAP (0: Negative, 1: Positive)
    neg_score = prob_list[0]
    pos_score = prob_list[1]
    
    return predicted_label, confidence_score, neg_score, pos_score


# --- 3. Streamlit UI Layout ---

st.set_page_config(
    page_title="LoRA Sentiment Demo",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎬 LoRA Fine-Tuned Sentiment Analyzer Demo")
st.markdown("### Powered by DistilBERT + PEFT (Parameter-Efficient Fine-Tuning)")

# Create two columns for layout
col1, col2 = st.columns([2, 1])

# Initialize session state for tracking analysis result
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False

with col1:
    st.header("Enter a Movie Review")
    
    review_text = st.text_area(
        "Paste a short review here:",
        value="This movie was an absolute triumph—the direction was masterful, and the lead actor gave the performance of a lifetime. Highly recommend!",
        height=200,
        key="review_input"
    )

    if st.button("Analyze Sentiment", type="primary"):
        if model and tokenizer:
            with st.spinner('Analyzing sentiment...'):
                time.sleep(0.5) # Add a small delay for presentation effect
                label, confidence, neg_score, pos_score = predict_sentiment(review_text, model, tokenizer, device)
            
            st.session_state.result_label = label
            st.session_state.result_confidence = confidence
            st.session_state.result_neg = neg_score
            st.session_state.result_pos = pos_score
            st.session_state.analyzed = True
            st.rerun()

        else:
            st.error("Model is not loaded. Please check your `fine_tuned_distilbert_lora` folder and dependencies.")
            st.session_state.analyzed = False
            
    # REMOVED: Test Cases for Self-Healing Logic section
    
    

with col2:
    st.header("Results and Confidence ")
    
    if st.session_state.get("analyzed", False):
        label = st.session_state.result_label
        confidence = st.session_state.result_confidence
        neg_score = st.session_state.result_neg
        pos_score = st.session_state.result_pos

        # Display Prediction
        st.subheader("Final Prediction:")
        
        emoji = "😊" if label == "Positive" else "😔"
        color = COLOR_MAP.get(label, "gray")
        
        # Display the result using markdown for color and size
        st.markdown(f"**<span style='color:{color}; font-size: 32px;'>{emoji} {label}</span>**", unsafe_allow_html=True)
        
        # Display Confidence
        st.metric(
            # MODIFIED: Label to reflect raw confidence (higher is better)
            label="Confidence Score (Model Certainty)",
            value=f"{confidence * 100:.2f}%",
            delta_color="off"
        )
        
        # Display the confidence breakdown - FIX IS HERE
        st.markdown("---")
        st.text("Score Breakdown:")
        
        # FIX: Create a DataFrame with a single row where columns are the scores
        df_chart = pd.DataFrame({
            'Negative': [neg_score],
            'Positive': [pos_score]
        })
        
        # Use st.bar_chart without setting index and specify colors explicitly based on column order
        st.bar_chart(
            df_chart, 
            height=200, 
            color=['#ef4444', '#10b981'] # Red for Negative, Green for Positive
        )
        
        # REMOVED: Note about Temperature Scaling/Fallback
    else:
        st.info("Hit 'Analyze Sentiment' to see the model's prediction and confidence.")

# --- 4. Seminar Explanation Section (Sidebar) ---

with st.sidebar:
    st.title("Model Architecture")
    st.markdown("This demo uses **DistilBERT** fine-tuned with **LoRA**.")

    st.subheader("Efficiency (Why it's fast)")
    st.markdown(
        """
        - **LoRA (Low-Rank Adaptation):** Only trained ~1% of the model's parameters, keeping the base model frozen.
        - **Result:** Faster training, smaller file size, and instant loading time in this application.
        """
    )
    
    st.subheader("Technical Detail")
    st.code(
        """
@st.cache_resource
def load_sentiment_model(model_path):
    base_model = AutoModelForSequenceClassification.from_pretrained(...)
    model = PeftModel.from_pretrained(base_model, model_path)
    return model
        """,
        language='python'
    )

    # st.warning("For this demo to work locally, ensure you have the `fine_tuned_distilbert_lora` folder in the same directory as this script.")
