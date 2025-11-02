# 🎬 Interactive Sentiment Analysis Demo (Seminar Project)

Welcome to the **live demo** of a highly accurate sentiment analysis application, built to demonstrate the power of **Parameter-Efficient Fine-Tuning (PEFT)**.

This app uses a **custom fine-tuned transformer model** to determine whether a movie review is **Positive** or **Negative** — all in real time.

---

## 🔗 Live Application Link

👉 Try it yourself on Streamlit:  
https://ranjana12-shukla-sentiment-analysis-streamlit-app-obvu6s.streamlit.app/
---

## 🧠 Model & Fine-Tuning Explained

### 1. Fine-Tuning with LoRA (Low-Rank Adaptation)

Instead of retraining the entire **268M-parameter DistilBERT model**, we used **LoRA**, a PEFT method that makes training **faster, lighter, and more efficient**.

- **Efficiency:** LoRA freezes the original DistilBERT weights and only trains a few small adapter layers.  
- **Speed & Size:** This reduces the trainable parameters by **over 99%**, letting us reach high accuracy in minutes while keeping the deployed model compact and quick to load.

### 2. Deployment Efficiency

The model and tokenizer are loaded directly in the Streamlit app via the `peft` and `transformers` libraries — enabling **instant sentiment analysis** once the app starts.

---

## 💻 How to Use the App

1. **Paste a Review:** Enter any text (e.g., a movie or product review).  
2. **Analyze:** Click **“Analyze Sentiment.”**  
3. **View Results:** Instantly see:
   - ✅ **Final Prediction:** Positive or Negative (with an emoji).  
   - 📊 **Confidence Score:** The model’s certainty.  
   - 📈 **Probability Chart:** Visual breakdown of Positive vs. Negative probabilities.

---


## 🚀 Getting Started Locally

You can also run this project on your local machine.

```bash
# 1️⃣ Clone the repository
git clone https://github.com/RANJANA12-SHUKLA/Sentiment_Analysis

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Run the Streamlit app
streamlit run streamlit_app.py
