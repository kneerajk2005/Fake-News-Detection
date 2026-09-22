# ====================================
# app.py
# ====================================
from flask import Flask, render_template, request, jsonify
import torch
import pandas as pd
import tldextract
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import os
import requests
import bs4
from bs4 import BeautifulSoup


# Initialize Flask app
app = Flask(__name__)

# ====================================
# 1. Load cleaned domain list
# ====================================
domain_csv = os.path.join("data", "domain_list_cleaned.csv")
domain_df = pd.read_csv(domain_csv)
domain_df['domain'] = domain_df['domain'].str.strip().str.lower()
domain_map = dict(zip(domain_df['domain'], domain_df['category']))

# ====================================
# 2. Domain check function
# ====================================
def check_domain(url):
    domain = tldextract.extract(url).top_domain_under_public_suffix.lower().strip()
    return domain_map.get(domain, "unknown")

# ====================================
# 3. Load trained DistilBERT model
# ====================================
model_path = "saved_model_distilbert"  # Adjust to your trained model folder
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model.to(device)
model.eval()

def run_ml_model(text):
    """Return ML prediction: 'real' or 'fake'"""
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
    # Adjust based on training labels
    return "real" if prediction == 1 else "fake"

# ====================================
# 4. Metadata checks
# ====================================
import requests
from bs4 import BeautifulSoup

def check_metadata(url):
    """Extract and evaluate metadata from a webpage."""
    suspicious = []
    metadata = {}

    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract all meta tags
        for meta in soup.find_all("meta"):
            if meta.get("name") and meta.get("content"):
                metadata[meta.get("name").lower()] = meta["content"].strip()
            elif meta.get("property") and meta.get("content"):
                metadata[meta.get("property").lower()] = meta["content"].strip()

        # Simple checks
        if "author" not in metadata:
            suspicious.append("No Author")
        if "description" not in metadata:
            suspicious.append("No Description")
        if "keywords" not in metadata:
            suspicious.append("No Keywords")
        if len(metadata.get("description", "").split()) < 5:
            suspicious.append("Description Too Short")

    except Exception as e:
        suspicious.append(f"Metadata fetch error: {e}")

    return {
        "status": "suspect" if suspicious else "clean",
        "suspicious_flags": suspicious,
        "metadata": metadata
    }


# ====================================
# 5. Final verification function
# ====================================
def verify_news(url, article_text):
    domain_status = check_domain(url)
    ml_prediction = run_ml_model(article_text)
    meta_result = check_metadata(url)  
    meta_status = meta_result["status"]

    if domain_status == "blacklist":
        return "❌ Fake News (Blacklisted Domain)"
    elif domain_status == "whitelist":
        if ml_prediction == "fake":
            return "⚠️ Suspicious (Trusted Source but ML flagged as fake)"
        else:
            return "✅ Real News (Trusted Source)"
    else:  # Unknown domain
        if ml_prediction == "fake" or meta_status == "suspect":
            return f"❌ Fake News (Unknown Domain + Flags: {meta_result['suspicious_flags']})"
        else:
            return "✅ Likely Real (Unknown Domain + Clean Metadata)"

# ====================================
# 6. Flask routes
# ====================================
@app.route('/')
def home():
    return render_template('index.html')  # Your HTML form page

@app.route('/verify', methods=['POST'])
def verify():
    data = request.form
    url = data.get('url', '').strip()
    article = data.get('article', '').strip()

    if not url or not article:
        return jsonify({"error": "URL and Article text are required."})

    result = verify_news(url, article)
    return jsonify({"url": url, "verdict": result})

# ====================================
# 7. Run Flask app
# ====================================
if __name__ == '__main__':
    app.run(debug=True)
    