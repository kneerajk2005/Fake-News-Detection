# ====================================
# verify_news.py
# ====================================
import torch
import pandas as pd
import tldextract
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import os

# ====================================
# 1. Load cleaned domain list
# ====================================
domain_csv = os.path.join("data", "domain_list_cleaned.csv")

# Load CSV
domain_df = pd.read_csv(domain_csv)

# Create domain map directly
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
model_path = "saved_model_distilbert"  # adjust path to your model folder
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
    return "real" if prediction == 1 else "fake"

# ====================================
# 4. Metadata checks
# ====================================
def check_metadata(article):
    suspicious = []
    if len(article.strip().split()) < 5:
        suspicious.append("Too short")
    if article.isupper():
        suspicious.append("All Caps")
    return "suspect" if suspicious else "clean"

# ====================================
# 5. Final verification function
# ====================================
def verify_news(url, article_text):
    domain_status = check_domain(url)
    ml_prediction = run_ml_model(article_text)
    meta_status = check_metadata(article_text)

    if domain_status == "blacklist":
        return "Fake News (Blacklisted Domain)"
    elif domain_status == "whitelist":
        if ml_prediction == "fake":
            return " Suspicious (Trusted Source but ML flagged as fake)"
        else:
            return " Real News (Trusted Source)"
    else:  # Unknown domain
        if ml_prediction == "fake" or meta_status == "suspect":
            return " Fake News (Unknown Domain + Flags)"
        else:
            return " Likely Real (Unknown Domain + Clean Metadata)"

# ====================================
# 6. Example usage
# ====================================
if __name__ == "__main__":
    examples = [
        ("https://www.bbc.com/news/world-asia-india-12345",
         "India successfully launched its new satellite today."),
        ("http://fakenews.com/story/999",
         "ALIENS HAVE LANDED IN DELHI AND ARE TAKING OVER PARLIAMENT"),
        ("https://randomblog.net/article",
         "Breaking: Free gold for everyone in Hyderabad market.")
    ]

    for url, text in examples:
        print(f"URL: {url}\nResult: {verify_news(url, text)}\n")
