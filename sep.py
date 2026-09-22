import pandas as pd

# Load MBFC dataset (replace with your file name)
df = pd.read_csv("media-bias-scrubbed-results.csv")

print("Columns available:", df.columns)
print("Dataset shape:", df.shape)

# ✅ Ensure proper column names exist
# The Gist version usually has: 'source', 'bias_rating', 'factual_reporting_rating'
df = df.rename(columns=lambda x: x.strip().lower())

# --- Whitelist: Highly factual sources ---
whitelist = df[df['factual_reporting_rating'].isin(['HIGH', 'VERY HIGH'])]

# --- Blacklist: Low factual or Questionable sources ---
blacklist = df[df['factual_reporting_rating'].isin(['LOW', 'VERY LOW'])]

print("Whitelist count:", whitelist.shape[0])
print("Blacklist count:", blacklist.shape[0])

# Save results
whitelist.to_csv("whitelist_domains.csv", index=False)
blacklist.to_csv("blacklist_domains.csv", index=False)

print("✅ Whitelist and Blacklist saved as CSV files.")
