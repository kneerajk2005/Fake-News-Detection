import pandas as pd
import tldextract
import os

# Input CSV
input_file = os.path.join("data", "domain_list.csv")

# Output cleaned CSV
output_file = os.path.join("data", "domain_list_cleaned.csv")

# Load CSV
domain_df = pd.read_csv(input_file)

# Extract registered domain from URL
domain_df['registered_domain'] = domain_df['url'].apply(lambda x: tldextract.extract(x).registered_domain.lower())

# Keep only domain + category and remove duplicates
cleaned_df = domain_df[['registered_domain', 'category']].drop_duplicates()
cleaned_df.columns = ['domain', 'category']

# Save cleaned CSV
cleaned_df.to_csv(output_file, index=False)
print(f"Cleaned domain list saved to: {output_file}")

# Optional: create domain map in memory
domain_map = dict(zip(cleaned_df['domain'], cleaned_df['category']))
print("Sample domain map:", list(domain_map.items())[:5])
