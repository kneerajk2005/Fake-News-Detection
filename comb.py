import pandas as pd

# Load cleaned lists
blacklist_df = pd.read_csv("data/blacklist_domains_cleaned.csv")
whitelist_df = pd.read_csv("data/whitelist_domains_cleaned.csv")

# Add category column
blacklist_df['category'] = 'blacklist'
whitelist_df['category'] = 'whitelist'

# Combine into one DataFrame
domain_df = pd.concat([blacklist_df, whitelist_df], ignore_index=True)

# Save combined list
domain_df.to_csv("data/domain_list.csv", index=False)
print("Combined domain list saved as domain_list.csv")
