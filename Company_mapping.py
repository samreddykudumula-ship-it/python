import pandas as pd
import re
from collections import Counter

def map_companies_to_tweets():
    """company mapping for UK tweets"""
    
    print("🏢 COMPANY MAPPING SYSTEM")
    print("="*40)
    
    # Load tweets
    df = pd.read_csv('uk_tweets_medium_confidence.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"✅ Loaded {len(df)} tweets")
    
    # Company patterns - focused on what we actually found
    companies = {
        'Tesla': [r'\$TSLA\b', r'\bTesla\b'],
        'BP': [r'\$BP\b', r'\bBP\b'],
        'AstraZeneca': [r'\$AZN\b', r'\bAstraZeneca\b'],
        'Unilever': [r'\$UL\b', r'\bUnilever\b'],
        'Barclays': [r'\$BCS\b', r'\bBarclays\b'],
        'Shell': [r'\$SHELL\b', r'\bShell\b', r'\$SHEL\b'],
        'Vodafone': [r'\$VOD\b', r'\bVodafone\b'],
        'GSK': [r'\$GSK\b', r'\bGSK\b', r'\bGlaxoSmithKline\b']
    }
    
    # Stock ticker mapping
    tickers = {
        'Tesla': {'uk': None, 'us': 'TSLA', 'sector': 'Automotive'},
        'BP': {'uk': 'BP.L', 'us': 'BP', 'sector': 'Energy'},
        'AstraZeneca': {'uk': 'AZN.L', 'us': 'AZN', 'sector': 'Healthcare'},
        'Unilever': {'uk': 'ULVR.L', 'us': 'UL', 'sector': 'Consumer'},
        'Barclays': {'uk': 'BARC.L', 'us': 'BCS', 'sector': 'Banking'},
        'Shell': {'uk': 'SHEL.L', 'us': 'SHEL', 'sector': 'Energy'},
        'Vodafone': {'uk': 'VOD.L', 'us': 'VOD', 'sector': 'Telecom'},
        'GSK': {'uk': 'GSK.L', 'us': 'GSK', 'sector': 'Healthcare'}
    }
    
    # Find company mentions
    company_tweets = []
    company_counts = Counter()
    
    for idx, row in df.iterrows():
        text = row['Tweet']
        for company, patterns in companies.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    company_tweets.append({
                        'tweet_index': idx,
                        'date': row['Date'],
                        'text': text,
                        'company': company,
                        'sector': tickers[company]['sector'],
                        'uk_ticker': tickers[company]['uk'],
                        'us_ticker': tickers[company]['us']
                    })
                    company_counts[company] += 1
                    break
    
    # Create results dataframe
    results_df = pd.DataFrame(company_tweets)
    
    # Save results
    results_df.to_csv('company_tweet_mapping.csv', index=False)
    
    # Print summary
    print(f"\n📊 RESULTS:")
    for company, count in company_counts.most_common():
        uk_tick = tickers[company]['uk'] or 'N/A'
        us_tick = tickers[company]['us']
        print(f"   {company}: {count} tweets ({uk_tick}/{us_tick})")
    
    print(f"\n✅ Saved {len(results_df)} company-tweet mappings")
    print("✅ Ready for stock data collection")
    
    return results_df, company_counts

if __name__ == "__main__":
    results, counts = map_companies_to_tweets()