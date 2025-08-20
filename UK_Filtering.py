import pandas as pd
import re
import numpy as np
from collections import Counter

def uk_filter(df):
    
    
    print("Applying UK filtering...")
    print(f"Original dataset size: {len(df)}")
    
    # Create a copy to work with
    df_copy = df.copy()
    
    #  UK FILTERING PATTERNS
    # We're being much more inclusive to capture more tweets
    
    # 1. DIRECT UK INDICATORS (High confidence)
    direct_uk_patterns = [
        r'£',                           # Pound symbol
        r'\bGBP\b',                    # Great British Pound
        r'\bFTSE\b',                   # FTSE index
        r'\bLSE\b',                    # London Stock Exchange
        r'\bLondon Stock Exchange\b',   # Full name
        r'\bBank of England\b',         # UK central bank
        r'\bBoE\b',                    # Bank of England abbreviation
        r'\bUK\b',                     # United Kingdom
        r'\bBritain\b',                # Britain
        r'\bBritish\b',                # British
        r'\bLondon\b',                 # London (financial context)
        r'\bEngland\b',                # England
        r'\bScotland\b',               # Scotland
        r'\bWales\b'                   # Wales
    ]
    
    # 2. UK COMPANIES (Including international mentions)
    uk_companies = [
        # Major FTSE companies - including variations and US ticker equivalents
        r'\bBP\b',              # BP plc
        r'\bBP\.\b',            # BP.
        r'\$BP\b',              # $BP (US ticker)
        
        r'\bShell\b',           # Shell
        r'\bSHELL\b',           # SHELL
        r'\$SHELL\b',           # $SHELL
        r'\bRDS\.A\b',          # Royal Dutch Shell A
        r'\bRDS\.B\b',          # Royal Dutch Shell B
        r'\$RDS\b',             # $RDS
        r'\bRoyal Dutch Shell\b',
        
        r'\bAstraZeneca\b',     # AstraZeneca
        r'\bAZN\b',             # AZN ticker
        r'\$AZN\b',             # $AZN
        
        r'\bVodafone\b',        # Vodafone
        r'\bVOD\b',             # VOD ticker
        r'\$VOD\b',             # $VOD
        
        r'\bGSK\b',             # GlaxoSmithKline
        r'\$GSK\b',             # $GSK
        r'\bGlaxoSmithKline\b',
        
        r'\bHSBC\b',            # HSBC
        r'\$HSBC\b',            # $HSBC
        
        r'\bBarclays\b',        # Barclays
        r'\bBARC\b',            # BARC ticker
        r'\$BCS\b',             # $BCS (US ticker)
        
        r'\bLloyds\b',          # Lloyds
        r'\bLLOY\b',            # LLOY ticker
        r'\$LYG\b',             # $LYG (US ticker)
        r'\bLloyds Banking\b',
        
        r'\bTesco\b',           # Tesco
        r'\bTSCO\b',            # TSCO ticker
        r'\$TSCO\b',            # $TSCO
        
        r'\bUnilever\b',        # Unilever
        r'\$UL\b',              # $UL (US ticker)
        r'\$UN\b',              # $UN (US ticker)
        
        r'\bBT Group\b',        # BT Group
        r'\bBritish Telecom\b', # British Telecom
        r'\$BT\b',              # $BT
        
        r'\bRio Tinto\b',       # Rio Tinto
        r'\$RIO\b',             # $RIO
        
        r'\bBritish American Tobacco\b',  # BAT
        r'\$BTI\b',             # $BTI (US ticker)
        
        r'\bPrudential\b',      # Prudential
        r'\$PUK\b',             # $PUK (US ticker)
        
        r'\bStandard Chartered\b',  # Standard Chartered
        r'\$SCBFF\b',           # $SCBFF (US ticker)
        
        r'\bRolls.?Royce\b',    # Rolls Royce (with or without hyphen)
        r'\$RYCEY\b'            # $RYCEY (US ticker)
    ]
    
    # 3. BROADER CONTEXT TERMS (Lower confidence but still relevant)
    context_patterns = [
        r'\bBrexit\b',                      # Brexit context
        r'\bEurope\b',                      # European context
        r'\bEuropean\b',                    # European mentions
        r'\binternational\b',               # International context
        r'\bglobal\b',                      # Global context
        r'\boverseas\b',                    # Overseas investment
        r'\bforeign\b',                     # Foreign market mentions
        r'\bECB\b',                         # European Central Bank
        r'\bEuropean Central Bank\b'        # ECB full name
    ]
    
    # 4. UK STOCK TICKER PATTERNS
    ticker_patterns = [
        r'\$[A-Z]{2,5}\.L\b',              # Tickers ending in .L (London)
        r'\b[A-Z]{2,5}\.L\b'               # Without $ symbol
    ]
    
    # APPLY FILTERING WITH SCORING SYSTEM
    df_copy['uk_score'] = 0
    df_copy['uk_matches'] = ''
    
    text_column = 'Tweet'  # Adjust if your column name is different
    
    print("Applying filtering patterns...")
    
    # Score tweets based on different pattern categories
    for i, pattern_group in enumerate([
        ('Direct UK', direct_uk_patterns, 3),       # High score for direct UK terms
        ('UK Companies', uk_companies, 2),          # Medium score for UK companies
        ('Context', context_patterns, 1),           # Low score for context terms
        ('Tickers', ticker_patterns, 2)             # Medium score for UK tickers
    ]):
        
        category_name, patterns, score_weight = pattern_group
        combined_pattern = '|'.join(patterns)
        
        try:
            mask = df_copy[text_column].str.contains(combined_pattern, case=False, na=False, regex=True)
            df_copy.loc[mask, 'uk_score'] += score_weight
            
            # Track which patterns matched
            matches = df_copy.loc[mask, text_column].str.findall(combined_pattern, flags=re.IGNORECASE)
            df_copy.loc[mask, 'uk_matches'] += f"{category_name}: {matches.str.join(', ')}; "
            
            print(f"  {category_name}: {mask.sum()} tweets matched")
            
        except Exception as e:
            print(f"  Error in {category_name} patterns: {e}")
            continue
    
    # FILTER BASED ON SCORE THRESHOLDS
    # We'll create different confidence levels
    
    # High confidence: Score >= 3 (strong UK indicators)
    high_confidence = df_copy[df_copy['uk_score'] >= 3]
    
    # Medium confidence: Score >= 2 (likely UK relevant)
    medium_confidence = df_copy[df_copy['uk_score'] >= 2]
    
    # Low confidence: Score >= 1 (possibly UK relevant)
    low_confidence = df_copy[df_copy['uk_score'] >= 1]
    
    print(f"\nFiltering Results:")
    print(f"  High confidence (score >= 3): {len(high_confidence)} tweets")
    print(f"  Medium confidence (score >= 2): {len(medium_confidence)} tweets")
    print(f"  Low confidence (score >= 1): {len(low_confidence)} tweets")
    
    return {
        'high_confidence': high_confidence,
        'medium_confidence': medium_confidence, 
        'low_confidence': low_confidence,
        'all_scored': df_copy[df_copy['uk_score'] > 0]
    }

def analyze_results(filtered_results):
    """Analyze the filtering results"""
    
    print("\n" + "="*60)
    print("DETAILED ANALYSIS OF FILTERED RESULTS")
    print("="*60)
    
    for confidence_level, df in filtered_results.items():
        if confidence_level == 'all_scored':
            continue
            
        print(f"\n{confidence_level.upper()} TWEETS:")
        print(f"Count: {len(df)}")
        
        if len(df) > 0:
            print("Score distribution:")
            score_dist = df['uk_score'].value_counts().sort_index()
            for score, count in score_dist.items():
                print(f"  Score {score}: {count} tweets")
            
            print(f"\nSample tweets (showing top 3):")
            sample = df.nlargest(3, 'uk_score')
            for i, (_, row) in enumerate(sample.iterrows()):
                print(f"  {i+1}. [Score: {row['uk_score']}]")
                print(f"     Text: {row['Tweet'][:100]}...")
                print(f"     Matches: {row['uk_matches'][:100]}...")
                print()

def save_results(filtered_results, base_filename="uk_tweets"):
    """Save the filtered results to CSV files"""
    
    print(f"\nSaving results to CSV files...")
    
    for confidence_level, df in filtered_results.items():
        if len(df) > 0:
            filename = f"{base_filename}_{confidence_level}.csv"
            
            # Select relevant columns for export
            export_columns = ['Date', 'Tweet', 'Stock Name', 'Company Name', 'uk_score']
            
            # Only include columns that exist
            available_columns = [col for col in export_columns if col in df.columns]
            
            df[available_columns].to_csv(filename, index=False)
            print(f"  Saved {len(df)} tweets to {filename}")
    
    print("All files saved successfully!")

def main():
    """Main function to run the UK filtering"""
    
    print(" UK FILTERING FOR STOCK TWEETS")
    print("="*50)
    
    # STEP 1: Load your dataset
    print("Loading dataset...")
    try:
        # Adjust the file path to your dataset
        df = pd.read_csv('stock_tweets.csv')  # Change this to your file path
        print(f"Dataset loaded successfully: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
    except FileNotFoundError:
        print("Error: Could not find 'stock_tweets.csv'")
        print("Please make sure the file path is correct")
        print("Current working directory files:")
        import os
        print(os.listdir('.'))
        return
    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # STEP 2: Apply UK filtering
    print(f"\nApplying UK filtering...")
    filtered_results = uk_filter(df)
    
    # STEP 3: Analyze results
    analyze_results(filtered_results)
    
    # STEP 4: Save results
    save_results(filtered_results)
    
    # STEP 5: Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    medium_conf_count = len(filtered_results['medium_confidence'])
    
    if medium_conf_count >= 200:
        print("✅ EXCELLENT: You have sufficient data for analysis!")
        print(f"   Recommended: Use medium confidence tweets ({medium_conf_count} tweets)")
        
    elif medium_conf_count >= 100:
        print("✅ GOOD: You have reasonable data for analysis")
        print(f"   Recommended: Use medium confidence tweets ({medium_conf_count} tweets)")
        
    elif medium_conf_count >= 50:
        print("⚠️ MODERATE: Limited but workable data")
        print(f"   Recommended: Combine medium + low confidence tweets")
        
    else:
        print("❌ LOW: Very limited data found")
        print("   Recommended: Use all confidence levels")
        print("   Consider expanding filtering criteria")
    
    print(f"\nNext steps:")
    print(f"1. Review the saved CSV files")
    print(f"2. Choose your confidence level based on data quality vs quantity")
    print(f"3. Proceed with company identification and sentiment analysis")
    print(f"4. Collect corresponding stock price data for identified companies")
    
    return filtered_results

# Run the script
if __name__ == "__main__":
    results = main()