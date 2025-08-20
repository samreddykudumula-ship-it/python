import yfinance as yf
import pandas as pd
from datetime import datetime

def collect_stock_data():
    """stock data collection for sentiment analysis"""
    
    print("📈 STOCK DATA COLLECTION")
    print("="*40)
    
    # Company tickers (based on our mapping results)
    tickers = {
        'Tesla': ['TSLA'],
        'BP': ['BP.L', 'BP'], 
        'AstraZeneca': ['AZN.L', 'AZN'],
        'Unilever': ['ULVR.L', 'UL'],
        'Barclays': ['BARC.L', 'BCS'],
        'Shell': ['SHEL.L', 'SHEL'],
        'Vodafone': ['VOD.L', 'VOD'],
        'GSK': ['GSK.L', 'GSK']
    }
    
    # Date range (tweet period + buffer)
    start_date = '2021-09-01'
    end_date = '2022-10-29'
    
    all_stock_data = []
    
    # Collect data for each company
    for company, ticker_list in tickers.items():
        print(f"\n📊 Collecting {company}...")
        
        for ticker in ticker_list:
            try:
                # Get stock data
                stock = yf.Ticker(ticker)
                data = stock.history(start=start_date, end=end_date)
                
                if len(data) > 0:
                    # Add metadata
                    data = data.reset_index()
                    data['Company'] = company
                    data['Ticker'] = ticker
                    data['Exchange'] = 'UK' if '.L' in ticker else 'US'
                    
                    # Calculate basic features
                    data['Daily_Return'] = data['Close'].pct_change()
                    data['Next_Day_Return'] = data['Daily_Return'].shift(-1)
                    
                    all_stock_data.append(data)
                    print(f"   ✅ {ticker}: {len(data)} days")
                
            except Exception as e:
                print(f"   ❌ {ticker}: Failed ({str(e)[:30]}...)")
    
    # Combine all data
    master_df = pd.concat(all_stock_data, ignore_index=True)
    master_df.to_csv('stock_data_complete.csv', index=False)
    
    print(f"\n✅ Collected {len(master_df)} stock records")
    print(f"✅ Companies: {master_df['Company'].nunique()}")
    print("✅ Ready for tweet-stock alignment")
    
    return master_df

def align_tweets_with_stocks():
    """Align tweet data with stock data"""
    
    print("\n🔗 ALIGNING TWEETS WITH STOCKS")
    print("="*40)
    
    # Load data
    tweets = pd.read_csv('company_tweet_mapping.csv')
    stocks = pd.read_csv('stock_data_complete.csv')
    
    print(f"📨 Loaded {len(tweets)} tweet mappings")
    print(f"📈 Loaded {len(stocks)} stock records")
    
    # Debug: Check what's actually in the date columns
    print(f"\n🔍 DEBUGGING DATE FORMATS:")
    print(f"Tweet date column type: {tweets['date'].dtype}")
    print(f"First 3 tweet dates: {tweets['date'].head(3).tolist()}")
    print(f"Stock Date column type: {stocks['Date'].dtype}")
    print(f"First 3 stock dates: {stocks['Date'].head(3).tolist()}")
    
    # Simple string-based date extraction instead of pandas datetime
    print(f"\n🛠️ CONVERTING DATES...")
    
    # For tweets - extract just YYYY-MM-DD part if it's a string
    def extract_date_string(date_str):
        if pd.isna(date_str):
            return None
        date_str = str(date_str)
        # Extract YYYY-MM-DD pattern
        import re
        match = re.search(r'(\d{4}-\d{2}-\d{2})', date_str)
        if match:
            return match.group(1)
        return None
    
    tweets['date_clean'] = tweets['date'].apply(extract_date_string)
    stocks['date_clean'] = stocks['Date'].apply(extract_date_string)
    
    # Remove rows with no valid dates
    tweets = tweets.dropna(subset=['date_clean'])
    stocks = stocks.dropna(subset=['date_clean'])
    
    print(f"✅ Cleaned dates:")
    print(f"   Tweets: {len(tweets)} valid dates")
    print(f"   Stocks: {len(stocks)} valid dates")
    print(f"   Sample tweet date: {tweets['date_clean'].iloc[0] if len(tweets) > 0 else 'None'}")
    print(f"   Sample stock date: {stocks['date_clean'].iloc[0] if len(stocks) > 0 else 'None'}")
    
    # Merge on company and date
    aligned = pd.merge(
        tweets, 
        stocks, 
        left_on=['company', 'date_clean'], 
        right_on=['Company', 'date_clean'],
        how='inner'
    )
    
    # Save aligned data
    aligned.to_csv('tweets_stocks_aligned.csv', index=False)
    
    print(f"\n✅ Aligned {len(aligned)} tweet-stock pairs")
    
    # Show breakdown by company
    if len(aligned) > 0:
        company_counts = aligned['company'].value_counts()
        print("📊 Alignment by company:")
        for company, count in company_counts.items():
            print(f"   {company}: {count} pairs")
    else:
        print("❌ No alignments found - checking date overlap...")
        print(f"Tweet date range: {tweets['date_clean'].min()} to {tweets['date_clean'].max()}")
        print(f"Stock date range: {stocks['date_clean'].min()} to {stocks['date_clean'].max()}")
    
    print("✅ Ready for sentiment analysis")
    
    return aligned

if __name__ == "__main__":
    stock_data = collect_stock_data()
    aligned_data = align_tweets_with_stocks()