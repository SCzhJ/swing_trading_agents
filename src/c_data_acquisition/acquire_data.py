from .alpha_vantage_stock import get_stock
from .alpha_vantage_indicator import get_indicator
from .alpha_vantage_fundamentals import get_fundamentals, get_balance_sheet, get_cashflow, get_income_statement
from .alpha_vantage_news import get_news, get_insider_transactions
from .web_search_llm import *
from datetime import datetime, timedelta
import pandas as pd
from io import StringIO
import json

def calculate_vwma(stock_data_csv: str, time_period: int=14) -> str:
    """Calculate Volume Weighted Moving Average (VWMA) from stock data.
    
    Args:
        stock_data_csv: CSV string containing OHLCV data from get_stock function
        time_period: Number of days for the moving average calculation
        
    Returns:
        String containing VWMA values in the format similar to other indicators
    """
    try:
        # Parse CSV data
        df = pd.read_csv(StringIO(stock_data_csv))
        
        # Ensure the dataframe has the necessary columns
        required_columns = ['timestamp', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in stock data")
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # sort by timestamp (newest first)
        df = df.sort_values('timestamp', ascending=True).reset_index(drop=True)
        
        # calculate vwma
        # vwma = σ(price * volume) / σ(volume)
        df['price_volume'] = df['close'] * df['volume']
        df['cumulative_price_volume'] = df['price_volume'].rolling(window=time_period).sum()
        df['cumulative_volume'] = df['volume'].rolling(window=time_period).sum()
        df['vwma'] = df['cumulative_price_volume'] / df['cumulative_volume']

        df = df.sort_values('timestamp', ascending=False).reset_index(drop=True)
        # Prepare the result string
        result_data = []
        for _, row in df.iterrows():
            if pd.notna(row['vwma']):
                date_str = row['timestamp'].strftime('%Y-%m-%d')
                vwma_value = f"{row['vwma']:.6f}"
                result_data.append((date_str, vwma_value))
        
        # Create the output string in the format similar to other indicators
        ind_string = ""
        for date_str, value in result_data:
            ind_string += f"{date_str}: {value}\n"
        
        # Get date range for the header
        start_date = df['timestamp'].min().strftime('%Y-%m-%d')
        end_date = df['timestamp'].max().strftime('%Y-%m-%d')
        
        # VWMA description
        vwma_description = "VWMA: A moving average weighted by volume. Usage: Confirm trends by integrating price action with volume data. Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses."
        
        result_str = (
            f"## VWMA values from {start_date} to {end_date} (Period: {time_period}):\n\n"
            + ind_string
            + "\n\n"
            + vwma_description
        )
        
        return result_str
        
    except Exception as e:
        print(f"Error calculating VWMA: {e}")
        return f"Error calculating VWMA: {str(e)}"

def acquire_technical_data(symbol: str, curr_date: str, look_back_days: int=30):
    """Acquire technical data from Alpha Vantage API.
    
    Args:
        symbol: Stock symbol (e.g., "AAPL")
        start_date: Start date for data acquisition (e.g., "2023-01-01")
        look_back_days: Number of days of data to acquire
        
    Returns:
        None, only write all the data acquired to .txt file
    """

    supported_indicators = {
        "close_50_sma": ("50 SMA", "close"),
        "close_200_sma": ("200 SMA", "close"),
        "close_10_ema": ("10 EMA", "close"),
        "macd": ("MACD", "close"),
        "macds": ("MACD Signal", "close"),
        "macdh": ("MACD Histogram", "close"),
        "rsi": ("RSI", "close"),
        "boll": ("Bollinger Middle", "close"),
        "boll_ub": ("Bollinger Upper Band", "close"),
        "boll_lb": ("Bollinger Lower Band", "close"),
        "atr": ("ATR", None)
    }
    # VWMA will be calculated separately using the stock data

    # Calculate end date based on start date and look_back_days
    start_date = datetime.strptime(curr_date, "%Y-%m-%d") - timedelta(days=look_back_days)
    start_date = start_date.strftime("%Y-%m-%d")
    with open(f"{symbol}_technical_data.txt", "w") as f:
        # Write the Stock Data First
        stock_data = get_stock(symbol, start_date, curr_date)
        # reorder the stock_data
        df = pd.read_csv(StringIO(stock_data))
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp', ascending=True).reset_index(drop=True)
        stock_data = df.to_csv(index=False)
        f.write(f"\"Stock Data from {start_date} to {curr_date}\": "+"{"+"\n\n\"")
        f.write(stock_data)
        f.write("\"\n"+"}"+"\n\n")
        # Write the Technical Data
        for indicator in list(supported_indicators.keys()):
            f.write(f"\"{supported_indicators[indicator][0]} values from {start_date} to {curr_date}\": "+"{"+"\n\n\"")
            indicator_data = get_indicator(symbol, indicator, curr_date, look_back_days)
            f.write(indicator_data)
            f.write("\"\n"+"}"+"\n\n")
        # Calculate and write VWMA data separately
        time_period = 14
        new_start_date = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=time_period)
        stock_data_for_vwma = get_stock(symbol, new_start_date.strftime("%Y-%m-%d"), curr_date)
        vwma_data = calculate_vwma(stock_data_for_vwma, time_period)
        f.write(f"\"VWMA values from {start_date} to {curr_date}\": "+"{"+"\n\n\"")
        f.write(vwma_data)
        f.write("\"\n"+"}"+"\n\n")

def acquire_sentiment_data(symbol: str, curr_date: str, look_back_days: int=30, limit: int=50):
    """Acquire sentiment data from Alpha Vantage API.
    
    Args:
        symbol: Stock symbol (e.g., "AAPL")
        curr_date: Current date for data acquisition (e.g., "2023-01-01")
        look_back_days: Number of days of data to acquire
        limit: Number of news articles to acquire
        
    Returns:
        None, only write all the data acquired to .txt file
    """
    start_date = datetime.strptime(curr_date, "%Y-%m-%d") - timedelta(days=look_back_days)
    start_date = start_date.strftime("%Y-%m-%d")
    sentiment_data = get_news(symbol, start_date, curr_date)
    with open(f"{symbol}_sentiment_data.txt", "w") as f:
        f.write(f"\"Sentiment Data from {start_date} to {curr_date}\": \n\n")
        f.write(sentiment_data)
        f.write("\n\n")

def acquire_news_data(symbol: str, curr_date: str, look_back_days: int=30, limit: int=50):
    """Acquire news data from Alpha Vantage API.
    
    Args:
        symbol: Stock symbol (e.g., "AAPL")
        curr_date: Current date for data acquisition (e.g., "2023-01-01")
        look_back_days: Number of days of data to acquire
        limit: Number of news articles to acquire
        
    Returns:
        None, only write all the data acquired to .txt file
    """
    start_date = datetime.strptime(curr_date, "%Y-%m-%d") - timedelta(days=look_back_days)
    start_date = start_date.strftime("%Y-%m-%d")

    # Get the news data 
    news_data = get_news(symbol, start_date, curr_date, limit)
    print(news_data)
    news_data = json.loads(news_data)

    # Add the relevance score to each news article
    for i in range(len(news_data['feed'])):
        for symbol_dict in news_data['feed'][i]['ticker_sentiment']:
            if symbol_dict['ticker'] == symbol:
                relevance_score = float(symbol_dict['relevance_score'])
                break
        news_data['feed'][i]['relevance_score'] = relevance_score
    
    # Sort the news articles by relevance score in descending order, and only keep the ones with relevance score >= 0.5
    sorted_news_data = sorted(news_data['feed'], key=lambda x: float(x['relevance_score']), reverse=True)
    sorted_news_data = [x for x in sorted_news_data if float(x['relevance_score']) >= 0.5]
    
    with open(f"{symbol}_news_data.txt", "w") as f:
        f.write(f"\"News Data from {start_date} to {curr_date}\": "+"{"+"\n\n\"")
        f.write(json.dumps(sorted_news_data, indent=4))
        f.write("\"\n"+"}"+"\n\n")

def test_acquire_news_data():
    symbol = "CRYPTO:BTC"
    curr_date = "2023-01-04"
    look_back_days = 30
    limit = 50
    acquire_news_data(symbol, curr_date, look_back_days, limit)
def test_acquire_technical_data():
    symbol = "AAPL"
    curr_date = "2023-01-04"
    look_back_days = 30
    acquire_technical_data(symbol, curr_date, look_back_days)
def test_acquire_sentiment_data():
    symbol = "AAPL"
    curr_date = "2023-01-04"
    look_back_days = 30
    acquire_sentiment_data(symbol, curr_date, look_back_days)
def test_calculate_vwma():
    # Example stock data in CSV format (similar to what get_stock returns)
    sample_stock_data = """timestamp,open,high,low,close,adjusted_close,volume,dividend_amount,split_coefficient
    2023-01-31,142.7,144.34,142.28,144.29,142.297408193064,65874459,0.0,1.0
    2023-01-30,144.955,145.55,142.85,143.0,141.025222618395,64015274,0.0,1.0
    2023-01-27,143.155,147.23,143.08,145.93,143.91476039652,70555843,0.0,1.0
    2023-01-26,143.17,144.25,141.9,143.96,141.971965371637,54105068,0.0,1.0
    2023-01-25,140.89,142.43,138.81,141.86,139.900965598919,65799349,0.0,1.0
    2023-01-24,140.305,143.16,140.3,142.53,140.561713145453,66435142,0.0,1.0
    2023-01-23,138.12,143.315,137.9,141.11,139.161322822949,81760313,0.0,1.0
    2023-01-20,135.28,138.02,134.22,137.87,135.966066030756,80223626,0.0,1.0
    2023-01-19,134.08,136.25,133.77,135.27,133.401971074058,58280413,0.0,1.0
    2023-01-18,136.815,138.61,135.03,135.21,133.34279965198,69672800,0.0,1.0
    2023-01-17,134.83,137.29,134.13,135.94,134.062718620591,63646627,0.0,1.0
    2023-01-13,132.03,134.92,131.66,134.76,132.899013986398,57809719,0.0,1.0
    2023-01-12,133.88,134.26,131.44,133.41,131.567656989651,71379648,0.0,1.0
    2023-01-11,131.25,133.51,130.46,133.49,131.646552219087,69458949,0.0,1.0
    2023-01-10,130.26,131.2636,128.12,130.73,128.924666803516,63896155,0.0,1.0
    2023-01-09,130.465,133.41,129.89,130.15,128.352676390098,70790813,0.0,1.0
    2023-01-06,126.01,130.29,124.89,129.62,127.829995495079,87754715,0.0,1.0
    2023-01-05,127.13,127.77,124.76,125.02,123.293519802459,80962708,0.0,1.0
    2023-01-04,126.89,128.6557,125.08,126.36,124.615014895527,89113633,0.0,1.0
    2023-01-03,130.28,130.9,124.17,125.07,123.342829320858,112117471,0.0,1.0"""
    
    # Test with a shorter time period for demonstration
    vwma_result = calculate_vwma(sample_stock_data, time_period=14)
    print("VWMA Calculation Test Result:")
    print(vwma_result)
    return vwma_result

if __name__ == "__main__":
    test_acquire_news_data()
    # test_calculate_vwma()
    # test_acquire_technical_data()
