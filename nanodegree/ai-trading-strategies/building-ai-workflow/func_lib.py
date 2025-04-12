# Import yfinance library
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_hist_prices(start_date = '2000-01-01', end_date = '2024-05-01'):    
    # Define the list of tickers
    sp500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
    
    # Filter out Class B shares that have a '.B' in the ticker name
    sp500_tickers = [ticker for ticker in sp500_tickers if '.B' not in ticker]
    
    # Download historical prices for the list of tickers
    historical_prices = yf.download(sp500_tickers, start=start_date, end=end_date, auto_adjust=False)
    
    # Filter and keep only columns where the first level of the MultiIndex is 'Adj Close'
    historical_prices  = historical_prices.loc[:, historical_prices.columns.get_level_values(0) == 'Adj Close']
    
    # Remove the MultiIndex and keep only the second level
    historical_prices.columns = historical_prices.columns.droplevel(0)   
    
    MIN_REQUIRED_NUM_OBS_PER_TICKER = 100
    
    # Count non-missing values for each ticker
    ticker_counts = historical_prices.count()
    
    # Filter out tickers with fewer than n=MIN_REQUIRED_NUM_OBS_PER_TICKER=100 non-missing values
    valid_tickers_mask = ticker_counts[ticker_counts >= MIN_REQUIRED_NUM_OBS_PER_TICKER].index
    
    # Filter the DataFrame based on valid tickers
    historical_prices = historical_prices[valid_tickers_mask]

    return historical_prices


def compute_returns(historical_prices, list_of_momentums): 
    # Set the forecast horizon (how many days into the future we want to predict)
    forecast_horizon = 1

    # --- TARGET: Compute 1-day forward returns ---
    # Step 1: Calculate percent change over the horizon (from t to t+1)
    f_returns = historical_prices.pct_change(forecast_horizon, fill_method=None)
    
    # Step 2: Shift returns *backward* so that the return from t to t+1 is aligned with time t
    # i.e., the future return becomes the label for today
    f_returns = f_returns.shift(-forecast_horizon)

    # Step 3: Unstack (reshape) into a long-form DataFrame (MultiIndex to 2 columns)
    f_returns = pd.DataFrame(f_returns.unstack())
    
    # Step 4: Rename column to something like "F_1_d_returns" (Forward 1-day return)
    target_col = f"F_{forecast_horizon}_d_returns"
    f_returns.rename(columns={0: target_col}, inplace=True)

    # Initialize total_returns with just the target variable
    total_returns = f_returns

    # --- FEATURES: Compute past returns (momentum) over different lookback periods ---
    for i in list_of_momentums:   
        # Step 1: Calculate past i-day returns at time t (i.e., from t-i to t)
        feature = historical_prices.pct_change(i, fill_method=None)
        
        # Step 2: Reshape and rename the column to something like "5_d_returns"
        feature = pd.DataFrame(feature.unstack())
        feature_col = f"{i}_d_returns"
        feature.rename(columns={0: feature_col}, inplace=True)

        # Step 3: Merge this feature into the total_returns DataFrame
        total_returns = pd.merge(
            total_returns, feature, left_index=True, right_index=True, how='outer'
        )

    # Drop any rows with missing data (NaNs from early periods, missing prices, etc.)
    total_returns.dropna(axis=0, how='any', inplace=True) 

    # Final DataFrame contains:
    # - index: (Ticker, Date)
    # - columns: [F_1_d_returns, 5_d_returns, 10_d_returns, ...]
    return total_returns


def compute_BM_Perf(total_returns):
    # Compute the daily mean of all stocks. This will be our equal weighted benchmark
    daily_mean  = pd.DataFrame(total_returns.loc[:,'F_1_d_returns'].groupby(level='Date').mean())
    daily_mean.rename(columns={'F_1_d_returns':'SP&500'}, inplace=True)
    
    # Convert daily returns to cumulative returns
    cum_returns = pd.DataFrame((daily_mean[['SP&500']]+1).cumprod())
    
    # Plotting the cumulative returns
    cum_returns.plot()
    
    # Customizing the plot
    plt.title('Cumulative Returns Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Cumulative Return', fontsize=14)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend(title_fontsize='13', fontsize='11')
    
    # Display the plot
    plt.show()
    
    # Calculate the number of years in the dataset
    number_of_years = len(daily_mean) / 252  # Assuming 252 trading days in a year
    
    ending_value    = cum_returns['SP&500'].iloc[-1]
    beginning_value = cum_returns['SP&500'].iloc[1]
    
    # Compute the Compound Annual Growth Rate (CAGR)
    ratio = ending_value/beginning_value
    cagr = round((ratio**(1/number_of_years)-1)*100,2)
    print(f'The CAGR is: {cagr}%')
    
    # Compute the Sharpe Ratio by annualizing the daily mean and the daily std
    average_daily_return    = daily_mean[['SP&500']].describe().iloc[1,:] * 252
    stand_dev_dail_return   = daily_mean[['SP&500']].describe().iloc[2,:] * pow(252,1/2)
    
    sharpe  = average_daily_return/stand_dev_dail_return
    
    print(f'Sharpe Ratio of Strategy: {round(sharpe.iloc[0],2)}')
    
    
    #df_daily_mean.rename(columns={target:'Strategy'},inplace=True)
    ann_returns = (pd.DataFrame((daily_mean[['SP&500']]+1).groupby(daily_mean.index.get_level_values(0).year).cumprod())-1)*100
    calendar_returns  = pd.DataFrame(ann_returns['SP&500'].groupby(daily_mean.index.get_level_values(0).year).last())
    
    calendar_returns.plot.bar(rot=30,  legend='top_left')#.opts(multi_level=False) 

    return cum_returns, calendar_returns


def compute_strat_perf(total_returns, cum_returns, calendar_returns, trading_strategy, model_name):    
    # Apply trading strategy to each RSI value
    total_returns['Position'] = total_returns[model_name].transform(trading_strategy)
    # Create Returns for each Trade
    total_returns[f'{model_name}_Return'] = total_returns['F_1_d_returns'] *  total_returns['Position'] 
    
    # Compute the daily mean of all stocks. This will be our equal weighted benchmark
    daily_mean  = pd.DataFrame(total_returns.loc[:,f'{model_name}_Return'].groupby(level='Date').mean())
    
    # Convert daily returns to cumulative returns
    cum_returns.loc[:,f'{model_name}_Return']  = pd.DataFrame((daily_mean[[f'{model_name}_Return']]+1).cumprod())

    # Plotting the cumulative returns
    cum_returns.plot()
    
    # Customizing the plot
    plt.title('Cumulative Returns Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Cumulative Return', fontsize=14)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend(title_fontsize='13', fontsize='11')
    
    # Display the plot
    plt.show()
    
    # Calculate the number of years in the dataset
    number_of_years = len(daily_mean) / 252  # Assuming 252 trading days in a year
    
    ending_value    = cum_returns[f'{model_name}_Return'].iloc[-1]
    beginning_value = cum_returns[f'{model_name}_Return'].iloc[1]
    
    ratio = ending_value/beginning_value
    # Compute the Compound Annual Growth Rate (CAGR)
    cagr = round((ratio**(1/number_of_years)-1)*100,2)
    
    print(f'The CAGR is: {cagr}%')
    
    # Compute the Sharpe Ratio by annualizing the daily mean and the daily std
    average_daily_return  = daily_mean[[f'{model_name}_Return']].describe().iloc[1,:] * 252
    stand_dev_dail_return   = daily_mean[[f'{model_name}_Return']].describe().iloc[2,:] * pow(252,1/2)
    
    # Compute the Sharpe Ratio and print it out
    sharpe  = average_daily_return/stand_dev_dail_return
    
    print(f'Sharpe Ratio of Strategy: {round(sharpe.iloc[0],2)}')
    
    ann_returns = (pd.DataFrame((daily_mean[f'{model_name}_Return']+1).groupby(daily_mean.index.get_level_values(0).year).cumprod())-1)*100
    
    
    calendar_returns.loc[:,f'{model_name}_Return']  = pd.DataFrame(ann_returns[f'{model_name}_Return'].groupby(daily_mean.index.get_level_values(0).year).last())
    
    calendar_returns.plot.bar(rot=30,  legend='top_left')#.opts(multi_level=False) 
    return cum_returns, calendar_returns


def calculate_rsi(returns, window=14):    
    gain = returns[returns>0].dropna().rolling(window=window).mean()
    gain.name = 'gain'
    loss = returns[returns<0].dropna().rolling(window=window).mean()
    loss.name = 'loss'
    returns = pd.merge(returns, gain, left_index=True, right_index=True, how='left')
    returns = pd.merge(returns, loss, left_index=True, right_index=True, how='left')
    returns = returns.ffill()
    returns.dropna(inplace=True)
    ratio = returns['gain']/abs(returns['loss'])
    rsi = 100 - (100 / (1 + ratio))
    return rsi


# Function to find the elbow point
def find_elbow_point(sse, k_range):
    # Normalize the SSE to a 0-1 scale
    sse = np.array(sse)
    sse_normalized = (sse - sse.min()) / (sse.max() - sse.min())

    # Normalize the k values to a 0-1 scale
    k = np.array(k_range)
    k_normalized = (k - k.min()) / (k.max() - k.min())

    # Compute the distances from the line connecting the first and last points
    distances = []
    for i in range(len(k_normalized)):
        p1 = np.array([k_normalized[0], sse_normalized[0]])
        p2 = np.array([k_normalized[-1], sse_normalized[-1]])
        p = np.array([k_normalized[i], sse_normalized[i]])
        dist = np.abs(np.cross(p2-p1, p1-p)) / np.linalg.norm(p2-p1)
        distances.append(dist)
    # Find the index of the maximum curvature
    elbow_idx = np.argmax(distances)
    optimal_k = k_range[elbow_idx]
    print(f"Optimal number of clusters: {optimal_k}")
    return k_range[elbow_idx], distances, k_normalized, sse_normalized


def plot_optimal_cluster_point(sse, k_range):
    optimal_k, distances, k_normalized, sse_normalized = find_elbow_point(sse, k_range)
    plt.figure(figsize=(10, 5))

    # Plotting the first subplot (Elbow Method)
    plt.subplot(1, 2, 1)
    plt.plot(k_range, sse, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Sum of Squared Errors')
    plt.grid(True)
    
    # Plotting the second subplot (Normalized SSE and Distances)
    plt.subplot(1, 2, 2)
    plt.plot(k_normalized, sse_normalized, marker='o', label='Normalized SSE')
    plt.plot([0, 1], [sse_normalized[0], sse_normalized[-1]], 'r--', label='Reference Line')
    
    # Calculating and plotting lines to extend to the reference line
    for i in range(len(k_normalized)):
        p1 = np.array([0, sse_normalized[0]])  # Start of the reference line
        p2 = np.array([1, sse_normalized[-1]])  # End of the reference line
        p = np.array([k_normalized[i], sse_normalized[i]])
        
        # Vector between p1 and p2
        vec_p1p2 = p2 - p1
        
        # Vector between p1 and p
        vec_p1p = p - p1
        
        # Project vec_p1p onto vec_p1p2
        scalar_proj = np.dot(vec_p1p, vec_p1p2) / np.dot(vec_p1p2, vec_p1p2)
        projected_point = p1 + scalar_proj * vec_p1p2
        
        plt.plot([k_normalized[i], projected_point[0]], [sse_normalized[i], projected_point[1]], 'k:', alpha=0.7)
    
    # Highlighting the elbow point
    elbow_index = np.argmax(sse_normalized)
    plt.scatter(k_normalized[elbow_index], sse_normalized[elbow_index], color='red', label='Elbow Point')
    
    plt.title('Normalized SSE and Errors')
    plt.xlabel('Normalized Number of Clusters')
    plt.ylabel('Normalized Sum of Squared Errors')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()