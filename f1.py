import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import yfinance as yf
from datetime import datetime, timedelta

def discrete_delta_hedging_simulator(S0, K, T, mu, sigma, r, N, num_paths=10000, option_type='call'):
    """
    Discrete Delta Hedging Simulator for European options
    """
    np.random.seed(42)
    
    dt = T / N
    replication_errors = np.zeros(num_paths)
    hedge_portfolio_values = np.zeros(num_paths)
    
    for path in range(num_paths):
        # Generate stock price path
        Z = np.random.standard_normal(N)
        stock_prices = np.zeros(N + 1)
        stock_prices[0] = S0
        
        for i in range(1, N + 1):
            stock_prices[i] = stock_prices[i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[i-1])
        
        # Initialize hedge portfolio
        initial_option_price = black_scholes_price(S0, K, T, r, sigma, option_type)
        cash_account = initial_option_price
        stock_held = 0
        
        # Discrete delta hedging
        for i in range(N):
            current_time = i * dt
            time_to_maturity = T - current_time
            current_stock_price = stock_prices[i]
            
            # Calculate Black-Scholes delta
            if time_to_maturity > 1e-10:
                d1 = (np.log(current_stock_price / K) + (r + 0.5 * sigma**2) * time_to_maturity) / (sigma * np.sqrt(time_to_maturity))
                if option_type == 'call':
                    delta = norm.cdf(d1)
                else:
                    delta = norm.cdf(d1) - 1
            else:
                if option_type == 'call':
                    delta = 1.0 if current_stock_price > K else 0.0
                else:
                    delta = -1.0 if current_stock_price < K else 0.0
            
            # Rebalance portfolio
            stock_difference = delta - stock_held
            cash_account -= stock_difference * current_stock_price
            stock_held = delta
            
            # Earn interest on cash account
            cash_account *= np.exp(r * dt)
        
        # Final settlement at maturity
        final_stock_price = stock_prices[-1]
        if option_type == 'call':
            option_payoff = max(final_stock_price - K, 0)
        else:
            option_payoff = max(K - final_stock_price, 0)
        
        final_portfolio_value = stock_held * final_stock_price + cash_account
        replication_error = final_portfolio_value - option_payoff
        
        replication_errors[path] = replication_error
        hedge_portfolio_values[path] = final_portfolio_value
    
    mean_error = np.mean(replication_errors)
    std_error = np.std(replication_errors)
    
    return mean_error, std_error, replication_errors, hedge_portfolio_values

def black_scholes_price(S0, K, T, r, sigma, option_type='call'):
    """Calculate Black-Scholes option price"""
    if T <= 0:
        if option_type == 'call':
            return max(S0 - K, 0)
        else:
            return max(K - S0, 0)
    
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    
    return price

def analyze_hedging_frequency(S0, K, T, mu, sigma, r, N_list, num_paths=10000):
    """Analyze how hedging frequency affects replication error"""
    means = []
    stds = []
    
    for N in N_list:
        mean_error, std_error, _, _ = discrete_delta_hedging_simulator(
            S0, K, T, mu, sigma, r, N, num_paths
        )
        means.append(mean_error)
        stds.append(std_error)
        print(f"N={N:3d}, Δt={T/N:.4f}, Mean Error: {mean_error:.6f}, Std Error: {std_error:.6f}")
    
    return means, stds

def analyze_drift_dependence(S0, K, T, sigma, r, N, mu_list, num_paths=5000):
    """Analyze how replication error depends on drift rate mu"""
    means = []
    stds = []
    
    for mu in mu_list:
        mean_error, std_error, _, _ = discrete_delta_hedging_simulator(
            S0, K, T, mu, sigma, r, N, num_paths
        )
        means.append(mean_error)
        stds.append(std_error)
        print(f"μ={mu:.3f}, Mean Error: {mean_error:.6f}, Std: {std_error:.6f}")
    
    return means, stds

def estimate_convergence_rate(N_list, variances):
    """
    Estimate convergence rate of variance with respect to Δt
    Theoretical expectation: Variance ∝ Δt^α, where α ≈ 1
    """
    dt_list = [1/N for N in N_list]  # Using 1/N as proxy for Δt (since T=1)
    
    # Log-log regression: log(Variance) = α * log(Δt) + constant
    X = np.log(dt_list).reshape(-1, 1)
    y = np.log(variances)
    
    model = LinearRegression()
    model.fit(X, y)
    
    alpha = model.coef_[0]
    r_squared = model.score(X, y)
    
    print(f"\n=== Convergence Rate Analysis ===")
    print(f"Estimated convergence rate α: {alpha:.4f}")
    print(f"R-squared: {r_squared:.4f}")
    print(f"Theoretical expectation: α ≈ 1.0")
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.loglog(dt_list, variances, 'bo-', label='Simulated Variance')
    
    # Plot theoretical line with estimated slope
    theoretical_variances = np.exp(model.predict(X))
    plt.loglog(dt_list, theoretical_variances, 'r--', label=f'Fit (α={alpha:.3f})')
    
    plt.xlabel('Hedging Interval Δt (log scale)')
    plt.ylabel('Variance of Replication Error (log scale)')
    plt.title('Variance Convergence vs Hedging Interval')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.show()
    
    return alpha, r_squared

def ci_grid_over_KT(S0, mu, sigma, r, N, Ks, Ts, num_paths=1000, z_score=1.96, iv_df=None):
    """
    Analyze confidence intervals for different strikes and maturities
    iv_df: optional DataFrame with columns [Strike, Maturity, Implied_Vol]
    """
    results = []
    iv_lookup = None
    if iv_df is not None:
        iv_lookup = iv_df.copy()
        iv_lookup['Strike_key'] = iv_lookup['Strike'].astype(float)
        iv_lookup['Maturity_key'] = iv_lookup['Maturity'].astype(float).round(8)
    
    for K in Ks:
        for T in Ts:
            # Skip invalid combinations
            if T <= 0:
                continue
            
            # Determine sigma for this (K, T) combination
            this_sigma = sigma
            if iv_lookup is not None:
                keyK = float(K)
                keyT = round(float(T), 8)
                hit = iv_lookup[(iv_lookup['Strike_key'] == keyK) & (iv_lookup['Maturity_key'] == keyT)]
                if len(hit) > 0 and not np.isnan(hit['Implied_Vol'].values[0]):
                    this_sigma = float(hit['Implied_Vol'].values[0])
                
            mean_error, std_error, _, _ = discrete_delta_hedging_simulator(
                S0, K, T, mu, this_sigma, r, N, num_paths
            )
            bs_price = black_scholes_price(S0, K, T, r, this_sigma, 'call')
            
            # Calculate confidence interval
            ci_lower = bs_price - z_score * std_error
            ci_upper = bs_price + z_score * std_error
            ci_width = ci_upper - ci_lower
            
            # Determine moneyness
            moneyness = "ATM" if abs(S0 - K) < 2 else ("ITM" if S0 > K else "OTM")
            
            results.append({
                'Strike': K,
                'Maturity': T,
                'Moneyness': moneyness,
                'Sigma_Used': this_sigma,
                'BS_Price': bs_price,
                'Mean_Error': mean_error,
                'Std_Error': std_error,
                'CI_Lower': ci_lower,
                'CI_Upper': ci_upper,
                'CI_Width': ci_width,
                'CI_Width_Pct': (ci_width / bs_price * 100) if bs_price > 0 else 0
            })
    
    return pd.DataFrame(results)

def analyze_zero_one_frequency(S0, K, T, mu, sigma, r, N_list, num_paths=1000):
    """Analyze how zero/one strategy error depends on hedging frequency"""
    means = []
    stds = []
    
    for N in N_list:
        mean_error, std_error, _ = zero_one_strategy(
            S0, K, T, mu, sigma, r, N, num_paths
        )
        means.append(mean_error)
        stds.append(std_error)
        print(f"Zero/One - N={N:3d}, Δt={T/N:.4f}, Mean Error: {mean_error:.6f}, Std: {std_error:.6f}")
    
    return means, stds

def analyze_zero_one_drift(S0, K, T, sigma, r, N, mu_list, num_paths=1000):
    """Analyze how zero/one strategy error depends on drift rate"""
    means = []
    stds = []
    
    for mu in mu_list:
        mean_error, std_error, _ = zero_one_strategy(
            S0, K, T, mu, sigma, r, N, num_paths
        )
        means.append(mean_error)
        stds.append(std_error)
        print(f"Zero/One - μ={mu:.3f}, Mean Error: {mean_error:.6f}, Std: {std_error:.6f}")
    
    return means, stds

def zero_one_strategy(S0, K, T, mu, sigma, r, N, num_paths=10000):
    """
    Zero/One strategy alternative to delta hedging
    """
    np.random.seed(42)
    
    dt = T / N
    replication_errors = np.zeros(num_paths)
    
    for path in range(num_paths):
        # Generate stock price path
        Z = np.random.standard_normal(N)
        stock_prices = np.zeros(N + 1)
        stock_prices[0] = S0
        
        for i in range(1, N + 1):
            stock_prices[i] = stock_prices[i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[i-1])
        
        # Implement zero/one strategy
        shares_held = 0
        cash_account = 0
        previous_price = stock_prices[0]
        
        for i in range(1, N + 1):
            current_price = stock_prices[i]
            
            # Check for crossing events
            if shares_held == 0 and previous_price <= K and current_price > K:
                cash_account -= current_price
                shares_held = 1
            elif shares_held == 1 and previous_price >= K and current_price < K:
                cash_account += current_price
                shares_held = 0
            
            previous_price = current_price
            
            if i < N:
                cash_account *= np.exp(r * dt)
        
        # Final settlement
        final_price = stock_prices[-1]
        option_payoff = max(final_price - K, 0)
        portfolio_value = shares_held * final_price + cash_account
        replication_error = portfolio_value - option_payoff
        
        replication_errors[path] = replication_error
    
    return np.mean(replication_errors), np.std(replication_errors), replication_errors

def plot_comprehensive_results(N_list, delta_means_freq, delta_stds_freq, zero_one_means_freq, zero_one_stds_freq, 
                             mu_list, delta_means_drift, delta_stds_drift, 
                             zero_one_means_drift, zero_one_stds_drift):
    """Plot comprehensive results comparing both strategies"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    dt_list = [1/N for N in N_list]  # T=1 assumed
    
    # Row 1: Frequency dependence
    # Delta hedging mean error vs Δt
    axes[0,0].semilogx(dt_list, delta_means_freq, 'bo-', label='Delta Hedging', markersize=8)
    axes[0,0].set_xlabel('Hedging Interval Δt')
    axes[0,0].set_ylabel('Mean Error')
    axes[0,0].set_title('Mean Error vs Δt (Delta Hedging)')
    axes[0,0].grid(True, which="both", ls="-")
    axes[0,0].legend()
    
    # Delta hedging variance vs Δt
    axes[0,1].loglog(dt_list, np.array(delta_stds_freq)**2, 'bo-', label='Delta Hedging', markersize=8)
    axes[0,1].set_xlabel('Hedging Interval Δt')
    axes[0,1].set_ylabel('Variance')
    axes[0,1].set_title('Variance vs Δt (Delta Hedging)')
    axes[0,1].grid(True, which="both", ls="-")
    axes[0,1].legend()
    
    # Comparison of standard deviations
    axes[0,2].semilogx(dt_list, delta_stds_freq, 'bo-', label='Delta Hedging', markersize=8)
    axes[0,2].semilogx(dt_list, zero_one_stds_freq, 'ro-', label='Zero/One Strategy', markersize=8)
    axes[0,2].set_xlabel('Hedging Interval Δt')
    axes[0,2].set_ylabel('Standard Deviation')
    axes[0,2].set_title('Standard Deviation Comparison')
    axes[0,2].grid(True, which="both", ls="-")
    axes[0,2].legend()
    
    # Row 2: Drift dependence
    # Delta hedging vs drift
    axes[1,0].plot(mu_list, delta_means_drift, 'bo-', label='Mean Error', markersize=8)
    axes[1,0].set_xlabel('Drift Rate μ')
    axes[1,0].set_ylabel('Mean Error')
    axes[1,0].set_title('Mean Error vs μ (Delta Hedging)')
    axes[1,0].grid(True)
    axes[1,0].legend()
    
    # Zero/One vs drift
    axes[1,1].plot(mu_list, zero_one_means_drift, 'ro-', label='Mean Error', markersize=8)
    axes[1,1].set_xlabel('Drift Rate μ')
    axes[1,1].set_ylabel('Mean Error')
    axes[1,1].set_title('Mean Error vs μ (Zero/One Strategy)')
    axes[1,1].grid(True)
    axes[1,1].legend()
    
    # Standard deviation comparison vs drift
    axes[1,2].plot(mu_list, delta_stds_drift, 'bo-', label='Delta Hedging', markersize=8)
    axes[1,2].plot(mu_list, zero_one_stds_drift, 'ro-', label='Zero/One Strategy', markersize=8)
    axes[1,2].set_xlabel('Drift Rate μ')
    axes[1,2].set_ylabel('Standard Deviation')
    axes[1,2].set_title('Standard Deviation vs μ')
    axes[1,2].grid(True)
    axes[1,2].legend()
    
    plt.tight_layout()


def run_scenario_analysis():
    """Comprehensive analysis across various market environments"""
    
    print("=== COMPREHENSIVE SCENARIO ANALYSIS ===")
    
    # Base parameters
    base_params = {
        'S0': 100, 'K': 100, 'T': 1, 
        'mu': 0.1, 'sigma': 0.3, 'r': 0.04, 'N': 4
    }
    
    # Scenario 1: Volatility changes
    print("\n--- Volatility Scenarios ---")
    vol_results = []
    for sigma in [0.15, 0.3, 0.5, 0.8]:
        params = base_params.copy()
        params['sigma'] = sigma
        mean_err, std_err, _, _ = discrete_delta_hedging_simulator(**params, num_paths=1000)
        vol_results.append({
            'Sigma': sigma,
            'Mean_Error': mean_err,
            'Std_Error': std_err,
            'BS_Price': black_scholes_price(params['S0'], params['K'], params['T'], params['r'], sigma)
        })
        print(f"σ={sigma}: Mean Error={mean_err:.4f}, Std={std_err:.4f}")
    
    # Scenario 2: Interest rate changes
    print("\n--- Interest Rate Scenarios ---")
    rate_results = []
    for r in [0.01, 0.04, 0.08, 0.15]:
        params = base_params.copy()
        params['r'] = r
        mean_err, std_err, _, _ = discrete_delta_hedging_simulator(**params, num_paths=1000)
        rate_results.append({
            'Rate': r,
            'Mean_Error': mean_err,
            'Std_Error': std_err,
            'BS_Price': black_scholes_price(params['S0'], params['K'], params['T'], r, params['sigma'])
        })
        print(f"r={r}: Mean Error={mean_err:.4f}, Std={std_err:.4f}")
    
    # Scenario 3: Moneyness changes
    print("\n--- Moneyness Scenarios ---")
    moneyness_results = []
    for K in [80, 90, 100, 110, 120]:
        params = base_params.copy()
        params['K'] = K
        mean_err, std_err, _, _ = discrete_delta_hedging_simulator(**params, num_paths=1000)
        moneyness = "Deep ITM" if K <= 85 else "ITM" if K <= 95 else "ATM" if K == 100 else "OTM" if K <= 115 else "Deep OTM"
        moneyness_results.append({
            'Strike': K,
            'Moneyness': moneyness,
            'Mean_Error': mean_err,
            'Std_Error': std_err,
            'BS_Price': black_scholes_price(params['S0'], K, params['T'], params['r'], params['sigma'])
        })
        print(f"K={K}({moneyness}): Mean Error={mean_err:.4f}, Std={std_err:.4f}")
    
    return {
        'volatility': pd.DataFrame(vol_results),
        'rates': pd.DataFrame(rate_results),
        'moneyness': pd.DataFrame(moneyness_results)
    }

def plot_scenario_results(scenario_results):
    """Plot results from scenario analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Volatility scenario
    axes[0,0].plot(scenario_results['volatility']['Sigma'], 
                  scenario_results['volatility']['Std_Error'], 'bo-', markersize=8)
    axes[0,0].set_xlabel('Volatility (σ)')
    axes[0,0].set_ylabel('Standard Deviation of Error')
    axes[0,0].set_title('Hedging Error vs Volatility')
    axes[0,0].grid(True)
    
    # Interest rate scenario
    axes[0,1].plot(scenario_results['rates']['Rate'], 
                  scenario_results['rates']['Std_Error'], 'ro-', markersize=8)
    axes[0,1].set_xlabel('Risk-free Rate (r)')
    axes[0,1].set_ylabel('Standard Deviation of Error')
    axes[0,1].set_title('Hedging Error vs Interest Rate')
    axes[0,1].grid(True)
    
    # Moneyness scenario
    moneyness_order = ['Deep ITM', 'ITM', 'ATM', 'OTM', 'Deep OTM']
    moneyness_data = scenario_results['moneyness'].set_index('Moneyness').loc[moneyness_order]
    axes[1,0].plot(range(len(moneyness_data)), moneyness_data['Std_Error'], 'go-', markersize=8)
    axes[1,0].set_xticks(range(len(moneyness_data)))
    axes[1,0].set_xticklabels(moneyness_data.index)
    axes[1,0].set_xlabel('Moneyness')
    axes[1,0].set_ylabel('Standard Deviation of Error')
    axes[1,0].set_title('Hedging Error vs Moneyness')
    axes[1,0].grid(True)
    
    # Volatility vs Moneyness interaction
    for moneyness in ['ITM', 'ATM', 'OTM']:
        subset = scenario_results['moneyness'][scenario_results['moneyness']['Moneyness'] == moneyness]
        if len(subset) > 0:
            axes[1,1].scatter([moneyness] * len(scenario_results['volatility']), 
                            scenario_results['volatility']['Std_Error'], 
                            label=f'{moneyness}', s=100, alpha=0.7)
    axes[1,1].set_xlabel('Moneyness')
    axes[1,1].set_ylabel('Std Error across Volatility Scenarios')
    axes[1,1].set_title('Error Distribution by Moneyness and Volatility')
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.show()

def fetch_stock_data(ticker, start_date, end_date=None):
    """
    Fetch stock price data from Yahoo Finance
    """
    if end_date is None:
        end_date = datetime.now()
    
    # Suppress the warning by explicitly setting auto_adjust
    stock = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
    return stock

def calculate_historical_volatility(stock_data):
    """
    Calculate historical volatility from stock data
    """
    returns = np.log(stock_data['Close'] / stock_data['Close'].shift(1))
    historical_vol = returns.std() * np.sqrt(252)  # Annualized
    
    return historical_vol

def get_real_option_data(ticker, expiration_date=None):
    """
    Fetch real option market data from Yahoo Finance
    Returns: DataFrame with real bid/ask prices and implied volatilities
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Get available expiration dates
        expiration_dates = stock.options
        if not expiration_dates:
            print(f"No option data available for {ticker}")
            return None
        
        # Use specified expiration date or the nearest one
        if expiration_date is None:
            expiration_date = expiration_dates[0]
        elif expiration_date not in expiration_dates:
            print(f"Expiration {expiration_date} not available. Using {expiration_dates[0]}")
            expiration_date = expiration_dates[0]
        
        # Fetch option chain
        options_chain = stock.option_chain(expiration_date)
        calls = options_chain.calls
        puts = options_chain.puts
        
        print(f"\n📊 Real Option Data for {ticker} (Expiry: {expiration_date})")
        print(f"Available calls: {len(calls)}, Available puts: {len(puts)}")
        
        # Select relevant columns and add moneyness info
        current_price = stock.info.get('currentPrice', stock.info.get('regularMarketPrice', 0))
        if current_price == 0:
            # Fallback: use recent close price from historical data
            hist = stock.history(period="1d")
            current_price = hist['Close'].iloc[-1] if not hist.empty else 100
        
        calls_data = calls[['strike', 'bid', 'ask', 'impliedVolatility', 'lastPrice']].copy()
        calls_data['moneyness'] = calls_data['strike'].apply(
            lambda k: "ITM" if k < current_price else "OTM" if k > current_price else "ATM"
        )
        calls_data['spread'] = calls_data['ask'] - calls_data['bid']
        calls_data['spread_pct'] = (calls_data['spread'] / ((calls_data['bid'] + calls_data['ask']) / 2)) * 100
        
        # Display sample of the data
        print("\nSample Call Options (sorted by strike):")
        sample_calls = calls_data.sort_values('strike').head(8)
        print(sample_calls.round(3))
        
        return calls_data
        
    except Exception as e:
        print(f"Error fetching option data for {ticker}: {e}")
        return None

def compare_ci_with_real_market(ci_df, real_option_data, tolerance=5):
    """
    Compare theoretical confidence intervals with real market bid/ask spreads
    """
    if real_option_data is None or ci_df.empty:
        print("No data available for comparison")
        return None
    
    comparison_results = []
    
    for _, ci_row in ci_df.iterrows():
        strike = ci_row['Strike']
        maturity = ci_row['Maturity']
        
        # Find matching option in real data (nearest strike)
        matching_options = real_option_data[
            abs(real_option_data['strike'] - strike) <= tolerance
        ]
        
        if not matching_options.empty:
            real_option = matching_options.iloc[0]
            
            comparison_results.append({
                'Strike': strike,
                'Maturity': maturity,
                'Moneyness': ci_row['Moneyness'],
                'BS_Price': ci_row['BS_Price'],
                'CI_Lower': ci_row['CI_Lower'],
                'CI_Upper': ci_row['CI_Upper'],
                'CI_Width': ci_row['CI_Width'],
                'Market_Bid': real_option['bid'],
                'Market_Ask': real_option['ask'],
                'Market_Spread': real_option['spread'],
                'Market_IV': real_option['impliedVolatility'],
                'CI_vs_Market_Ratio': ci_row['CI_Width'] / real_option['spread'] if real_option['spread'] > 0 else float('inf')
            })
    
    if comparison_results:
        comparison_df = pd.DataFrame(comparison_results)
        print("\n" + "="*80)
        print("THEORETICAL vs REAL MARKET COMPARISON")
        print("="*80)
        print(comparison_df[['Strike', 'Maturity', 'Moneyness', 'BS_Price', 'Market_Bid', 
                           'Market_Ask', 'CI_Width', 'Market_Spread', 'CI_vs_Market_Ratio']].round(3))
        
        # Summary statistics
        avg_ratio = comparison_df['CI_vs_Market_Ratio'].mean()
        print(f"Summary Statistics:")
        print(f"Average CI Width / Market Spread Ratio: {avg_ratio:.2f}")
        print(f"Theoretical CI is {avg_ratio:.1f}x wider than market spread on average")
        
        return comparison_df
    else:
        print("No matching options found for comparison")
        return None

def plot_market_comparison(comparison_df):
    """Plot comparison between theoretical CI and market spreads"""
    if comparison_df is None or comparison_df.empty:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: CI Width vs Market Spread
    axes[0].scatter(comparison_df['Market_Spread'], comparison_df['CI_Width'], 
                   c=comparison_df['CI_vs_Market_Ratio'], cmap='viridis', s=100, alpha=0.7)
    axes[0].set_xlabel('Market Spread')
    axes[0].set_ylabel('Theoretical CI Width')
    axes[0].set_title('CI Width vs Market Spread')
    axes[0].grid(True)
    
    # Add identity line
    max_val = max(comparison_df['Market_Spread'].max(), comparison_df['CI_Width'].max())
    axes[0].plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='y=x')
    axes[0].legend()
    
    # Plot 2: Ratio by Moneyness
    moneyness_colors = {'ITM': 'green', 'ATM': 'blue', 'OTM': 'red'}
    for moneyness in comparison_df['Moneyness'].unique():
        subset = comparison_df[comparison_df['Moneyness'] == moneyness]
        axes[1].scatter(subset['Strike'], subset['CI_vs_Market_Ratio'], 
                       label=moneyness, color=moneyness_colors.get(moneyness, 'gray'), 
                       s=100, alpha=0.7)
    
    axes[1].set_xlabel('Strike Price')
    axes[1].set_ylabel('CI Width / Market Spread Ratio')
    axes[1].set_title('Theoretical vs Market Spread Ratio by Strike')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

def get_realistic_parameters(ticker, lookback_years=3):
    """
    Calculate realistic parameters from actual stock data
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_years*365)
    
    # Fetch stock data
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    
    if stock_data.empty:
        raise ValueError(f"Could not fetch data for: {ticker}")
    
    # FIXED: Use .item() or .iloc[0] to extract scalar values properly
    S0 = stock_data['Close'].iloc[-1].item()  # Fixed: use .item() method
    returns = np.log(stock_data['Close'] / stock_data['Close'].shift(1)).dropna()
    
    # FIXED: Extract scalar values properly
    mu = (returns.mean() * 252).item()           # Fixed: use .item()
    sigma = (returns.std() * np.sqrt(252)).item() # Fixed: use .item()
    
    # Risk-free rate (in practice, use treasury yield)
    r = 0.04  # Assumption
    
    print(f"=== {ticker} Parameters ===")
    print(f"Current Price: {S0:.2f}")
    print(f"Expected Return (μ): {mu:.4f}")
    print(f"Volatility (σ): {sigma:.4f}")
    print(f"Risk-free Rate (r): {r:.4f}")
    
    return {
        'S0': S0,
        'mu': mu,
        'sigma': sigma,
        'r': r,
        'stock_data': stock_data
    }

def analyze_specific_stock(ticker, K=None, T=1, N=52, num_paths=1000):
    """
    Perform comprehensive analysis for a specific stock
    """
    print(f"\n{'='*50}")
    print(f"Analysis Target: {ticker}")
    print(f"{'='*50}")
    
    try:
        # Get realistic parameters
        params = get_realistic_parameters(ticker)
        
        # Set strike price (based on current price)
        if K is None:
            K = params['S0']  # ATM
        
        # Delta hedging simulation
        mean_error, std_error, errors, hedge_vals = discrete_delta_hedging_simulator(
            params['S0'], K, T, params['mu'], params['sigma'], params['r'], N, num_paths
        )
        
        bs_price = black_scholes_price(params['S0'], K, T, params['r'], params['sigma'])
        
        print(f"\n=== Simulation Results ===")
        print(f"Black-Scholes Price: {bs_price:.4f}")
        print(f"Mean Replication Error: {mean_error:.4f}")
        print(f"Error Standard Deviation: {std_error:.4f}")
        print(f"95% Confidence Interval: [{bs_price-1.96*std_error:.4f}, {bs_price+1.96*std_error:.4f}]")
        
        return {
            'parameters': params,
            'results': (mean_error, std_error, errors, hedge_vals),
            'bs_price': bs_price
        }
    
    except Exception as e:
        print(f"Error in analyze_specific_stock for {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main_with_real_stocks():
    """Analysis using real stocks"""
    
    # List of stocks to analyze
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY']
    
    all_results = {}
    successful_tickers = []
    
    for ticker in tickers:
        try:
            print(f"\n{'='*50}")
            print(f"Starting analysis for: {ticker}")
            print(f"{'='*50}")
            
            results = analyze_specific_stock(
                ticker=ticker,
                K=None,  # ATM
                T=0.5,   # 6 months
                N=26,    # Bi-weekly
                num_paths=1000
            )
            
            if results is not None:
                all_results[ticker] = results
                successful_tickers.append(ticker)
                print(f"✓ Successfully analyzed {ticker}")
            else:
                print(f"✗ Failed to analyze {ticker}")
            
        except Exception as e:
            print(f"Error in main_with_real_stocks for {ticker}: {e}")
            import traceback
            traceback.print_exc()
    
    # Cross-stock comparison - only for successful analyses
    if successful_tickers:
        print(f"\n=== Cross-Stock Comparison ({len(successful_tickers)} stocks) ===")
        comparison_data = []
        for ticker in successful_tickers:
            result = all_results[ticker]
            params = result['parameters']
            mean_err, std_err, _, _ = result['results']
            
            comparison_data.append({
                'Ticker': ticker,
                'Price': params['S0'],
                'Volatility': params['sigma'],
                'BS_Price': result['bs_price'],
                'Mean_Error': mean_err,
                'Std_Error': std_err
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.round(4))
        
        return all_results, comparison_df
    else:
        print("\n=== No stocks were successfully analyzed ===")
        return {}, pd.DataFrame()

def plot_stock_comparison(comparison_df):
    """Plot stock comparison results"""
    if comparison_df.empty:
        print("No data to plot - comparison dataframe is empty")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Volatility vs Hedging Error
    axes[0,0].scatter(comparison_df['Volatility'], comparison_df['Std_Error'], 
                     s=100, alpha=0.7)
    for i, row in comparison_df.iterrows():
        axes[0,0].annotate(row['Ticker'], (row['Volatility'], row['Std_Error']),
                          xytext=(5, 5), textcoords='offset points')
    axes[0,0].set_xlabel('Volatility')
    axes[0,0].set_ylabel('Standard Deviation of Error')
    axes[0,0].set_title('Volatility vs Hedging Error')
    axes[0,0].grid(True)
    
    # Stock Price vs Option Price
    axes[0,1].scatter(comparison_df['Price'], comparison_df['BS_Price'], 
                     s=100, alpha=0.7)
    for i, row in comparison_df.iterrows():
        axes[0,1].annotate(row['Ticker'], (row['Price'], row['BS_Price']),
                          xytext=(5, 5), textcoords='offset points')
    axes[0,1].set_xlabel('Stock Price')
    axes[0,1].set_ylabel('Option Price')
    axes[0,1].set_title('Stock Price vs Option Price')
    axes[0,1].grid(True)
    
    # Hedging Error Comparison
    axes[1,0].bar(comparison_df['Ticker'], comparison_df['Std_Error'])
    axes[1,0].set_ylabel('Standard Deviation of Error')
    axes[1,0].set_title('Hedging Error by Stock')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Volatility Comparison
    axes[1,1].bar(comparison_df['Ticker'], comparison_df['Volatility'])
    axes[1,1].set_ylabel('Volatility')
    axes[1,1].set_title('Volatility by Stock')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Base parameters for theoretical analysis
    S0 = 100
    K_delta = 100  # ATM for delta hedging comparison
    K_zero_one = 110  # OTM for zero/one strategy (K > S0 as in text description)
    T = 1
    mu = 0.1
    sigma = 0.3
    r = 0.04
    N = 4
    
    print("=== DISCRETE DELTA HEDGING SIMULATION ===")
    print(f"Base Parameters: S0={S0}, K_delta={K_delta}, K_zero_one={K_zero_one}, T={T}, μ={mu}, σ={sigma}, r={r}")
    print(f"Delta Hedging: ATM (K=S0), Zero/One Strategy: OTM (K>S0) as described in text")
    
    # ============================================================================
    # 1. THEORETICAL ANALYSIS WITH BASE PARAMETERS
    # ============================================================================
    
    print("\n--- 1. Basic Delta Hedging Simulation ---")
    mean_error, std_error, errors, hedge_vals = discrete_delta_hedging_simulator(
        S0, K_delta, T, mu, sigma, r, N, num_paths=2000
    )
    
    bs_price = black_scholes_price(S0, K_delta, T, r, sigma, 'call')
    print(f"Black-Scholes Price: {bs_price:.6f}")
    print(f"Mean Replication Error: {mean_error:.6f}")
    print(f"Standard Deviation: {std_error:.6f}")
    
    # ============================================================================
    # 2. HEDGING FREQUENCY ANALYSIS
    # ============================================================================
    
    print("\n--- 2. Hedging Frequency Analysis ---")
    N_list = [4, 8, 12, 26, 52, 104, 252]
    means_freq, stds_freq = analyze_hedging_frequency(S0, K_delta, T, mu, sigma, r, N_list, num_paths=1000)
    variances = [std**2 for std in stds_freq]
    alpha, r_squared = estimate_convergence_rate(N_list, variances)
    
    # ============================================================================
    # 3. DRIFT RATE SENSITIVITY ANALYSIS
    # ============================================================================
    
    print("\n--- 3. Drift Rate Sensitivity Analysis ---")
    mu_list = [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
    means_drift, stds_drift = analyze_drift_dependence(S0, K_delta, T, sigma, r, N, mu_list, num_paths=1000)
    
    # ============================================================================
    # 4. ALTERNATIVE STRATEGY COMPARISON - USING OTM FOR ZERO/ONE
    # ============================================================================
    
    print("\n--- 4. Alternative Strategy Analysis ---")
    
    print("\n4.1 Zero/One Strategy - Frequency Analysis (OTM)")
    zero_one_means_freq, zero_one_stds_freq = analyze_zero_one_frequency(
        S0, K_zero_one, T, mu, sigma, r, N_list, num_paths=500
    )
    
    print("\n4.2 Zero/One Strategy - Drift Analysis (OTM)")
    zero_one_means_drift, zero_one_stds_drift = analyze_zero_one_drift(
        S0, K_zero_one, T, sigma, r, N, mu_list, num_paths=500
    )
    
    # ============================================================================
    # 5. COMPREHENSIVE VISUALIZATION
    # ============================================================================
    
    print("\n--- 5. Generating Comprehensive Plots ---")
    plot_comprehensive_results(
        N_list, 
        means_freq,
        stds_freq, 
        zero_one_means_freq, 
        zero_one_stds_freq,
        mu_list, 
        means_drift, 
        stds_drift, 
        zero_one_means_drift, 
        zero_one_stds_drift
    )
    
    # ============================================================================
    # 6. STRIKE AND MATURITY GRID ANALYSIS
    # ============================================================================
    
    print("\n--- 6. Strike and Maturity Grid Analysis ---")
    Ks = [250, 260, 270, 280, 290]  # OTM strikes for S0=100
    Ts = [0.25, 0.5, 1.0, 2.0]
    grid_results = ci_grid_over_KT(S0, mu, sigma, r, N, Ks, Ts, num_paths=500)
    
    print("\nConfidence Interval Summary:")
    print(grid_results[['Strike', 'Maturity', 'Moneyness', 'BS_Price', 'CI_Width_Pct']].round(3))
    
    # ============================================================================
    # 7. SCENARIO ANALYSIS
    # ============================================================================
    
    print("\n" + "="*60)
    print("7. SCENARIO ANALYSIS - Parameter Sensitivity")
    print("="*60)
    
    scenario_results = run_scenario_analysis()
    
    print("\nScenario Analysis Summary:")
    print("Volatility Impact:")
    print(scenario_results['volatility'].round(4))
    print("\nInterest Rate Impact:")
    print(scenario_results['rates'].round(4))
    print("\nMoneyness Impact:")
    print(scenario_results['moneyness'].round(4))
    
    plot_scenario_results(scenario_results)
    
    # ============================================================================
    # 8. REAL STOCK ANALYSIS
    # ============================================================================

    print("\n" + "="*60)
    print("8. REAL STOCK ANALYSIS - Market Data Application")
    print("="*60)

    # Analyze multiple real stocks
    print("\n--- 8.1 Multi-Stock Comparative Analysis ---")
    all_results, comparison_df = main_with_real_stocks()

    # Get real option market data for comparison
    print("\n--- 8.2 Real Option Market Data Analysis ---")
    real_option_data = get_real_option_data('AAPL')

    # Compare theoretical CI with real market spreads
    if real_option_data is not None:
        market_comparison = compare_ci_with_real_market(grid_results, real_option_data)
    if market_comparison is not None:
        plot_market_comparison(market_comparison)

    if not comparison_df.empty:
        plot_stock_comparison(comparison_df)

    # Detailed analysis of specific stock
    print("\n--- 8.3 Apple Inc. (AAPL) Detailed Analysis ---")
    aapl_results = analyze_specific_stock('AAPL', T=1, N=52, num_paths=5000)

    # Strike price sensitivity for AAPL - using OTM strikes
    print("\n--- 8.4 AAPL Strike Price Sensitivity (OTM) ---")
    # Get current AAPL price for OTM strike calculation
    aapl_params = get_realistic_parameters('AAPL')
    aapl_price = aapl_params['S0']
    otm_strikes = [int(aapl_price * 1.1), int(aapl_price * 1.15), int(aapl_price * 1.2), 
                   int(aapl_price * 1.25), int(aapl_price * 1.3)]
    
    for strike in otm_strikes:
        analyze_specific_stock('AAPL', K=strike, T=0.5, N=26, num_paths=1000)

    # ============================================================================
    # 9. FINAL SUMMARY AND KEY FINDINGS
    # ============================================================================

    print("\n" + "="*60)
    print("FINAL SUMMARY AND KEY FINDINGS")
    print("="*60)

    # Add market comparison result to summary
    market_ratio = market_comparison['CI_vs_Market_Ratio'].mean() if market_comparison is not None else 0

    print(f"1. Convergence rate: α = {alpha:.3f} (theoretical expectation: 1.0)")
    print(f"2. Delta hedging error Std: {stds_freq[-1]:.3f} (daily) vs {stds_freq[0]:.3f} (quarterly)")
    print(f"3. Zero/One strategy (OTM) performs poorly (Std ≈ {zero_one_stds_freq[0]:.1f})")
    print(f"4. Volatility impact: σ=0.15→Std={scenario_results['volatility']['Std_Error'].iloc[0]:.3f}, σ=0.8→Std={scenario_results['volatility']['Std_Error'].iloc[-1]:.3f}")
    print(f"5. Moneyness risk: ITM Std={scenario_results['moneyness']['Std_Error'].iloc[0]:.3f}, OTM Std={scenario_results['moneyness']['Std_Error'].iloc[-1]:.3f}")
    print(f"6. Real stocks: TSLA (σ={comparison_df['Volatility'].iloc[3]:.1%}) has highest hedging error (Std={comparison_df['Std_Error'].iloc[3]:.2f})")
    print(f"7. AAPL OTM strike sensitivity analyzed for strikes: {otm_strikes}")
    print(f"8. Interest rate impact limited: r=0.01→Std={scenario_results['rates']['Std_Error'].iloc[0]:.3f}, r=0.15→Std={scenario_results['rates']['Std_Error'].iloc[-1]:.3f}")
    print(f"9. Market comparison: Theoretical CI is {market_ratio:.1f}x wider than actual market spreads")
    print(f"10. Hedging frequency critical: daily vs quarterly reduces error by {(1-stds_freq[-1]/stds_freq[0])*100:.1f}%")