import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pandas.tseries.holiday import USFederalHolidayCalendar
import uuid

# --- 1. CẤU HÌNH PORTFOLIO ---
ASSETS = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'ADA-USD', 'LTC-USD']
TOTAL_CAPITAL = 60000   # Tổng vốn
ALLOCATION_PER_ASSET = TOTAL_CAPITAL / len(ASSETS) # Chia đều vốn ($10k/coin)

# --- THÔNG SỐ CHIẾN LƯỢC ---
TRADE_SIZE_PCT = 1.0    # Dùng 100% vốn được cấp cho coin đó (All-in trên sub-account)
FEE = 0.001             # Phí 0.1%
N_DAYS_HIGH = 30        
SMA_PERIOD = 50         
HOLIDAY_WINDOW_START = -5
HOLIDAY_WINDOW_END = 1

# --- 2. DATA LOADER ---
def fetch_data(symbol, start_date, end_date):
    try:
        # Load dư để tính SMA
        start_dt = pd.to_datetime(start_date) - timedelta(days=200)
        df = yf.download(symbol, start=start_dt, end=end_date, progress=False, auto_adjust=False)
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = df.columns.str.lower()
        df.index = df.index.tz_localize(None)
        
        df = df.reset_index()
        df.rename(columns={'Date': 'timestamp', 'open': 'o', 'high': 'h', 'low': 'l', 'close': 'c', 'volume': 'v'}, inplace=True)
        df['date'] = df['timestamp'].dt.date
        return df
    except:
        return pd.DataFrame()

# --- 3. INDICATORS ---
def add_indicators(df):
    df['sma'] = df['c'].rolling(window=SMA_PERIOD).mean()
    df['n_high'] = df['h'].shift(1).rolling(window=N_DAYS_HIGH).max()
    
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=df['timestamp'].min(), end=df['timestamp'].max())
    
    df['is_holiday_window'] = False
    for h in holidays:
        mask = (df['timestamp'] >= h + timedelta(days=HOLIDAY_WINDOW_START)) & \
               (df['timestamp'] <= h + timedelta(days=HOLIDAY_WINDOW_END))
        df.loc[mask, 'is_holiday_window'] = True
        
        # Đánh dấu ngày exit
        end_window_day = (h + timedelta(days=HOLIDAY_WINDOW_END)).date()
        df.loc[df['date'] == end_window_day, 'is_exit_day'] = True

    df['is_exit_day'] = df['is_exit_day'].fillna(False)
    return df

# --- 4. ENGINE: SINGLE ASSET RUNNER ---
def run_single_asset(symbol, start_date, end_date, initial_capital):
    """
    Chạy backtest cho 1 coin, trả về đường cong vốn (Equity Curve) của coin đó.
    """
    print(f"🔄 Running: {symbol} (Capital: ${initial_capital:,.0f})...")
    
    df = fetch_data(symbol, start_date, end_date)
    if df.empty: return None
    df = add_indicators(df)
    df = df[df['timestamp'] >= pd.to_datetime(start_date)].reset_index(drop=True)
    
    capital = initial_capital
    open_trades = []
    equity_curve = [] # List of {date, equity}
    
    for i in range(len(df)):
        row = df.iloc[i]
        current_price = row['c']
        
        # A. QUẢN LÝ LỆNH (EXIT)
        active_trades = []
        for trade in open_trades:
            if row['is_exit_day']:
                # Bán
                exit_price = current_price
                revenue = trade['amount'] * (exit_price / trade['entry_price'])
                revenue_after_fee = revenue * (1 - FEE)
                
                capital += revenue_after_fee
                # (Đã đóng lệnh, không add vào active_trades)
            else:
                active_trades.append(trade)
        open_trades = active_trades

        # B. TÌM LỆNH MỚI (ENTRY)
        if len(open_trades) == 0: # Chỉ vào lệnh nếu đang cầm tiền
            if row['is_holiday_window'] and row['c'] > row['sma'] and row['c'] > row['n_high']:
                # Mua
                pos_size_usd = capital * TRADE_SIZE_PCT # Mua hết tiền cash đang có
                cost_after_fee = pos_size_usd * (1 - FEE)
                
                capital -= pos_size_usd # Trừ tiền mặt
                
                new_trade = {
                    "entry_price": row['c'],
                    "amount": cost_after_fee # Giá trị nắm giữ (USD)
                }
                open_trades.append(new_trade)
        
        # C. TÍNH EQUITY
        # Equity = Tiền mặt + Giá trị Coin đang giữ
        holding_value = 0
        for t in open_trades:
            # Giá trị hiện tại = Giá trị lúc mua * (Giá hiện tại / Giá mua)
            current_val = t['amount'] * (current_price / t['entry_price'])
            holding_value += current_val
            
        total_equity = capital + holding_value
        equity_curve.append({"timestamp": row['timestamp'], "equity": total_equity})
        
    return pd.DataFrame(equity_curve).set_index("timestamp")

# --- 5. ENGINE: PORTFOLIO AGGREGATOR ---
def run_portfolio_backtest(assets, start_date="2022-01-01", end_date="2025-01-01"):
    print(f"🚀 BẮT ĐẦU BACKTEST PORTFOLIO (Total Capital: ${TOTAL_CAPITAL:,.0f})")
    
    all_equities = pd.DataFrame()
    
    # 1. Chạy từng coin
    for symbol in assets:
        df_equity = run_single_asset(symbol, start_date, end_date, ALLOCATION_PER_ASSET)
        if df_equity is not None:
            # Đổi tên cột equity thành tên coin để gộp
            df_equity.rename(columns={'equity': symbol}, inplace=True)
            # Merge vào bảng tổng
            if all_equities.empty:
                all_equities = df_equity
            else:
                all_equities = all_equities.join(df_equity, how='outer')
    
    # 2. Xử lý dữ liệu gộp
    # Forward fill: Nếu coin nào không có data ngày hôm đó (do lệch giờ), lấy giá trị ngày trước
    all_equities = all_equities.fillna(method='ffill')
    # Fill ban đầu: Những ngày chưa có dữ liệu thì coi như vẫn giữ nguyên vốn gốc
    all_equities = all_equities.fillna(ALLOCATION_PER_ASSET)
    
    # 3. Tính Tổng Portfolio
    all_equities['Portfolio'] = all_equities.sum(axis=1)
    
    # 4. Báo cáo
    final_equity = all_equities['Portfolio'].iloc[-1]
    total_ret = (final_equity - TOTAL_CAPITAL) / TOTAL_CAPITAL
    
    # Tính Max Drawdown
    rolling_max = all_equities['Portfolio'].cummax()
    drawdown = (all_equities['Portfolio'] / rolling_max) - 1
    max_dd = drawdown.min()
    
    # Tính CAGR
    days = (all_equities.index[-1] - all_equities.index[0]).days
    cagr = (final_equity / TOTAL_CAPITAL) ** (365/days) - 1
    
    print("\n" + "="*40)
    print(f"📊 KẾT QUẢ PORTFOLIO ({len(assets)} ASSETS)")
    print(f"► Vốn đầu: ${TOTAL_CAPITAL:,.0f}")
    print(f"► Vốn cuối: ${final_equity:,.0f}")
    print(f"► Lợi nhuận (Total Return): {total_ret:.2%}")
    print(f"► CAGR: {cagr:.2%}")
    print(f"► Max Drawdown: {max_dd:.2%}")
    print("="*40)
    
    # 5. Vẽ biểu đồ
    plt.figure(figsize=(12, 8))
    
    # Vẽ các coin thành phần (mờ)
    for col in assets:
        if col in all_equities.columns:
            plt.plot(all_equities.index, all_equities[col], alpha=0.3, label=f"{col} (Component)")
            
    # Vẽ Portfolio (Đậm)
    plt.plot(all_equities.index, all_equities['Portfolio'], color='green', linewidth=3, label='TOTAL PORTFOLIO')
    
    plt.title('Holiday Strategy: Portfolio Equity Curve')
    plt.ylabel('Total Equity ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

# --- MAIN ---
if __name__ == "__main__":
    run_portfolio_backtest(ASSETS, start_date="2022-01-01", end_date="2025-01-01")