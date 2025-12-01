#!/usr/bin/env python3

import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.tseries.holiday import USFederalHolidayCalendar
from typing import Dict, List, Tuple

# ==========================================
# 1. CONFIGURATIONS (Cấu hình chiến lược)
# ==========================================
STRATEGY_CONFIGS = {
    # Danh sách các tài sản cần test
    "ASSETS": ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD'],
    
    # Khoảng thời gian backtest
    "START_DATE": '2018-01-01',
    "END_DATE": '2023-12-31',
    
    # GRID SEARCH PARAMS (Các bộ thông số cần test)
    # Ví dụ: Test SMA 50, 100, 200 kết hợp với Lookback 10, 20, 30 ngày
    "GRID_SEARCH": {
        "N_DAYS": [10, 20, 30],       # Số ngày nhìn lại (High cũ)
        "SMA_PERIODS": [50, 100, 200] # Xu hướng dài hạn
    },
    
    "FEE": 0.001 # Phí giao dịch 0.1%
}

# ==========================================
# 2. DATA ENGINE
# ==========================================
def load_data_store(assets: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """Tải dữ liệu từ Yahoo Finance."""
    data_store = {}
    print(f"📥 Loading data for {len(assets)} assets...")
    
    for symbol in assets:
        try:
            # Tải dữ liệu
            df = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=False)
            
            # Xử lý format của yfinance mới (MultiIndex)
            if isinstance(df.columns, pd.MultiIndex):
                try:
                    df.columns = df.columns.get_level_values(0)
                except Exception as e:
                    print(f"⚠️ Warning format {symbol}: {e}")
            
            # Chuẩn hóa tên cột và index
            df.columns = df.columns.str.lower()
            df.index = df.index.tz_localize(None)
            df = df.resample('1D').last().ffill()
            
            # Validate dữ liệu
            if len(df) == 0:
                raise ValueError(f"Dữ liệu {symbol} trống!")
            
            if 'close' not in df.columns:
                raise ValueError(f"Dữ liệu {symbol} thiếu cột 'close'!")

            data_store[symbol] = df
            print(f"✅ Loaded {symbol}: {len(df)} rows")
            
        except Exception as e:
            # Đây là chỗ chụp lỗi nếu load data fail
            print(f"❌ [DATA ERROR] Could not load {symbol}. Reason: {str(e)}")
            
    return data_store

# ==========================================
# 3. STRATEGY LOGIC (Core)
# ==========================================
def execute_strategy_logic(df: pd.DataFrame, n_days: int, sma_period: int, fee: float) -> pd.Series:
    """
    Logic: Mua khi giá > Đỉnh N ngày cũ VÀ giá > SMA (Trong kỳ nghỉ lễ)
    """
    try:
        data = df.copy()
        
        # 1. Tính toán Indicators
        # Tìm giá cao nhất trong N ngày trước
        data['n_high'] = data['close'].shift(1).rolling(window=n_days).max()
        # Tính đường trung bình SMA
        data['sma'] = data['close'].rolling(window=sma_period).mean()
        
        # 2. Logic Holiday (Vùng lịch nghỉ lễ Mỹ)
        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(start=data.index.min(), end=data.index.max())
        
        data['in_window'] = False
        idx_dates = data.index
        # Tạo window xung quanh ngày lễ (-5 đến +1 ngày)
        for h in holidays:
            mask = (idx_dates >= h + pd.Timedelta(days=-5)) & \
                   (idx_dates <= h + pd.Timedelta(days=1))
            data.loc[mask, 'in_window'] = True

        # 3. Entry Condition (Điều kiện vào lệnh)
        # Strat: Có Window Lễ + Giá Breakout đỉnh cũ + Giá trên SMA
        entry_condition = (
            (data['in_window']) & 
            (data['close'] > data['n_high']) & 
            (data['close'] > data['sma'])
        )
        
        
        # Chuyển đổi sang Signal (1: Giữ hàng, 0: Cash)
        data['signal'] = np.where(entry_condition, 1, 0)
        
        # 4. Tính PnL (Lợi nhuận)
        market_ret = data['close'].pct_change()
        # Lợi nhuận chiến lược = Signal hôm qua * Biến động giá hôm nay
        strat_ret = data['signal'].shift(1) * market_ret
        
        # Trừ phí giao dịch (Mỗi lần signal thay đổi là 1 lần trade)
        trades = data['signal'].diff().abs()
        costs = trades * fee
        
        net_pnl = strat_ret - costs
        return net_pnl.fillna(0)

    except Exception as e:
        # Chụp lỗi logic nếu tính toán fail
        print(f"❌ [STRAT ERROR] Logic failed at N={n_days}, SMA={sma_period}. Reason: {e}")
        return pd.Series(0, index=df.index)

# ==========================================
# 4. BACKTEST ENGINE (Grid Search)
# ==========================================
def run_backtest_grid(data_store: Dict[str, pd.DataFrame], configs: dict) -> pd.DataFrame:
    results = []
    
    n_options = configs['GRID_SEARCH']['N_DAYS']
    sma_options = configs['GRID_SEARCH']['SMA_PERIODS']
    fee = configs['FEE']
    
    total_runs = len(n_options) * len(sma_options)
    print(f"\n🚀 Starting Grid Search: {total_runs} combinations...")
    
    count = 0
    for n in n_options:
        for sma in sma_options:
            count += 1
            # print(f"   Running combo {count}/{total_runs}: N={n}, SMA={sma}...", end='\r')
            
            portfolio_returns = pd.DataFrame()
            
            # Chạy loop qua từng tài sản
            for symbol, df in data_store.items():
                pnl = execute_strategy_logic(df, n, sma, fee)
                portfolio_returns[symbol] = pnl
            
            # Tổng hợp Portfolio (Equal weight)
            avg_pnl = portfolio_returns.mean(axis=1).fillna(0)
            
            # Tính chỉ số hiệu quả (Sharpe, Drawdown)
            if avg_pnl.std() != 0:
                sharpe = (avg_pnl.mean() / avg_pnl.std()) * np.sqrt(365)
            else:
                sharpe = 0
            
            cum_ret = (1 + avg_pnl).cumprod()
            if not cum_ret.empty:
                total_return = cum_ret.iloc[-1] - 1
                mdd = (cum_ret / cum_ret.cummax() - 1).min()
            else:
                total_return = 0
                mdd = 0
                
            results.append({
                'N_Days': n,
                'SMA_Period': sma,
                'Sharpe': sharpe,
                'Return': total_return,
                'Max_Drawdown': mdd
            })
            
    print(f"\n✅ Grid Search Completed.")
    return pd.DataFrame(results)

# ==========================================
# 5. MAIN (Chạy chương trình)
# ==========================================
if __name__ == "__main__":
    print("--- STARTING BACKTEST SIMULATION ---\n")
    
    try:
        # Bước 1: Load Data
        data = load_data_store(
            STRATEGY_CONFIGS['ASSETS'], 
            STRATEGY_CONFIGS['START_DATE'], 
            STRATEGY_CONFIGS['END_DATE']
        )
        
        if not data:
            print("❌ [CRITICAL] No data loaded. Stopping backtest.")
            exit()

        # Bước 2: Chạy Backtest theo Configs
        df_results = run_backtest_grid(data, STRATEGY_CONFIGS)
        
        # Bước 3: Show Kết quả
        if not df_results.empty:
            df_sorted = df_results.sort_values(by='Sharpe', ascending=False)
            
            print("\n🏆 TOP 5 CONFIGURATIONS:")
            print(df_sorted.head(5).to_string(index=False))
            
            # Vẽ biểu đồ Heatmap (nếu chạy local)
            try:
                best_config = df_sorted.iloc[0]
                print(f"\n💡 Best Config: N={best_config['N_Days']}, SMA={best_config['SMA_Period']}")
                
                pivot = df_results.pivot(index='N_Days', columns='SMA_Period', values='Sharpe')
                plt.figure(figsize=(10, 6))
                sns.heatmap(pivot, annot=True, cmap='RdYlGn', fmt='.2f')
                plt.title('Strategy Performance Heatmap (Sharpe Ratio)')
                plt.show()
            except Exception as e:
                print(f"\n⚠️ Could not plot heatmap: {e}")
        else:
            print("❌ No results generated.")

    except Exception as fatal_e:
        print("\n" + "="*40)
        print(f"❌ FATAL ERROR: {fatal_e}")
        print("Please screenshot this error and send to dev.")
        print("="*40)
