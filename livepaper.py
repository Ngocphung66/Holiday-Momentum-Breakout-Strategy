#!/usr/bin/env python3
"""
Live Trading Bot: Holiday Momentum Strategy
Structure: Class-based (Version 1 Logic) + Config-driven (Version 2 Setup)
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import yfinance as yf # Trong thực tế có thể thay bằng ccxt hoặc binance api
from datetime import datetime, timedelta
from pandas.tseries.holiday import USFederalHolidayCalendar

# --- 1. SETUP ĐƯỜNG DẪN & IMPORT HỆ THỐNG ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
sys.path.insert(0, PROJECT_ROOT)

# Giả lập import các module hệ thống (nếu bạn chạy local thì bỏ comment các dòng thật)
try:
    from logger import logger_database, logger_error, logger_access
    from exchange_api_spot.user import get_client_exchange
    from utils import get_line_number, update_key_and_insert_error_log, generate_random_string
    from constants import get_constants
except ImportError:
    # Fallback cho logging nếu không có file hệ thống
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger_access = logging.getLogger('access')
    logger_error = logging.getLogger('error')
    logger_database = logging.getLogger('database')
    
    # Mock function
    def get_constants(): return {}
    def generate_random_string(): return "test_run"
    def update_key_and_insert_error_log(*args): pass
    def get_line_number(): return 0
    def get_client_exchange(*args, **kwargs): 
        class MockClient:
            def place_order(self, **kwargs): return {'code': 0, 'msg': 'Mock Order Placed'}
            def get_account_balance(self): return {'USDT': {'available': 50000}}
        return MockClient()

# --- 2. CLASS CHIẾN LƯỢC (HOLIDAY MOMENTUM) ---
class HolidayMomentumLiveStrategy:
    def __init__(self, api_key, secret_key, passphrase, session_key, strategy_config):
        """
        Khởi tạo Bot với Config động.
        """
        self.run_key = generate_random_string()
        self.session_key = session_key
        
        # A. LOAD CONFIG (Từ Dictionary truyền vào)
        self.assets = strategy_config.get("ASSETS", ['BTC-USD']) # List các coin
        self.quote = strategy_config.get("QUOTE", "USDT")
        self.timeframe = strategy_config.get("TIMEFRAME", "1d") # Chiến lược này chạy khung D1
        self.trade_amount = float(strategy_config.get("TRADE_AMOUNT", 100)) # $100 mỗi lệnh
        
        # Tham số Chiến lược (Tối ưu từ Grid Search)
        self.n_days = int(strategy_config.get("N_DAYS", 30))      # Breakout 30 ngày
        self.sma_period = int(strategy_config.get("SMA_PERIOD", 50)) # Trend Filter 50 ngày
        
        # Quản lý trạng thái (Giả sử đang giữ tiền mặt)
        # Trong thực tế cần check balance hoặc file state để biết đang giữ coin nào
        self.positions = {symbol: False for symbol in self.assets} 
        
        # B. INIT CLIENT
        try:
            account_info = {"api_key": api_key, "secret_key": secret_key, "passphrase": passphrase}
            self.client = get_client_exchange(
                exchange_name="binance",
                acc_info=account_info,
                symbol=self.assets[0], # Init đại diện
                quote=self.quote,
                session_key=session_key,
            )
            logger_access.info(f"✅ Client initialized. Assets: {self.assets}")
            logger_database.info(f"Strategy Config: N={self.n_days}, SMA={self.sma_period}")
        except Exception as e:
            logger_error.error(f"❌ Init Client Failed: {e}")
            raise

    def fetch_data(self, symbol):
        """
        Lấy dữ liệu nến Nhật. 
        Cần ít nhất 200 nến để tính SMA và N-High.
        """
        try:
            # Lưu ý: Live trade dùng yfinance hơi chậm, nên dùng CCXT hoặc API sàn trực tiếp
            # Ở đây giữ yfinance theo code cũ của bạn để đồng bộ logic
            df = yf.download(symbol, period='1y', interval='1d', progress=False, auto_adjust=False)
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = df.columns.str.lower()
            df.index = df.index.tz_localize(None)
            
            return df
        except Exception as e:
            logger_error.error(f"Error fetching data for {symbol}: {e}")
            return None

    def check_signal(self, symbol):
        """
        Logic cốt lõi: Holiday + SMA + Breakout
        """
        df = self.fetch_data(symbol)
        if df is None or len(df) < self.sma_period:
            return "SKIP"

        # 1. Tính toán Chỉ báo (Indicators)
        # SMA Trend
        sma_val = df['close'].rolling(window=self.sma_period).mean().iloc[-1]
        
        # N-Day High (Breakout)
        # Lưu ý: So sánh giá hôm nay với High của 30 ngày TRƯỚC ĐÓ (không tính hôm nay)
        n_high_val = df['high'].shift(1).rolling(window=self.n_days).max().iloc[-1]
        
        current_price = df['close'].iloc[-1]
        current_date = df.index[-1]

        # 2. Kiểm tra Holiday Window
        cal = USFederalHolidayCalendar()
        # Lấy lịch lễ trong khoảng thời gian gần đây
        holidays = cal.holidays(start=current_date - timedelta(days=10), end=current_date + timedelta(days=10))
        
        in_holiday_window = False
        for h in holidays:
            # Cửa sổ: [T-5 đến T+1]
            start_window = h - timedelta(days=5)
            end_window = h + timedelta(days=1)
            
            if start_window <= current_date <= end_window:
                in_holiday_window = True
                break # Đã nằm trong vùng lễ

        # 3. Logic Quyết định (Decision Making)
        is_uptrend = current_price > sma_val
        is_breakout = current_price > n_high_val
        
        logger_access.info(f"🔍 {symbol}: Price={current_price:.2f} | SMA({self.sma_period})={sma_val:.2f} | High({self.n_days})={n_high_val:.2f} | Holiday={in_holiday_window}")

        # LOGIC MUA
        if in_holiday_window and is_uptrend and is_breakout:
            return "BUY"
        
        # LOGIC BÁN (Thoát khi hết kỳ nghỉ lễ)
        elif not in_holiday_window:
            return "SELL"
            
        return "HOLD"

    def execute_strategy(self):
        """Hàm chạy chính, duyệt qua portfolio."""
        logger_access.info("--- Scanning Portfolio ---")
        
        for symbol in self.assets:
            signal = self.check_signal(symbol)
            
            if signal == "BUY" and not self.positions[symbol]:
                logger_access.info(f"🚀 BUY SIGNAL for {symbol}")
                # Đặt lệnh mua
                res = self.client.place_order(
                    side_order='BUY', 
                    quantity=self.trade_amount, # Cần tính ra số lượng coin dựa trên giá
                    order_type='MARKET'
                )
                if res and res.get('code') == 0:
                    self.positions[symbol] = True
                    logger_database.info(f"Opened Long {symbol}")

            elif signal == "SELL" and self.positions[symbol]:
                logger_access.info(f"📉 SELL SIGNAL for {symbol} (Holiday Ended)")
                # Đặt lệnh bán
                res = self.client.place_order(
                    side_order='SELL', 
                    quantity=self.trade_amount, 
                    order_type='MARKET'
                )
                if res and res.get('code') == 0:
                    self.positions[symbol] = False
                    logger_database.info(f"Closed Long {symbol}")
            
            else:
                logger_access.info(f"Basic check {symbol}: No Action ({signal})")

# --- 3. MAIN LOOP (CẤU TRÚC BẢN 2) ---
def main():
    logger_access.info("🚀 STARTING HOLIDAY MOMENTUM BOT...")
    
    # A. LOAD ENV (Credentials)
    # Giả lập lấy từ biến môi trường
    env_vars = get_constants()
    API_KEY = env_vars.get("API_KEY", "dummy_key")
    SECRET_KEY = env_vars.get("SECRET_KEY", "dummy_secret")
    PASSPHRASE = env_vars.get("PASSPHRASE", "dummy_pass")
    SESSION_ID = env_vars.get("SESSION_ID", "holiday_bot_v1")

    # B. CONFIGS (Tham số Chiến lược & Portfolio)
    # Đây là chỗ bạn điều chỉnh danh mục và tham số
    STRATEGY_CONFIG = {
        # Portfolio: Đa dạng hóa 6 coin như bài báo cáo
        "ASSETS": ['BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'ADA-USD', 'LTC-USD'],
        "QUOTE": "USDT",
        "TIMEFRAME": "1d",     # Khung ngày
        "TRADE_AMOUNT": 500,   # Giá trị vào lệnh ($500)
        
        # Tham số Tối ưu (Optimal Parameters)
        "N_DAYS": 30,          # Breakout
        "SMA_PERIOD": 50       # Trend Filter (Dùng 50 như bài báo cáo mới nhất)
    }
    
    # C. INIT & INFINITE LOOP
    try:
        # Khởi tạo Bot với Configs
        bot = HolidayMomentumLiveStrategy(API_KEY, SECRET_KEY, PASSPHRASE, SESSION_ID, STRATEGY_CONFIG)
        
        iteration = 0
        while True:
            iteration += 1
            logger_access.info(f"\n🔄 Iteration #{iteration} (Daily Check)")
            
            # Chạy logic
            bot.execute_strategy()
            
            # Sleep: Vì đánh khung D1, ta chỉ cần check vài tiếng một lần hoặc 1 ngày 1 lần.
            # Để demo, sleep 60s. Thực tế nên sleep 1 giờ (3600s).
            logger_access.info("💤 Sleeping 60s...")
            time.sleep(60)

    except KeyboardInterrupt:
        logger_access.info("🛑 Bot stopped by user.")
    except Exception as e:
        logger_error.error(f"🔥 Fatal Error in Main: {e}")
        # Ghi log lỗi vào DB
        update_key_and_insert_error_log(
            generate_random_string(), "PORTFOLIO", 
            get_line_number(), "MAIN", "holiday_bot.py", str(e)
        )

if __name__ == "__main__":
    main()