#!/usr/bin/env python3
"""Quick database size and prediction count checker"""

import sqlite3
import os
from datetime import datetime

def check_database():
    try:
        conn = sqlite3.connect('database/psc_trading.db')
        cursor = conn.cursor()
        
        # First check what tables exist
        cursor.execute('SELECT name FROM sqlite_master WHERE type="table"')
        tables = [row[0] for row in cursor.fetchall()]
        print(f"📋 Available tables: {tables}")
        
        # Check signals table for ML predictions
        if 'signals' in tables:
            table_name = 'signals'
        else:
            print("No signals table found!")
            return
            
        print(f"Using table: {table_name}")
        
        # Total ML signals (predictions)
        cursor.execute(f'SELECT COUNT(*) FROM {table_name}')
        total = cursor.fetchone()[0]
        
        # Check table structure first
        cursor.execute(f'PRAGMA table_info({table_name})')
        columns = [row[1] for row in cursor.fetchall()]
        print(f"   Table columns: {columns}")
        
        # ML signals only (checking what columns exist)
        if 'source' in columns:
            cursor.execute(f'SELECT COUNT(*) FROM {table_name} WHERE source LIKE "%ML%"')
            ml_signals = cursor.fetchone()[0]
        elif 'signal_type' in columns:
            cursor.execute(f'SELECT COUNT(*) FROM {table_name} WHERE signal_type LIKE "%ML%"')
            ml_signals = cursor.fetchone()[0]
        else:
            ml_signals = 0
        
        # Recent predictions (last hour)
        cursor.execute(f'SELECT COUNT(*) FROM {table_name} WHERE created_at > datetime("now", "-1 hour")')
        last_hour = cursor.fetchone()[0]
        
        # Very recent (last 10 minutes)
        cursor.execute(f'SELECT COUNT(*) FROM {table_name} WHERE created_at > datetime("now", "-10 minutes")')
        last_10min = cursor.fetchone()[0]
        
        # Database size
        size_mb = os.path.getsize('database/psc_trading.db') / 1024 / 1024
        
        print(f"📊 Database Analysis:")
        print(f"   Total signals: {total:,}")
        print(f"   ML signals: {ml_signals:,}")
        print(f"   Last hour: {last_hour}")
        print(f"   Last 10 minutes: {last_10min}")
        print(f"   Database size: {size_mb:.2f} MB")
        
        # Calculate growth rate
        if last_hour > 0:
            daily_rate = last_hour * 24
            print(f"   Projected daily: {daily_rate:,} predictions")
            print(f"   Projected yearly: {daily_rate * 365:,} predictions")
        
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_database()