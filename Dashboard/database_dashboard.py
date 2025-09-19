#!/usr/bin/env python3
"""
Database-Integrated Dashboard for PSC TON Trading System
Updated to use SQLite database instead of CSV files
"""

import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import logging
import sys
import os

# Add the parent directory to path for imports
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

try:
    from psc_data_manager import PSCDataManager
    from psc_database import PSCDatabase
except ImportError as e:
    st.error(f"Cannot import database modules: {e}")
    st.stop()

class DatabaseDashboard:
    """Dashboard that reads data from SQLite database instead of CSV files"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.db_path = project_root / "database" / "psc_trading.db"
        
        # Fallback to test database if main doesn't exist
        if not self.db_path.exists():
            self.db_path = project_root / "test_database.db"
            
        self.data_manager = None
        self.db_connection = None
        
        if self.db_path.exists():
            try:
                self.data_manager = PSCDataManager(str(self.db_path))
                self.db_connection = sqlite3.connect(str(self.db_path))
                st.success(f"✅ Connected to database: {self.db_path}")
            except Exception as e:
                st.error(f"❌ Database connection failed: {e}")
        else:
            st.error(f"❌ Database not found: {self.db_path}")
    
    def load_signals_from_db(self) -> pd.DataFrame:
        """Load signals from database"""
        if not self.db_connection:
            return pd.DataFrame()
            
        try:
            query = """
            SELECT id, timestamp, coin, signal_type, price, ratio, confidence, 
                   direction, exit_estimate, ml_prediction, signal_strength as strength, market_conditions
            FROM signals 
            ORDER BY timestamp DESC
            """
            df = pd.read_sql_query(query, self.db_connection)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception as e:
            st.error(f"Error loading signals: {e}")
            return pd.DataFrame()
    
    def load_trades_from_db(self) -> pd.DataFrame:
        """Load trades from database"""
        if not self.db_connection:
            return pd.DataFrame()
            
        try:
            query = """
            SELECT t.id, t.signal_id, t.timestamp, t.coin, t.trade_type,
                   t.side as action, t.entry_price, t.exit_price, t.quantity,
                   t.profit_pct, t.profit_usd, t.confidence, t.ml_prediction,
                   t.ratio, t.direction, t.exit_reason, t.status,
                   t.closed_at as exit_timestamp, s.signal_strength as strength, s.market_conditions
            FROM trades t
            LEFT JOIN signals s ON t.signal_id = s.id
            ORDER BY t.timestamp DESC
            """
            df = pd.read_sql_query(query, self.db_connection)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['exit_timestamp'] = pd.to_datetime(df['exit_timestamp'])
            return df
        except Exception as e:
            st.error(f"Error loading trades: {e}")
            return pd.DataFrame()
    
    def get_session_stats(self) -> dict:
        """Get session statistics from data manager"""
        if not self.data_manager:
            return {}
            
        try:
            return self.data_manager.get_session_stats()
        except Exception as e:
            st.error(f"Error getting session stats: {e}")
            return {}
    
    def get_performance_summary(self) -> dict:
        """Get performance summary from database"""
        if not self.db_connection:
            return {}
            
        try:
            cursor = self.db_connection.cursor()
            
            # Count totals
            cursor.execute("SELECT COUNT(*) FROM signals")
            total_signals = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'CLOSED'")
            total_trades = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'CLOSED' AND profit_pct > 0")
            profitable_trades = cursor.fetchone()[0]
            
            # Calculate profits
            cursor.execute("SELECT SUM(profit_pct), SUM(profit_usd) FROM trades WHERE status = 'CLOSED'")
            total_profit_pct, total_profit_usd = cursor.fetchone()
            
            success_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
            
            return {
                'total_signals': total_signals,
                'total_trades': total_trades,
                'profitable_trades': profitable_trades,
                'success_rate': success_rate,
                'total_profit_pct': total_profit_pct or 0,
                'total_profit_usd': total_profit_usd or 0
            }
        except Exception as e:
            st.error(f"Error getting performance summary: {e}")
            return {}

def main():
    st.set_page_config(
        page_title="PSC TON Trading Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📊 PSC TON Trading System Dashboard")
    st.markdown("**Database-Integrated Real-Time Monitoring**")
    
    # Initialize dashboard
    project_root = Path(__file__).parent.parent
    dashboard = DatabaseDashboard(project_root)
    
    # Sidebar
    st.sidebar.title("🎛️ Dashboard Controls")
    
    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh (30s)", value=False)
    if auto_refresh:
        st.rerun()
    
    # Manual refresh
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()
    
    # Time filter
    time_filter = st.sidebar.selectbox(
        "📅 Time Filter",
        ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "All Time"]
    )
    
    # Main content
    col1, col2, col3, col4 = st.columns(4)
    
    # Get performance summary
    perf_summary = dashboard.get_performance_summary()
    session_stats = dashboard.get_session_stats()
    
    with col1:
        st.metric(
            "🎯 Total Signals",
            perf_summary.get('total_signals', 0)
        )
    
    with col2:
        st.metric(
            "💼 Total Trades", 
            perf_summary.get('total_trades', 0)
        )
    
    with col3:
        st.metric(
            "✅ Success Rate",
            f"{perf_summary.get('success_rate', 0):.1f}%"
        )
    
    with col4:
        st.metric(
            "💰 Total Profit",
            f"${perf_summary.get('total_profit_usd', 0):.2f}"
        )
    
    # Load data
    st.subheader("📊 Recent Activity")
    
    # Signals
    signals_df = dashboard.load_signals_from_db()
    if not signals_df.empty:
        st.subheader("🎯 Recent Signals")
        st.dataframe(
            signals_df.head(10)[['timestamp', 'coin', 'direction', 'confidence', 'ratio', 'strength']],
            use_container_width=True
        )
    
    # Trades
    trades_df = dashboard.load_trades_from_db()
    if not trades_df.empty:
        st.subheader("💼 Recent Trades")
        st.dataframe(
            trades_df.head(10)[['timestamp', 'coin', 'direction', 'profit_pct', 'profit_usd', 'status']],
            use_container_width=True
        )
        
        # Performance chart
        if len(trades_df) > 0:
            st.subheader("📈 Profit Over Time")
            fig = px.line(
                trades_df[trades_df['status'] == 'CLOSED'].sort_values('timestamp'),
                x='timestamp',
                y='profit_pct',
                title="Profit Percentage Over Time"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Session stats
    if session_stats:
        st.subheader("📊 Session Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.json(session_stats)
        
        with col2:
            if 'daily_performance' in session_stats:
                daily_perf = session_stats['daily_performance']
                st.metric("Today's Signals", daily_perf.get('total_signals', 0))
                st.metric("Today's Trades", daily_perf.get('total_trades', 0))
                st.metric("Today's Profit", f"${daily_perf.get('total_profit_usd', 0):.2f}")
    
    # Database info
    st.sidebar.subheader("🗃️ Database Info")
    if dashboard.db_connection:
        st.sidebar.success("✅ Database Connected")
        st.sidebar.info(f"📁 {dashboard.db_path}")
        
        # Export button
        if st.sidebar.button("📥 Export to CSV"):
            try:
                csv_files = []
                for table in ['signals', 'trades']:
                    csv_file = dashboard.data_manager.db.export_to_csv(table)
                    csv_files.append(csv_file)
                st.sidebar.success(f"✅ Exported: {csv_files}")
            except Exception as e:
                st.sidebar.error(f"❌ Export failed: {e}")
    else:
        st.sidebar.error("❌ Database Disconnected")

if __name__ == "__main__":
    main()
