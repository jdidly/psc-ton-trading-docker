#!/usr/bin/env python3
"""
PSC TON Trading System - Minimal Dashboard
Works without external dependencies, provides basic functionality
"""

import json
import yaml
import time
from datetime import datetime, timedelta
from pathlib import Path
import logging
import sys
import os

# Add the parent directory to path for imports
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir / "src"))

print("🚀 PSC TON Trading System - Web Dashboard Alternative")
print("=" * 55)

# Check for Streamlit
try:
    import streamlit as st # type: ignore
    import pandas as pd
    import plotly.express as px
    STREAMLIT_AVAILABLE = True
    print("✅ Streamlit detected - Full web interface available")
except ImportError:
    STREAMLIT_AVAILABLE = False
    print("❌ Streamlit not available")
    print("📦 Install with: pip install streamlit pandas plotly")
    print("🔄 Using alternative interface...")

# Try to import trading system
try:
    from psc_ton_system import PSCTONTradingBot
    from src.ml_engine import MLEngine
    TRADING_SYSTEM_AVAILABLE = True
    print("✅ Trading system modules loaded")
except ImportError as e:
    TRADING_SYSTEM_AVAILABLE = False
    print(f"⚠️ Trading system import issue: {e}")

if not STREAMLIT_AVAILABLE:
    print("\n" + "="*55)
    print("🌐 WEB DASHBOARD SETUP GUIDE")
    print("="*55)
    print("To enable the full web dashboard:")
    print("1. Install required packages:")
    print("   pip install streamlit pandas plotly")
    print("2. Run the dashboard:")
    print("   streamlit run dashboard.py")
    print("3. Open your browser to: http://localhost:8501")
    
    print("\n📋 CURRENT SYSTEM STATUS:")
    print(f"   🔧 Python Version: {sys.version.split()[0]}")
    print(f"   📁 Working Directory: {Path.cwd()}")
    print(f"   🧠 ML Engine: {'✅ Available' if TRADING_SYSTEM_AVAILABLE else '❌ Not Found'}")
    print(f"   🤖 Trading Bot: {'✅ Available' if TRADING_SYSTEM_AVAILABLE else '❌ Not Found'}")
    
    print("\n🔄 ALTERNATIVE OPTIONS:")
    print("1. Use simple dashboard: python ../../simple_dashboard.py")
    print("2. Direct bot launch: python ../psc_ton_system.py") 
    print("3. Install web dependencies and retry")
    
    sys.exit(0)

# If we get here, Streamlit is available
if STREAMLIT_AVAILABLE:
    # Configure Streamlit page
    st.set_page_config(
        page_title="PSC TON Trading Dashboard",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Global variables for bot instance
    if 'trading_bot' not in st.session_state:
        st.session_state.trading_bot = None
    if 'bot_running' not in st.session_state:
        st.session_state.bot_running = False

    class TradingDashboard:
        def __init__(self):
            self.config_file = Path("../../config/settings.yaml")
            self.data_dir = Path("../../data")
            self.logs_dir = Path("../../logs")
            
        def load_config(self):
            """Load current configuration"""
            try:
                if self.config_file.exists():
                    with open(self.config_file, 'r') as f:
                        return yaml.safe_load(f)
                else:
                    return self.get_default_config()
            except Exception as e:
                st.error(f"Error loading config: {e}")
                return self.get_default_config()
        
        def get_default_config(self):
            """Default configuration values"""
            return {
                'trading': {
                    'scan_interval': 30,
                    'confidence_threshold': 0.7,
                    'ratio_threshold': 1.5,
                    'max_positions': 5,
                    'position_size': 1000,
                },
                'superp': {
                    'enabled': True,
                    'conservative_range': [1, 100],
                    'moderate_range': [100, 1000],
                    'aggressive_range': [1000, 5000],
                    'time_limit_minutes': 10
                },
                'ml': {
                    'enabled': True,
                    'retrain_interval': 50,
                    'confidence_boost': 0.1,
                }
            }
        
        def save_config(self, config):
            """Save configuration to file"""
            try:
                self.config_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.config_file, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, indent=2)
                return True
            except Exception as e:
                st.error(f"Error saving config: {e}")
                return False

    def main():
        """Main dashboard function"""
        dashboard = TradingDashboard()
        
        # Header
        st.title("🚀 PSC TON Trading System Dashboard")
        st.markdown("---")
        
        # System Status
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("System Status", "🟢 Online")
            
        with col2:
            st.metric("ML Engine", "✅ Ready" if TRADING_SYSTEM_AVAILABLE else "❌ Loading")
            
        with col3:
            st.metric("Trading Bot", "⏸️ Stopped")
        
        # Configuration Panel
        st.header("⚙️ Trading Configuration")
        
        config = dashboard.load_config()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Basic Settings")
            
            scan_interval = st.slider(
                "Scan Interval (seconds)",
                min_value=5,
                max_value=300,
                value=config['trading']['scan_interval'],
                step=5
            )
            
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.1,
                max_value=1.0,
                value=config['trading']['confidence_threshold'],
                step=0.05
            )
            
            position_size = st.number_input(
                "Position Size (USD)",
                min_value=100,
                max_value=10000,
                value=config['trading']['position_size'],
                step=100
            )
        
        with col2:
            st.subheader("🎢 Superp Settings")
            
            superp_enabled = st.checkbox(
                "Enable Superp Trading",
                value=config['superp']['enabled']
            )
            
            time_limit = st.number_input(
                "Time Limit (minutes)",
                min_value=1,
                max_value=60,
                value=config['superp']['time_limit_minutes']
            )
            
            ml_enabled = st.checkbox(
                "Enable ML Predictions",
                value=config['ml']['enabled']
            )
        
        # Save button
        if st.button("💾 Save Configuration"):
            config['trading']['scan_interval'] = scan_interval
            config['trading']['confidence_threshold'] = confidence_threshold
            config['trading']['position_size'] = position_size
            config['superp']['enabled'] = superp_enabled
            config['superp']['time_limit_minutes'] = time_limit
            config['ml']['enabled'] = ml_enabled
            
            if dashboard.save_config(config):
                st.success("✅ Configuration saved successfully!")
            else:
                st.error("❌ Failed to save configuration")
        
        # Quick Actions
        st.header("⚡ Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🧠 Test ML Engine"):
                if TRADING_SYSTEM_AVAILABLE:
                    try:
                        ml_engine = MLEngine()
                        prediction = ml_engine.predict_trade_outcome(1.5, 50.0, 1.5, 1000)
                        st.success(f"✅ ML Engine working! Prediction: {prediction['recommendation']}")
                    except Exception as e:
                        st.error(f"❌ ML Engine error: {e}")
                else:
                    st.error("❌ ML Engine not available")
        
        with col2:
            if st.button("📊 View Logs"):
                st.info("Log viewing would open here")
        
        with col3:
            if st.button("💰 Show Performance"):
                st.info("Performance analytics would show here")
        
        # Footer
        st.markdown("---")
        st.markdown("🚀 PSC TON Trading System | 🧠 ML-Powered | 🎢 Superp No-Liquidation")

    if __name__ == "__main__":
        main()
