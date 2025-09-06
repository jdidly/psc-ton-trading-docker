#!/usr/bin/env python3
"""
PSC TON Trading System Dashboard
Interactive web interface for monitoring and controlling the trading bot
"""

# Check for required dependencies first
missing_deps = []
try:
    import streamlit as st # type: ignore
except ImportError:
    missing_deps.append("streamlit")

try:
    import pandas as pd
except ImportError:
    missing_deps.append("pandas")

try:
    import plotly.graph_objs as go
    import plotly.express as px
    from plotly.subplots import make_subplots
except ImportError:
    missing_deps.append("plotly")

if missing_deps:
    print("❌ Missing required dependencies:")
    for dep in missing_deps:
        print(f"   - {dep}")
    print("\n📦 To install missing dependencies, run:")
    print(f"   pip install {' '.join(missing_deps)}")
    print("\n⚠️  If pip hangs, try using conda or manual installation")
    print("\n🔄 Falling back to simple dashboard...")
    print("Run: python ../../simple_dashboard.py")
    exit(1)

import json
import yaml
import asyncio
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
import logging
import sys
import os

# Add the parent directory to path for imports
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

def parse_timestamp_flexible(timestamp_value):
    """Parse timestamp from various formats (handles weird timestamps)"""
    if isinstance(timestamp_value, str):
        try:
            # Try ISO format first
            return pd.to_datetime(timestamp_value)
        except:
            # Try parsing as numeric string
            try:
                timestamp_value = float(timestamp_value)
            except:
                return pd.NaT
    
    if isinstance(timestamp_value, (int, float)):
        # Handle various timestamp formats
        if timestamp_value > 1e15:
            # Extremely large - might be microseconds or weird format
            timestamp_value = timestamp_value / 1000000
        elif timestamp_value > 1e12:
            # Likely milliseconds
            timestamp_value = timestamp_value / 1000
        
        try:
            return pd.to_datetime(timestamp_value, unit='s')
        except:
            return pd.NaT
    
    return pd.NaT
sys.path.insert(0, str(current_dir / "src"))

# Debug: Print current working directory and paths
print(f"🔍 Dashboard starting from: {Path.cwd()}")
print(f"🔍 Project root: {current_dir}")
print(f"🔍 Config file: {current_dir / 'config' / 'settings.yaml'}")
print(f"🔍 Data dir: {current_dir / 'data'}")
print(f"🔍 Absolute project root: {current_dir.absolute()}")
print(f"🔍 Dashboard file location: {Path(__file__).absolute()}")

try:
    from psc_ton_system import PSCTONTradingBot
    print("✅ PSC Trading Bot imported successfully")
    PSC_BOT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ PSC Bot import error: {e}")
    PSC_BOT_AVAILABLE = False

try:
    from src.ml_engine import MLEngine
    print("✅ ML Engine imported successfully") 
    ML_ENGINE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ ML Engine import error: {e}")
    ML_ENGINE_AVAILABLE = False

try:
    from src.models.live_microstructure_trainer import LiveMicrostructureTrainer
    print("✅ ML Microstructure Trainer imported successfully")
    ML_MICROSTRUCTURE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ ML Microstructure import error: {e}")
    ML_MICROSTRUCTURE_AVAILABLE = False

if not PSC_BOT_AVAILABLE or not ML_ENGINE_AVAILABLE:
    print("🔄 Some features may be limited - running in demo mode")
    
    # Create mock classes for dashboard to work
    class MockBot:
        def __init__(self):
            self.ml_engine = None
            self.open_positions = {}
    
    class MockMLEngine:
        def get_model_performance(self):
            return {'total_predictions': 0, 'accuracy': 0.0, 'model_status': 'Not available'}
    
    PSCTONTradingBot = MockBot
    MLEngine = MockMLEngine

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
if 'live_data' not in st.session_state:
    st.session_state.live_data = []

class TradingDashboard:
    def __init__(self):
        # Use absolute paths relative to the project root
        # Get the absolute path of the dashboard file, then go up one level
        dashboard_file = Path(__file__).absolute()
        self.project_root = dashboard_file.parent.parent
        self.config_file = self.project_root / "config" / "settings.yaml"
        self.data_dir = self.project_root / "data"
        self.logs_dir = self.project_root / "logs"
        
        # Debug: Print the paths we're actually using
        print(f"🔧 TradingDashboard initialized:")
        print(f"   Dashboard file: {dashboard_file}")
        print(f"   Project root: {self.project_root}")
        print(f"   Config file: {self.config_file}")
        print(f"   Data dir: {self.data_dir}")
        print(f"   Config exists: {self.config_file.exists()}")
        print(f"   Data dir exists: {self.data_dir.exists()}")
        
        # List actual files in data directory for debugging
        if self.data_dir.exists():
            csv_files = list(self.data_dir.glob("*.csv"))
            print(f"   CSV files found: {[f.name for f in csv_files]}")
        
    def load_config(self):
        """Load current configuration"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    return yaml.safe_load(f)
            else:
                return self.get_default_config()
        except Exception as e:
            if 'st' in globals():
                st.error(f"Error loading config: {e}")
            else:
                print(f"Error loading config: {e}")
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
                'stop_loss_pct': 5.0,
                'take_profit_pct': 10.0,
                'max_leverage': 1000,
                'min_leverage': 10
            },
            'superp': {
                'enabled': True,
                'conservative_range': [1, 100],
                'moderate_range': [100, 1000],
                'aggressive_range': [1000, 5000],
                'extreme_range': [5000, 10000],
                'time_limit_minutes': 10
            },
            'ml': {
                'enabled': True,
                'retrain_interval': 50,
                'confidence_boost': 0.1,
                'feature_count': 9
            },
            'telegram': {
                'enabled': False,
                'send_signals': True,
                'send_trades': True,
                'send_status': True
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
            if 'st' in globals():
                st.error(f"Error saving config: {e}")
            else:
                print(f"Error saving config: {e}")
            return False
    
    def load_trading_data(self):
        """Load trading data for display"""
        data = {}
        
        # Load trade logs
        trade_file = self.data_dir / "live_trades.csv"
        if trade_file.exists():
            try:
                data['trades'] = pd.read_csv(trade_file)
                # Convert timestamp to datetime if it exists
                if 'timestamp' in data['trades'].columns:
                    data['trades']['timestamp'] = data['trades']['timestamp'].apply(parse_timestamp_flexible)
                print(f"✅ Loaded {len(data['trades'])} trades from {trade_file}")
            except Exception as e:
                print(f"❌ Error loading trades: {e}")
                data['trades'] = pd.DataFrame()
        else:
            print(f"❌ Trade file not found: {trade_file}")
            data['trades'] = pd.DataFrame()
        
        # Load signal logs  
        signal_file = self.data_dir / "psc_signals.csv"
        if signal_file.exists():
            try:
                data['signals'] = pd.read_csv(signal_file)
                # Convert timestamp to datetime if it exists
                if 'timestamp' in data['signals'].columns:
                    data['signals']['timestamp'] = data['signals']['timestamp'].apply(parse_timestamp_flexible)
                print(f"✅ Loaded {len(data['signals'])} PSC signals from {signal_file}")
            except Exception as e:
                print(f"❌ Error loading PSC signals: {e}")
                data['signals'] = pd.DataFrame()
        else:
            print(f"❌ PSC signal file not found: {signal_file}")
            data['signals'] = pd.DataFrame()
        
        # Load ML signals (this has the most data - 639 lines)
        ml_signal_file = self.data_dir / "ml_signals.csv"
        if ml_signal_file.exists():
            try:
                data['ml_signals'] = pd.read_csv(ml_signal_file)
                # Convert timestamp to datetime if it exists
                if 'timestamp' in data['ml_signals'].columns:
                    data['ml_signals']['timestamp'] = data['ml_signals']['timestamp'].apply(parse_timestamp_flexible)
                print(f"✅ Loaded {len(data['ml_signals'])} ML signals from {ml_signal_file}")
            except Exception as e:
                print(f"❌ Error loading ML signals: {e}")
                data['ml_signals'] = pd.DataFrame()
        else:
            print(f"❌ ML signal file not found: {ml_signal_file}")
            data['ml_signals'] = pd.DataFrame()
        
        # Load prediction data
        pred_file = self.data_dir / "ml" / "prediction_history.json"
        if pred_file.exists():
            try:
                with open(pred_file, 'r') as f:
                    pred_data = json.load(f)
                    data['predictions'] = pred_data.get('predictions', [])
                print(f"✅ Loaded {len(data['predictions'])} predictions from {pred_file}")
            except Exception as e:
                print(f"❌ Error loading predictions: {e}")
                data['predictions'] = []
        else:
            print(f"❌ Prediction file not found: {pred_file}")
            data['predictions'] = []
        
        return data
    
    def get_system_status(self):
        """Get current system status"""
        # Check actual bot process status
        actual_bot_running = check_bot_status() if 'check_bot_status' in globals() else st.session_state.get('bot_running', False)
        
        status = {
            'bot_running': actual_bot_running,
            'ml_engine': ML_ENGINE_AVAILABLE,
            'psc_bot': PSC_BOT_AVAILABLE,
            'config_loaded': self.config_file.exists(),
            'data_available': False,
            'total_trades': 0,
            'total_signals': 0,
            'last_update': 'Never',
            'config_path': str(self.config_file),
            'data_path': str(self.data_dir)
        }
        
        # Update session state to match actual status
        st.session_state.bot_running = actual_bot_running
        
        # Check data availability
        try:
            data = self.load_trading_data()
            
            # Count all available data
            trades_count = len(data['trades']) if not data['trades'].empty else 0
            signals_count = len(data['signals']) if not data['signals'].empty else 0
            ml_signals_count = len(data['ml_signals']) if 'ml_signals' in data and not data['ml_signals'].empty else 0
            predictions_count = len(data['predictions']) if data['predictions'] else 0
            
            # Update status
            status['total_trades'] = trades_count
            status['total_signals'] = signals_count + ml_signals_count  # Combine both signal types
            status['total_predictions'] = predictions_count
            
            # Data is available if we have any of these
            if trades_count > 0 or signals_count > 0 or ml_signals_count > 0 or predictions_count > 0:
                status['data_available'] = True
                
                # Get latest timestamp from any available data
                latest_timestamps = []
                if not data['trades'].empty and 'timestamp' in data['trades'].columns:
                    latest_timestamps.append(data['trades']['timestamp'].iloc[-1])
                if not data['signals'].empty and 'timestamp' in data['signals'].columns:
                    latest_timestamps.append(data['signals']['timestamp'].iloc[-1])
                if 'ml_signals' in data and not data['ml_signals'].empty and 'timestamp' in data['ml_signals'].columns:
                    latest_timestamps.append(data['ml_signals']['timestamp'].iloc[-1])
                
                if latest_timestamps:
                    status['last_update'] = max(latest_timestamps)
                else:
                    status['last_update'] = 'Data exists'
            
        except Exception as e:
            print(f"Error checking data: {e}")
            status['data_error'] = str(e)
        
        return status

def main():
    dashboard = TradingDashboard()
    
    # Header
    st.title("🚀 PSC TON Trading System Dashboard")
    
    # Connection Status Banner
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if PSC_BOT_AVAILABLE:
            st.success("✅ PSC Bot Connected")
        else:
            st.error("❌ PSC Bot Unavailable")
    
    with col2:
        if ML_ENGINE_AVAILABLE:
            st.success("✅ ML Engine Connected") 
        else:
            st.error("❌ ML Engine Unavailable")
    
    with col3:
        if dashboard.config_file.exists():
            st.success("✅ Config Loaded")
        else:
            st.warning("⚠️ Config Missing")
    
    with col4:
        data_check = dashboard.load_trading_data()
        if not data_check['trades'].empty or not data_check['signals'].empty:
            st.success("✅ Data Available")
        else:
            st.warning("⚠️ No Trading Data")
    
    # Prediction Validation Status Banner
    st.subheader("🎯 Prediction Validation System")
    try:
        predictions_file = dashboard.data_dir / "ml_predictions.csv"
        validation_file = dashboard.data_dir / "prediction_validation.csv"
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if predictions_file.exists():
                predictions_df = pd.read_csv(predictions_file)
                st.metric("Total Predictions", len(predictions_df))
            else:
                st.metric("Total Predictions", "0")
        
        with col2:
            if validation_file.exists():
                validation_df = pd.read_csv(validation_file)
                validated_count = len(validation_df)
                st.metric("Validated", validated_count)
            else:
                st.metric("Validated", "0")
        
        with col3:
            if validation_file.exists() and not validation_df.empty:
                success_rate = len(validation_df[validation_df['success'] == True]) / len(validation_df) * 100
                st.metric("Success Rate", f"{success_rate:.1f}%")
            else:
                st.metric("Success Rate", "Pending")
        
        with col4:
            if validation_file.exists() and 'actual_profit_pct' in validation_df.columns:
                total_profit = validation_df['actual_profit_pct'].sum()
                st.metric("Total Profit %", f"{total_profit:.2f}%")
            else:
                st.metric("Total Profit %", "Pending")
    
    except Exception:
        # If there's any error, show a simple status
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Prediction System", "🔄 Initializing")
        with col2:
            st.info("Enhanced prediction validation system is ready to track ML prediction accuracy.")
    
    st.markdown("---")
    
    # Sidebar for controls
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # System Status
        st.subheader("📊 System Status")
        status = dashboard.get_system_status()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Bot Status", "🟢 Running" if status['bot_running'] else "🔴 Stopped")
            st.metric("ML Engine", "✅ Active" if status['ml_engine'] else "❌ Inactive")
        with col2:
            st.metric("Total Trades", status['total_trades'])
            st.metric("Total Signals", status['total_signals'])
            
        # ML Microstructure Status
        microstructure_status = "🟢 Active" if st.session_state.get('microstructure_running', False) else "🔴 Stopped"
        microstructure_available = "✅ Available" if ML_MICROSTRUCTURE_AVAILABLE else "❌ Unavailable"
        st.metric("ML Microstructure", f"{microstructure_status} ({microstructure_available})")
        
        # Diagnostic Info
        with st.expander("🔍 System Diagnostics"):
            st.text(f"PSC Bot: {'✅ Available' if status['psc_bot'] else '❌ Unavailable'}")
            st.text(f"ML Microstructure: {'✅ Available' if ML_MICROSTRUCTURE_AVAILABLE else '❌ Unavailable'}")
            st.text(f"Config: {'✅ Loaded' if status['config_loaded'] else '❌ Missing'}")
            st.text(f"Data: {'✅ Available' if status['data_available'] else '❌ No data'}")
            st.text(f"Config Path: {status['config_path']}")
            st.text(f"Data Path: {status['data_path']}")
            st.text(f"Last Update: {status['last_update']}")
            st.text(f"0.1% Profit Alignment: ✅ Enabled")
            if st.button("🔄 Refresh Status"):
                st.rerun()
        
        # Bot Controls
        st.subheader("🎮 Bot Controls")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Start Bot", disabled=status['bot_running']):
                with st.spinner("Starting bot..."):
                    start_bot()
        with col2:
            if st.button("⏹️ Stop Bot", disabled=not status['bot_running']):
                with st.spinner("Stopping bot..."):
                    stop_bot()
        
        if st.button("🔄 Restart Bot"):
            with st.spinner("Restarting bot..."):
                restart_bot()
        
        # Debug info for troubleshooting
        with st.expander("🔧 Debug Info"):
            st.write(f"Session bot_running: {st.session_state.get('bot_running', False)}")
            
            # Process-based bot info
            if 'bot_process' in st.session_state and st.session_state.bot_process:
                proc = st.session_state.bot_process
                st.write(f"Process ID: {proc.pid}")
                st.write(f"Process running: {proc.poll() is None}")
                st.write(f"Return code: {proc.returncode}")
            
            # Path debugging
            dashboard_dir = Path(__file__).parent.absolute()
            project_root = dashboard_dir.parent
            bot_script = project_root / "run_bot.py"
            test_script = project_root / "test_bot_start.py"
            
            st.write(f"Dashboard dir: {dashboard_dir}")
            st.write(f"Project root: {project_root}")
            st.write(f"Bot script exists: {bot_script.exists()}")
            st.write(f"Test script exists: {test_script.exists()}")
            
            # Test process button
            if st.button("🧪 Test Process Start"):
                try:
                    import subprocess
                    import os
                    
                    python_exe = project_root / ".venv" / "Scripts" / "python.exe"
                    if not python_exe.exists():
                        python_exe = "python"
                    
                    # Use test script
                    cmd = [str(python_exe), str(test_script)]
                    working_dir = str(project_root)
                    
                    st.info(f"Testing command: {' '.join(cmd)}")
                    st.info(f"Working dir: {working_dir}")
                    
                    process = subprocess.Popen(
                        cmd,
                        cwd=working_dir,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        env=os.environ.copy()
                    )
                    
                    # Wait for completion
                    stdout, stderr = process.communicate(timeout=30)
                    
                    if process.returncode == 0:
                        st.success("✅ Test process completed successfully!")
                        st.text("Output:")
                        st.code(stdout)
                    else:
                        st.error(f"❌ Test process failed (exit code: {process.returncode})")
                        if stderr:
                            st.error("Error output:")
                            st.code(stderr)
                        if stdout:
                            st.info("Standard output:")
                            st.code(stdout)
                            
                except subprocess.TimeoutExpired:
                    st.warning("⚠️ Test process timed out")
                    process.kill()
                except Exception as e:
                    st.error(f"❌ Test failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())
            if 'bot_process' in st.session_state and st.session_state.bot_process is not None:
                process = st.session_state.bot_process
                st.write(f"Bot process exists: True")
                st.write(f"Process ID: {process.pid}")
                st.write(f"Process running: {process.poll() is None}")
                st.write(f"Return code: {process.poll()}")
            else:
                st.write(f"Bot process exists: False")
            
            # Legacy bot instance info (if exists)
            if st.session_state.get('trading_bot'):
                bot = st.session_state.trading_bot
                st.write(f"Legacy bot instance: True")
                if hasattr(bot, 'running'):
                    st.write(f"Legacy bot.running: {bot.running}")
            else:
                st.write(f"Legacy bot instance: False")
        
        # Quick Actions
        st.subheader("⚡ Quick Actions")
        
        if st.button("📊 Refresh Data"):
            st.rerun()
        
        if st.button("🧠 Retrain ML"):
            retrain_ml_models()
        
        if st.button("💾 Export Data"):
            export_trading_data(dashboard)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["⚙️ Configuration", "📈 Trading Monitor", "🧠 ML Analytics", "⚡ ML Microstructure", "📋 Logs", "📊 Performance"])
    
    with tab1:
        show_configuration_panel(dashboard)
    
    with tab2:
        show_trading_monitor(dashboard)
    
    with tab3:
        show_ml_analytics(dashboard)
    
    with tab4:
        show_ml_microstructure(dashboard)
    
    with tab5:
        show_logs_panel(dashboard)
    
    with tab6:
        show_performance_panel(dashboard)

def show_configuration_panel(dashboard):
    """Configuration panel for trading parameters"""
    st.header("⚙️ Trading Configuration")
    
    config = dashboard.load_config()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Trading Parameters")
        
        # Scanning parameters with safe fallbacks
        scan_interval = st.slider(
            "Scan Interval (seconds)",
            min_value=5,
            max_value=300,
            value=config.get('trading', {}).get('scan_interval', 30),
            step=5,
            help="How often to scan for trading opportunities"
        )
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=config.get('trading', {}).get('confidence_threshold', 0.7),
            step=0.05,
            help="Minimum confidence required to open a position"
        )
        
        ratio_threshold = st.slider(
            "Ratio Threshold",
            min_value=1.0,
            max_value=5.0,
            value=config.get('trading', {}).get('ratio_threshold', 1.25),
            step=0.1,
            help="Minimum PSC/TON ratio to consider"
        )
        
        max_positions = st.number_input(
            "Max Concurrent Positions",
            min_value=1,
            max_value=20,
            value=config.get('trading', {}).get('max_positions', 5),
            help="Maximum number of open positions"
        )
        
        position_size = st.number_input(
            "Position Size (USD)",
            min_value=100,
            max_value=10000,
            value=config.get('trading', {}).get('position_size', 1000),
            step=100,
            help="Default position size in USD"
        )
    
    with col2:
        st.subheader("🎢 Superp Leverage Settings")
        
        superp_enabled = st.checkbox(
            "Enable Superp No-Liquidation",
            value=config.get('superp', {}).get('enabled', True),
            help="Enable Superp extreme leverage trading"
        )
        
        col2a, col2b = st.columns(2)
        with col2a:
            conservative_max = st.number_input(
                "Conservative Max (x)",
                min_value=1,
                max_value=1000,
                value=config.get('superp', {}).get('conservative_range', [1, 100])[1],
                help="Maximum leverage for conservative trades"
            )
            
            moderate_max = st.number_input(
                "Moderate Max (x)",
                min_value=100,
                max_value=5000,
                value=config.get('superp', {}).get('moderate_range', [100, 1000])[1],
                help="Maximum leverage for moderate confidence trades"
            )
        
        with col2b:
            aggressive_max = st.number_input(
                "Aggressive Max (x)",
                min_value=1000,
                max_value=10000,
                value=config.get('superp', {}).get('aggressive_range', [1000, 5000])[1],
                help="Maximum leverage for aggressive trades"
            )
            
            time_limit = st.number_input(
                "Time Limit (minutes)",
                min_value=1,
                max_value=60,
                value=config.get('superp', {}).get('time_limit_minutes', 10),
                help="Maximum time to hold Superp positions"
            )
        
        st.subheader("🧠 ML Configuration")
        
        ml_enabled = st.checkbox(
            "Enable ML Predictions",
            value=config.get('ml', {}).get('enabled', True),
            help="Use ML engine for trade predictions"
        )
        
        retrain_interval = st.number_input(
            "Retrain Interval",
            min_value=10,
            max_value=200,
            value=config.get('ml', {}).get('retrain_interval', 50),
            help="Retrain models every N predictions"
        )
    
    # Save configuration
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("💾 Save Configuration"):
            # Update config with new values
            config['trading']['scan_interval'] = scan_interval
            config['trading']['confidence_threshold'] = confidence_threshold
            config['trading']['ratio_threshold'] = ratio_threshold
            config['trading']['max_positions'] = max_positions
            config['trading']['position_size'] = position_size
            
            config['superp']['enabled'] = superp_enabled
            config['superp']['conservative_range'][1] = conservative_max
            config['superp']['moderate_range'][1] = moderate_max
            config['superp']['aggressive_range'][1] = aggressive_max
            config['superp']['time_limit_minutes'] = time_limit
            
            config['ml']['enabled'] = ml_enabled
            config['ml']['retrain_interval'] = retrain_interval
            
            if dashboard.save_config(config):
                st.success("✅ Configuration saved successfully!")
            else:
                st.error("❌ Failed to save configuration")
    
    with col2:
        if st.button("🔄 Load Defaults"):
            default_config = dashboard.get_default_config()
            dashboard.save_config(default_config)
            st.success("✅ Default configuration loaded!")
            st.rerun()

def show_trading_monitor(dashboard):
    """Real-time trading monitor"""
    st.header("📈 Trading Monitor")
    
    data = dashboard.load_trading_data()
    
    # Real-time metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if not data['trades'].empty and 'profit_usd' in data['trades'].columns:
            total_profit = data['trades']['profit_usd'].sum()
            st.metric("Total Profit", f"${total_profit:.2f}")
        else:
            st.metric("Total Profit", "$0.00")
    
    with col2:
        if not data['trades'].empty:
            successful_trades = len(data['trades'][data['trades']['successful'] == True]) if 'successful' in data['trades'].columns else 0
            success_rate = (successful_trades / len(data['trades']) * 100) if len(data['trades']) > 0 else 0
            st.metric("Success Rate", f"{success_rate:.1f}%")
        else:
            st.metric("Success Rate", "0%")
    
    with col3:
        active_positions = 0
        if st.session_state.trading_bot:
            try:
                active_positions = len([p for p in st.session_state.trading_bot.open_positions.values() if p.get('status') == 'ACTIVE'])
            except:
                active_positions = 0
        st.metric("Active Positions", active_positions)
    
    with col4:
        if not data['signals'].empty and 'timestamp' in data['signals'].columns:
            try:
                # Compare with datetime object (timestamps are already converted)
                cutoff_time = datetime.now() - timedelta(hours=1)
                recent_signals = len(data['signals'][data['signals']['timestamp'] > cutoff_time])
                st.metric("Signals (1h)", recent_signals)
            except Exception as e:
                st.metric("Signals (1h)", "Error")
                print(f"Error checking recent signals: {e}")
        else:
            st.metric("Signals (1h)", 0)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Recent Trades")
        if not data['trades'].empty:
            # Display recent trades table
            recent_trades = data['trades'].tail(10)
            if not recent_trades.empty:
                display_columns = ['timestamp', 'coin', 'profit_pct', 'successful', 'confidence']
                available_columns = [col for col in display_columns if col in recent_trades.columns]
                if available_columns:
                    st.dataframe(recent_trades[available_columns], use_container_width=True)
                else:
                    st.dataframe(recent_trades, use_container_width=True)
        else:
            st.info("No trading data available yet")
    
    with col2:
        st.subheader("🎯 Recent Signals")
        if not data['signals'].empty:
            recent_signals = data['signals'].tail(10)
            if not recent_signals.empty:
                display_columns = ['timestamp', 'coin', 'ratio', 'confidence', 'direction']
                available_columns = [col for col in display_columns if col in recent_signals.columns]
                if available_columns:
                    st.dataframe(recent_signals[available_columns], use_container_width=True)
                else:
                    st.dataframe(recent_signals, use_container_width=True)
        else:
            st.info("No signal data available yet")
    
    # Profit chart
    if not data['trades'].empty and 'timestamp' in data['trades'].columns and 'profit_usd' in data['trades'].columns:
        st.subheader("💰 Cumulative Profit")
        
        # Calculate cumulative profit
        trades_df = data['trades'].copy()
        try:
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'], format='mixed', errors='coerce')
        except:
            # If timestamp parsing fails, keep as string and skip time-based sorting
            pass
        
        if pd.api.types.is_datetime64_any_dtype(trades_df['timestamp']):
            trades_df = trades_df.sort_values('timestamp')
        
        trades_df['cumulative_profit'] = trades_df['profit_usd'].cumsum()
        
        fig = px.line(trades_df, x='timestamp', y='cumulative_profit', title='Cumulative Profit Over Time')
        fig.update_layout(yaxis_title='Profit (USD)', xaxis_title='Time')
        st.plotly_chart(fig, width='stretch', key='cumulative_profit_chart')

def show_ml_analytics(dashboard):
    """ML engine analytics and performance"""
    st.header("🧠 ML Analytics")
    
    data = dashboard.load_trading_data()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 ML Performance")
        
        if st.session_state.trading_bot and st.session_state.trading_bot.ml_engine:
            try:
                performance = st.session_state.trading_bot.ml_engine.get_model_performance()
                
                st.metric("Total Predictions", performance.get('total_predictions', 0))
                st.metric("Overall Accuracy", f"{performance.get('overall_accuracy', 0):.1%}")
                st.metric("High Confidence Accuracy", f"{performance.get('high_confidence_accuracy', 0):.1%}")
                st.metric("Model Status", performance.get('model_status', 'Unknown'))
                
            except Exception as e:
                st.error(f"Error getting ML performance: {e}")
        else:
            st.info("ML engine not available")
    
    with col2:
        st.subheader("📈 Prediction History")
        
        if data['predictions']:
            pred_df = pd.DataFrame(data['predictions'])
            
            if not pred_df.empty and 'timestamp' in pred_df.columns:
                try:
                    pred_df['timestamp'] = pd.to_datetime(pred_df['timestamp'], format='mixed', errors='coerce')
                except:
                    # If timestamp parsing fails, keep as string
                    pass
                recent_predictions = pred_df.tail(10)
                
                display_columns = ['timestamp', 'prediction', 'actual_outcome']
                available_columns = [col for col in display_columns if col in recent_predictions.columns]
                
                if available_columns:
                    st.dataframe(recent_predictions[available_columns], use_container_width=True)
                else:
                    st.dataframe(recent_predictions, use_container_width=True)
        else:
            st.info("No prediction data available")
    
    # ML Controls
    st.subheader("🎛️ ML Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Retrain Models"):
            if st.session_state.trading_bot and st.session_state.trading_bot.ml_engine:
                try:
                    success = st.session_state.trading_bot.ml_engine.retrain_models()
                    if success:
                        st.success("✅ Models retrained successfully!")
                    else:
                        st.warning("⚠️ Not enough data for retraining")
                except Exception as e:
                    st.error(f"❌ Retraining failed: {e}")
            else:
                st.error("ML engine not available")
    
    with col2:
        if st.button("💾 Save Models"):
            if st.session_state.trading_bot and st.session_state.trading_bot.ml_engine:
                try:
                    st.session_state.trading_bot.ml_engine.save_models()
                    st.success("✅ Models saved successfully!")
                except Exception as e:
                    st.error(f"❌ Save failed: {e}")
            else:
                st.error("ML engine not available")
    
    with col3:
        if st.button("📚 Load Models"):
            if st.session_state.trading_bot and st.session_state.trading_bot.ml_engine:
                try:
                    st.session_state.trading_bot.ml_engine.load_models()
                    st.success("✅ Models loaded successfully!")
                except Exception as e:
                    st.error(f"❌ Load failed: {e}")
            else:
                st.error("ML engine not available")

def show_ml_microstructure(dashboard):
    """ML Microstructure system control and monitoring"""
    st.header("⚡ ML Microstructure System")
    
    # Initialize microstructure trainer if not exists
    if 'ml_microstructure' not in st.session_state:
        if ML_MICROSTRUCTURE_AVAILABLE:
            try:
                st.session_state.ml_microstructure = LiveMicrostructureTrainer()
                st.session_state.microstructure_running = False
            except Exception as e:
                st.error(f"Failed to initialize ML Microstructure: {e}")
                st.session_state.ml_microstructure = None
        else:
            st.session_state.ml_microstructure = None
            
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 System Status")
        
        if st.session_state.ml_microstructure:
            # Status indicators
            status = "🟢 Active" if st.session_state.get('microstructure_running', False) else "🔴 Stopped"
            st.metric("Status", status)
            
            # Confidence methodology info
            st.info("🎯 **Confidence = 0.1% Profit Probability**\n\nConfidence scores represent the probability of achieving >0.1% profit, aligned with system break-even threshold.")
            
            # Control buttons
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("▶️ Start ML Microstructure", disabled=st.session_state.get('microstructure_running', False)):
                    st.session_state.microstructure_running = True
                    st.success("✅ ML Microstructure started!")
                    
            with col_b:
                if st.button("⏹️ Stop ML Microstructure", disabled=not st.session_state.get('microstructure_running', False)):
                    st.session_state.microstructure_running = False
                    st.success("✅ ML Microstructure stopped!")
                    
        else:
            st.error("❌ ML Microstructure not available")
            st.info("Please ensure all dependencies are installed")
            
    with col2:
        st.subheader("⚙️ Configuration")
        
        if st.session_state.ml_microstructure:
            # Display current configuration
            config = dashboard.load_config()
            ml_config = config.get('ml_microstructure', {})
            
            st.write("**Current Settings:**")
            st.write(f"• Min Confidence: {ml_config.get('min_confidence', 0.5):.1%}")
            st.write(f"• Min PSC Ratio (LONG): {ml_config.get('min_signal_ratio', 7.0)}")
            st.write(f"• Min PSC Ratio (SHORT): {ml_config.get('min_signal_ratio', 7.0) - 2.0}")
            st.write(f"• Timer Window: {ml_config.get('timer_window_minutes', 10)} minutes")
            
            # Advanced settings
            with st.expander("🔧 Advanced Settings"):
                new_min_confidence = st.slider("Min Confidence Threshold", 0.3, 0.8, ml_config.get('min_confidence', 0.5), 0.05)
                new_min_ratio = st.slider("Min PSC Ratio (LONG)", 6.0, 9.0, ml_config.get('min_signal_ratio', 7.0), 0.1)
                
                if st.button("💾 Save Settings"):
                    # Update configuration
                    config['ml_microstructure'] = config.get('ml_microstructure', {})
                    config['ml_microstructure']['min_confidence'] = new_min_confidence
                    config['ml_microstructure']['min_signal_ratio'] = new_min_ratio
                    
                    try:
                        dashboard.save_config(config)
                        st.success("✅ Settings saved!")
                    except Exception as e:
                        st.error(f"❌ Failed to save: {e}")
    
    # Live monitoring section
    if st.session_state.get('microstructure_running', False) and st.session_state.ml_microstructure:
        st.subheader("📊 Live Monitoring")
        
        # Create monitoring columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Signals Generated", st.session_state.get('total_microstructure_signals', 0))
            
        with col2:
            st.metric("Avg Confidence", f"{st.session_state.get('avg_microstructure_confidence', 0.0):.1%}")
            
        with col3:
            st.metric("Success Rate", f"{st.session_state.get('microstructure_success_rate', 0.0):.1%}")
            
        # Signal generation test
        if st.button("🧪 Test Signal Generation"):
            with st.spinner("Generating test signals..."):
                try:
                    # Run a quick test to generate signals
                    test_results = run_microstructure_test()
                    
                    if test_results:
                        st.success(f"✅ Generated {len(test_results)} test signals")
                        
                        # Display recent signals
                        if test_results:
                            st.subheader("🔍 Recent Test Signals")
                            signals_df = pd.DataFrame([
                                {
                                    'Symbol': signal.symbol,
                                    'Direction': signal.direction,
                                    'PSC Ratio': f"{signal.psc_ratio:.3f}",
                                    'Confidence': f"{signal.confidence_score:.1%}",
                                    'Leverage': f"{signal.leverage:.0f}x",
                                    'Strength': signal.signal_strength
                                }
                                for signal in test_results[:10]  # Show top 10
                            ])
                            st.dataframe(signals_df, use_container_width=True)
                    else:
                        st.warning("⚠️ No signals generated in test")
                        
                except Exception as e:
                    st.error(f"❌ Test failed: {e}")

def run_microstructure_test():
    """Run a quick test of the ML microstructure system"""
    try:
        trainer = st.session_state.ml_microstructure
        if not trainer:
            return []
            
        # Generate test signals for monitoring coins
        test_signals = []
        
        for coin in trainer.monitoring_coins:
            try:
                # Simulate getting a signal (this would normally come from live data)
                signal = trainer.generate_signals_for_coin(coin['symbol'])
                if signal:
                    test_signals.append(signal)
            except Exception as e:
                st.write(f"Debug: Error for {coin['symbol']}: {e}")
                continue
                
        return test_signals
        
    except Exception as e:
        st.error(f"Microstructure test error: {e}")
        return []

def show_logs_panel(dashboard):
    """System logs and debugging information"""
    st.header("📋 System Logs")
    
    # Log level selector
    log_level = st.selectbox("Log Level", ["INFO", "DEBUG", "WARNING", "ERROR"], index=0)
    
    # Auto-refresh toggle
    auto_refresh = st.checkbox("Auto-refresh logs", value=False)
    
    if auto_refresh:
        # Auto-refresh every 5 seconds
        placeholder = st.empty()
        
        for i in range(12):  # 1 minute of auto-refresh
            with placeholder.container():
                display_logs(dashboard, log_level)
            time.sleep(5)
    else:
        display_logs(dashboard, log_level)
    
    # Manual refresh button
    if st.button("🔄 Refresh Logs"):
        st.rerun()

def display_logs(dashboard, log_level):
    """Display system logs"""
    try:
        # Check for log files
        log_files = []
        
        # System log
        system_log = dashboard.logs_dir / "hybrid_system.log"
        if system_log.exists():
            log_files.append(("System Log", system_log))
        
        # Trading log (if exists)
        trading_log = dashboard.logs_dir / "trading.log"
        if trading_log.exists():
            log_files.append(("Trading Log", trading_log))
        
        if not log_files:
            st.info("No log files found")
            return
        
        # Display logs in tabs
        if len(log_files) > 1:
            tabs = st.tabs([name for name, _ in log_files])
            for tab, (name, log_file) in zip(tabs, log_files):
                with tab:
                    show_log_content(log_file, log_level)
        else:
            show_log_content(log_files[0][1], log_level)
            
    except Exception as e:
        st.error(f"Error displaying logs: {e}")

def show_log_content(log_file, log_level):
    """Show content of a specific log file"""
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # Filter by log level
        filtered_lines = []
        for line in lines:
            if log_level in line or log_level == "INFO":
                filtered_lines.append(line.strip())
        
        # Show last 100 lines
        recent_lines = filtered_lines[-100:]
        
        if recent_lines:
            log_text = '\n'.join(recent_lines)
            st.text_area("Log Content", value=log_text, height=400, key=f"log_{log_file.name}")
        else:
            st.info(f"No {log_level} level logs found")
            
    except Exception as e:
        st.error(f"Error reading log file: {e}")

def show_performance_panel(dashboard):
    """Performance analytics and statistics"""
    st.header("📊 Performance Analytics")
    
    # Add tabs for different performance views
    tab1, tab2, tab3 = st.tabs(["📈 Trading Performance", "🎯 Prediction Validation", "🤖 ML Analytics"])
    
    with tab1:
        show_trading_performance(dashboard)
    
    with tab2:
        show_prediction_validation(dashboard)
    
    with tab3:
        show_ml_analytics(dashboard)

def show_trading_performance(dashboard):
    """Original trading performance metrics"""
    data = dashboard.load_trading_data()
    
    if data['trades'].empty:
        st.info("No trading data available for performance analysis")
        return
    
    trades_df = data['trades']
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'profit_usd' in trades_df.columns:
            total_profit = trades_df['profit_usd'].sum()
            avg_profit = trades_df['profit_usd'].mean()
            st.metric("Total Profit", f"${total_profit:.2f}")
            st.metric("Avg Profit/Trade", f"${avg_profit:.2f}")
    
    with col2:
        if 'successful' in trades_df.columns:
            win_rate = (trades_df['successful'].sum() / len(trades_df)) * 100
            total_trades = len(trades_df)
            st.metric("Win Rate", f"{win_rate:.1f}%")
            st.metric("Total Trades", total_trades)
    
    with col3:
        if 'profit_pct' in trades_df.columns:
            max_gain = trades_df['profit_pct'].max()
            max_loss = trades_df['profit_pct'].min()
            st.metric("Best Trade", f"{max_gain:.2f}%")
            st.metric("Worst Trade", f"{max_loss:.2f}%")
    
    with col4:
        if 'confidence' in trades_df.columns:
            avg_confidence = trades_df['confidence'].mean()
            high_conf_trades = len(trades_df[trades_df['confidence'] > 0.8])
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
            st.metric("High Conf Trades", high_conf_trades)
    
    # Performance charts
    col1, col2 = st.columns(2)
    
    with col1:
        if 'profit_pct' in trades_df.columns:
            st.subheader("📈 Profit Distribution")
            fig = px.histogram(trades_df, x='profit_pct', nbins=20, title='Trade Profit Distribution')
            fig.update_layout(xaxis_title='Profit %', yaxis_title='Number of Trades')
            st.plotly_chart(fig, use_container_width=True, key='profit_distribution_hist')
    
    with col2:
        if 'confidence' in trades_df.columns and 'successful' in trades_df.columns:
            st.subheader("🎯 Confidence vs Success")
            fig = px.scatter(trades_df, x='confidence', y='profit_pct', color='successful',
                           title='Confidence vs Profit')
            fig.update_layout(xaxis_title='Confidence', yaxis_title='Profit %')
            st.plotly_chart(fig, use_container_width=True, key='confidence_vs_success_scatter')

def show_prediction_validation(dashboard):
    """Enhanced prediction validation analytics"""
    st.subheader("🎯 Prediction Validation System")
    
    try:
        # Load prediction data
        predictions_file = dashboard.data_dir / "ml_predictions.csv"
        validation_file = dashboard.data_dir / "prediction_validation.csv"
        
        if not predictions_file.exists():
            st.info("No prediction data available yet. Run the system to generate predictions.")
            return
            
        predictions_df = pd.read_csv(predictions_file)
        
        if validation_file.exists():
            validation_df = pd.read_csv(validation_file)
            # Merge predictions with validation results
            combined_df = pd.merge(predictions_df, validation_df, on='prediction_id', how='left')
        else:
            combined_df = predictions_df
            st.warning("Validation data not yet available. Predictions are being tracked.")
        
        # Prediction validation metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_predictions = len(predictions_df)
            validated_predictions = len(validation_df) if validation_file.exists() else 0
            st.metric("Total Predictions", total_predictions)
            st.metric("Validated", validated_predictions)
        
        with col2:
            if validation_file.exists() and not validation_df.empty:
                success_rate = len(validation_df[validation_df['success'] == True]) / len(validation_df) * 100
                neutral_rate = len(validation_df[validation_df['success'].isna()]) / len(validation_df) * 100
                st.metric("Success Rate", f"{success_rate:.1f}%")
                st.metric("Neutral Rate", f"{neutral_rate:.1f}%")
            else:
                st.metric("Success Rate", "Pending")
                st.metric("Neutral Rate", "Pending")
        
        with col3:
            if 'confidence' in predictions_df.columns:
                avg_confidence = predictions_df['confidence'].mean()
                high_conf_preds = len(predictions_df[predictions_df['confidence'] > 0.8])
                st.metric("Avg Confidence", f"{avg_confidence:.2f}")
                st.metric("High Confidence", high_conf_preds)
        
        with col4:
            if validation_file.exists() and 'actual_profit_pct' in validation_df.columns:
                total_profit = validation_df['actual_profit_pct'].sum()
                avg_profit = validation_df['actual_profit_pct'].mean()
                st.metric("Total Profit %", f"{total_profit:.2f}%")
                st.metric("Avg Profit %", f"{avg_profit:.3f}%")
            else:
                st.metric("Total Profit %", "Pending")
                st.metric("Avg Profit %", "Pending")
        
        # Prediction validation charts
        if validation_file.exists() and not validation_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Prediction Accuracy by Confidence")
                if 'confidence' in combined_df.columns and 'success' in combined_df.columns:
                    # Create confidence bins
                    combined_df['confidence_bin'] = pd.cut(combined_df['confidence'], 
                                                         bins=[0, 0.5, 0.7, 0.8, 0.9, 1.0], 
                                                         labels=['<50%', '50-70%', '70-80%', '80-90%', '90%+'])
                    
                    conf_analysis = combined_df.groupby('confidence_bin')['success'].agg(['count', 'sum']).reset_index()
                    conf_analysis['success_rate'] = (conf_analysis['sum'] / conf_analysis['count'] * 100).fillna(0)
                    
                    fig = px.bar(conf_analysis, x='confidence_bin', y='success_rate',
                               title='Success Rate by Confidence Level')
                    fig.update_layout(xaxis_title='Confidence Range', yaxis_title='Success Rate %')
                    st.plotly_chart(fig, use_container_width=True, key='success_rate_by_confidence')
            
            with col2:
                st.subheader("💰 Profit by Coin")
                if 'coin' in combined_df.columns and 'actual_profit_pct' in combined_df.columns:
                    coin_profit = combined_df.groupby('coin')['actual_profit_pct'].agg(['sum', 'count', 'mean']).reset_index()
                    coin_profit = coin_profit[coin_profit['count'] >= 2]  # Only coins with 2+ predictions
                    coin_profit = coin_profit.sort_values('sum', ascending=False).head(10)
                    
                    fig = px.bar(coin_profit, x='coin', y='sum',
                               title='Total Profit % by Coin (Top 10)')
                    fig.update_layout(xaxis_title='Coin', yaxis_title='Total Profit %')
                    st.plotly_chart(fig, use_container_width=True, key='profit_by_coin_bar')
        
        # Recent predictions table
        st.subheader("📋 Recent Predictions")
        if not combined_df.empty:
            # Show last 10 predictions
            recent_preds = combined_df.tail(10)[['timestamp', 'coin', 'direction', 'confidence', 
                                               'entry_price', 'target_price']].copy()
            
            if 'success' in combined_df.columns:
                recent_preds['status'] = combined_df.tail(10)['success'].apply(
                    lambda x: "✅ Success" if x == True else "❌ Failed" if x == False else "⏳ Pending"
                )
            else:
                recent_preds['status'] = "⏳ Pending"
            
            st.dataframe(recent_preds, use_container_width=True)
        
        # Prediction validation insights
        if validation_file.exists() and not validation_df.empty:
            st.subheader("🤖 AI Insights")
            
            # Calculate some insights
            insights = []
            
            if 'confidence' in combined_df.columns and 'success' in combined_df.columns:
                high_conf_success = combined_df[combined_df['confidence'] > 0.8]['success'].mean()
                low_conf_success = combined_df[combined_df['confidence'] <= 0.5]['success'].mean()
                
                if not pd.isna(high_conf_success) and not pd.isna(low_conf_success):
                    if high_conf_success > low_conf_success + 0.1:
                        insights.append(f"✅ High confidence predictions perform {(high_conf_success - low_conf_success)*100:.1f}% better")
                    elif low_conf_success > high_conf_success:
                        insights.append("⚠️ Low confidence predictions are outperforming high confidence ones")
            
            if 'coin' in combined_df.columns and 'success' in combined_df.columns:
                coin_performance = combined_df.groupby('coin')['success'].mean().sort_values(ascending=False)
                best_coin = coin_performance.index[0] if len(coin_performance) > 0 else None
                if best_coin and not pd.isna(coin_performance.iloc[0]):
                    insights.append(f"🏆 Best performing coin: {best_coin} ({coin_performance.iloc[0]*100:.1f}% success rate)")
            
            if insights:
                for insight in insights:
                    st.info(insight)
            else:
                st.info("🔄 Gathering more data to provide insights...")
    
    except Exception as e:
        st.error(f"Error loading prediction validation data: {e}")
        st.info("Make sure the enhanced prediction validator is running and has generated data.")

def show_ml_analytics(dashboard):
    """ML model performance analytics"""
    st.subheader("🤖 ML Model Analytics")
    
    try:
        # Load ML signals data
        ml_signals_file = dashboard.data_dir / "ml_signals.csv"
        
        if not ml_signals_file.exists():
            st.info("No ML signals data available.")
            return
        
        ml_signals_df = pd.read_csv(ml_signals_file)
        
        # ML model metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_signals = len(ml_signals_df)
            actionable_signals = len(ml_signals_df[ml_signals_df['recommendation'].isin(['BUY', 'SELL'])])
            st.metric("Total Signals", total_signals)
            st.metric("Actionable Signals", actionable_signals)
        
        with col2:
            if 'confidence' in ml_signals_df.columns:
                avg_ml_confidence = ml_signals_df['confidence'].mean()
                high_conf_signals = len(ml_signals_df[ml_signals_df['confidence'] > 0.8])
                st.metric("Avg ML Confidence", f"{avg_ml_confidence:.2f}")
                st.metric("High Conf Signals", high_conf_signals)
        
        with col3:
            signal_counts = ml_signals_df['recommendation'].value_counts()
            buy_signals = signal_counts.get('BUY', 0)
            sell_signals = signal_counts.get('SELL', 0)
            st.metric("BUY Signals", buy_signals)
            st.metric("SELL Signals", sell_signals)
        
        with col4:
            if 'model_used' in ml_signals_df.columns:
                models_used = ml_signals_df['model_used'].nunique()
                most_used_model = ml_signals_df['model_used'].mode().iloc[0] if len(ml_signals_df) > 0 else "N/A"
                st.metric("Models Used", models_used)
                st.metric("Primary Model", most_used_model)
        
        # ML analytics charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Signal Distribution")
            signal_dist = ml_signals_df['recommendation'].value_counts()
            fig = px.pie(values=signal_dist.values, names=signal_dist.index, 
                        title='ML Signal Distribution')
            st.plotly_chart(fig, use_container_width=True, key='ml_signal_distribution_pie')
        
        with col2:
            st.subheader("🕒 Signals Over Time")
            if 'timestamp' in ml_signals_df.columns:
                ml_signals_df['timestamp'] = pd.to_datetime(ml_signals_df['timestamp'])
                daily_signals = ml_signals_df.groupby(ml_signals_df['timestamp'].dt.date).size().reset_index()
                daily_signals.columns = ['date', 'signal_count']
                
                fig = px.line(daily_signals, x='date', y='signal_count',
                            title='Daily Signal Count')
                fig.update_layout(xaxis_title='Date', yaxis_title='Number of Signals')
                st.plotly_chart(fig, use_container_width=True, key='daily_signal_count_line')
        
        # Recent ML signals
        st.subheader("📋 Recent ML Signals")
        recent_signals = ml_signals_df.tail(10)[['timestamp', 'coin', 'recommendation', 
                                               'confidence', 'price', 'model_used']].copy()
        st.dataframe(recent_signals, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading ML analytics data: {e}")

# Bot control functions
def start_bot():
    """Start the trading bot in a separate process"""
    try:
        import subprocess
        import sys
        import os
        
        # Get absolute paths relative to this script
        dashboard_dir = Path(__file__).parent.absolute()
        project_root = dashboard_dir.parent
        
        # Path to the bot runner script
        bot_script = project_root / "run_bot.py"
        if not bot_script.exists():
            # Fallback to main PSC system
            bot_script = project_root / "psc_ton_system.py"
            if not bot_script.exists():
                st.error(f"❌ Trading bot script not found at {bot_script}")
                st.error(f"📁 Project root: {project_root}")
                return
        
        # Start bot in separate process
        if 'bot_process' not in st.session_state or st.session_state.bot_process is None:
            # Use the virtual environment Python
            python_exe = project_root / ".venv" / "Scripts" / "python.exe"
            if not python_exe.exists():
                # Try system python
                python_exe = "python"
                st.info(f"🔧 Using system Python: {python_exe}")
            else:
                st.info(f"🔧 Using venv Python: {python_exe}")
            
            try:
                # Ensure working directory is correct
                working_dir = str(project_root)
                
                # Build command - Try different approaches
                test_script = project_root / "test_bot_start.py"
                
                # Instead of using test script, try direct bot script
                cmd = [str(python_exe), str(bot_script)]
                
                st.info(f"🚀 Starting bot with command: {' '.join(cmd)}")
                st.info(f"📁 Working directory: {working_dir}")
                st.info(f"📄 Bot script: {cmd[-1]}")
                
                # Set environment variables to handle encoding
                env = os.environ.copy()
                env['PYTHONIOENCODING'] = 'utf-8'
                env['PYTHONLEGACYWINDOWSSTDIO'] = '1'
                
                # Start the bot as a separate process
                process = subprocess.Popen(
                    cmd,
                    cwd=working_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    env=env  # Use updated environment with encoding settings
                )
                
                # Wait a moment to see if process starts successfully
                import time
                time.sleep(2)  # Give more time
                
                if process.poll() is None:
                    # Process is running
                    st.session_state.bot_process = process
                    st.session_state.bot_running = True
                    
                    st.success("✅ Bot started successfully in separate process!")
                    st.info(f"🔧 Process ID: {process.pid}")
                    st.info(f"📄 Script: {Path(cmd[-1]).name}")
                    
                    # Try to capture some initial output
                    try:
                        # Non-blocking read to get startup messages
                        import select
                        import time
                        
                        # Wait a bit more for output
                        time.sleep(1)
                        
                        # Read any available output
                        if process.stdout:
                            stdout_data = process.stdout.readline()
                            if stdout_data:
                                st.info(f"📝 Bot output: {stdout_data.strip()}")
                        
                    except Exception as output_e:
                        st.warning(f"⚠️ Could not capture output: {output_e}")
                    
                    # Initialize ML microstructure if available in dashboard
                    if ML_MICROSTRUCTURE_AVAILABLE and 'ml_microstructure' not in st.session_state:
                        try:
                            st.session_state.ml_microstructure = LiveMicrostructureTrainer()
                            st.info("✅ ML Microstructure system initialized in dashboard")
                        except Exception as e:
                            st.warning(f"⚠️ ML Microstructure initialization failed: {e}")
                    
                    st.rerun()
                else:
                    # Process failed to start
                    exit_code = process.returncode
                    stdout, stderr = process.communicate()
                    st.error(f"❌ Bot process failed to start (exit code: {exit_code})")
                    if stderr:
                        st.error(f"Error output: {stderr}")
                    if stdout:
                        st.info(f"Standard output: {stdout}")
                    
                    st.info("💡 Try clicking the 'Show Process Debug Info' button for more details")
                
            except Exception as e:
                st.error(f"❌ Failed to start bot process: {e}")
                st.error(f"Exception type: {type(e).__name__}")
        else:
            # Check if existing process is still running
            if st.session_state.bot_process.poll() is None:
                st.warning("⚠️ Bot process already running")
            else:
                # Process ended, clean up and try again
                st.session_state.bot_process = None
                st.session_state.bot_running = False
                start_bot()  # Recursive call to restart
        
    except Exception as e:
        st.error(f"❌ Failed to start bot: {e}")
        st.error(f"Exception type: {type(e).__name__}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")

def stop_bot():
    """Stop the trading bot process"""
    try:
        import subprocess
        
        if 'bot_process' in st.session_state and st.session_state.bot_process is not None:
            process = st.session_state.bot_process
            
            if process.poll() is None:  # Process is still running
                process.terminate()
                
                # Wait a bit for graceful shutdown
                try:
                    process.wait(timeout=5)
                    st.success("✅ Bot stopped gracefully!")
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't stop gracefully
                    process.kill()
                    st.warning("⚠️ Bot force-stopped (was unresponsive)")
                
                st.session_state.bot_process = None
                st.session_state.bot_running = False
                st.rerun()
            else:
                st.info("ℹ️ Bot process was already stopped")
                st.session_state.bot_process = None
                st.session_state.bot_running = False
        else:
            st.warning("⚠️ No bot process to stop")
            st.session_state.bot_running = False
        
    except Exception as e:
        st.error(f"❌ Failed to stop bot: {e}")

def check_bot_status():
    """Check if bot process is actually running"""
    if 'bot_process' in st.session_state and st.session_state.bot_process is not None:
        process = st.session_state.bot_process
        if process.poll() is None:
            return True  # Process is running
        else:
            # Process ended, clean up
            st.session_state.bot_process = None
            st.session_state.bot_running = False
            return False
    return False

def restart_bot():
    """Restart the trading bot"""
    try:
        stop_bot()
        time.sleep(1)
        start_bot()
        st.success("✅ Bot restarted successfully!")
        
    except Exception as e:
        st.error(f"❌ Failed to restart bot: {e}")

def retrain_ml_models():
    """Retrain ML models"""
    try:
        if st.session_state.trading_bot and st.session_state.trading_bot.ml_engine:
            success = st.session_state.trading_bot.ml_engine.retrain_models()
            if success:
                st.success("✅ ML models retrained successfully!")
            else:
                st.warning("⚠️ Not enough data for retraining")
        else:
            st.error("❌ ML engine not available")
            
    except Exception as e:
        st.error(f"❌ Retraining failed: {e}")

def export_trading_data(dashboard):
    """Export trading data"""
    try:
        data = dashboard.load_trading_data()
        
        if not data['trades'].empty:
            csv = data['trades'].to_csv(index=False)
            st.download_button(
                label="📥 Download Trading Data",
                data=csv,
                file_name=f"trading_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No trading data to export")
            
    except Exception as e:
        st.error(f"❌ Export failed: {e}")

if __name__ == "__main__":
    main()
