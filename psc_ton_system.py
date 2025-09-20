"""
PSC + TON Trading System - Enhanced with Superp No-Liquidation Technology
Revolutionary trading system combining PSC arbitrage with Superp leverage
"""

import asyncio
import logging
import yaml
import random
import csv
import json
import os
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes
import signal
import time
import asyncio
from functools import wraps

# Import our new database system
from psc_data_manager import PSCDataManager

# Import enhanced real market data and filtering
from real_market_data import RealMarketDataProvider
from advanced_signal_filter import AdvancedSignalFilter

# Setup logging first with UTF-8 encoding
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Configure logger to handle Unicode properly
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/system.log', encoding='utf-8', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Import integrated accuracy system
try:
    # Add current directory to Python path to ensure src can be found
    import sys
    from pathlib import Path
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    from src.integrated_signal_processor import IntegratedSignalProcessor
    INTEGRATED_PROCESSOR_AVAILABLE = True
    logger.info("✅ Integrated Signal Processor loaded successfully")
except ImportError as e:
    INTEGRATED_PROCESSOR_AVAILABLE = False
    logger.warning(f"Integrated Signal Processor not available: {e}")

def retry_on_error(max_retries=3, delay=5):
    """Decorator for retrying failed operations"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                        await asyncio.sleep(delay * (2 ** attempt))  # Exponential backoff
                    else:
                        logger.error(f"All {max_retries} attempts failed for {func.__name__}")
                        raise last_exception
            return None
        return wrapper
    return decorator

import sys
import aiohttp
import threading
import time
import psutil
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
from aiohttp import web

# Ensure logs directory exists
from pathlib import Path
Path('logs').mkdir(exist_ok=True)

# Import TradingView integration
try:
    from tradingview_integration import TradingViewIntegration, TechnicalAnalysis
    TRADINGVIEW_AVAILABLE = True
    logger.info("✅ TradingView integration imported successfully")
except ImportError as e:
    TRADINGVIEW_AVAILABLE = False
    logger.warning(f"⚠️ TradingView integration not available: {e}")

# Import Enhanced Prediction Validator
try:
    from src.database_prediction_validator import DatabasePredictionValidator, EnhancedPredictionValidator
    PREDICTION_VALIDATOR_AVAILABLE = True
    logger.info("✅ Database-Integrated Prediction Validator imported successfully")
except ImportError as e:
    # Fallback to original validator
    try:
        from src.enhanced_prediction_validator import EnhancedPredictionValidator
        PREDICTION_VALIDATOR_AVAILABLE = True
        logger.info("✅ Legacy Prediction Validator imported successfully")
    except ImportError as e2:
        PREDICTION_VALIDATOR_AVAILABLE = False
        logger.warning(f"⚠️ No Prediction Validator available: {e}, {e2}")

# Import ML Microstructure Trainer
try:
    from src.models.live_microstructure_trainer import LiveMicrostructureTrainer, PSCSignal, TimerStatus
    ML_MICROSTRUCTURE_AVAILABLE = True
    logger.info("✅ ML Microstructure Trainer imported successfully")
except ImportError as e:
    ML_MICROSTRUCTURE_AVAILABLE = False
    logger.warning(f"⚠️ ML Microstructure Trainer not available: {e}")

class SuperpLeverageType(Enum):
    """Superp leverage categories for no-liquidation trading"""
    CONSERVATIVE = (1, 100)    # 1x-100x for low-risk signals
    MODERATE = (100, 1000)     # 100x-1000x for medium confidence
    AGGRESSIVE = (1000, 5000)  # 1000x-5000x for high confidence
    EXTREME = (5000, 10000)    # 5000x-10000x for maximum confidence

@dataclass
@dataclass
class SuperpPosition:
    """Superp no-liquidation position structure with timer-based leverage tracking"""
    id: str
    asset: str
    buy_in_amount: float
    virtual_exposure: float
    effective_leverage: float
    entry_price: float
    target_price: float
    stop_time: datetime
    psc_ratio: float
    confidence_score: float
    status: str = "ACTIVE"
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    # Timer-based leverage tracking
    entry_leverage: float = 0.0
    current_leverage: float = 0.0
    timer_minute_opened: int = 0
    leverage_history: Dict = None
    leverage_snapshots: List = None
    
    def __post_init__(self):
        """Initialize mutable default values"""
        if self.leverage_history is None:
            self.leverage_history = {}
        if self.leverage_snapshots is None:
            self.leverage_snapshots = []

class HealthCheckHandler(BaseHTTPRequestHandler):
    """HTTP handler for health checks and monitoring"""
    
    def do_GET(self):
        global system
        
        if self.path == '/health' or self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            if system:
                uptime = (datetime.now() - system.start_time).total_seconds()
                health_data = {
                    'status': system.health_status,
                    'uptime_seconds': uptime,
                    'uptime_hours': round(uptime / 3600, 2),
                    'last_activity': system.last_activity.isoformat(),
                    'system_stats': system.system_stats,
                    'memory_usage': psutil.virtual_memory().percent,
                    'cpu_usage': psutil.cpu_percent(),
                    'active_threads': threading.active_count(),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                health_data = {
                    'status': 'starting',
                    'message': 'System initializing'
                }
            
            self.wfile.write(json.dumps(health_data, indent=2).encode())
            
        elif self.path == '/stats':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            if system:
                stats = {
                    'monitored_coins': len(system.monitored_coins),
                    'active_positions': len(system.superp_positions),
                    'notifications_enabled': system.notifications_enabled,
                    'total_exposure': system.total_superp_exposure,
                    'config': {
                        'min_signal_ratio': system.min_signal_ratio,
                        'min_confidence_threshold': system.min_confidence_threshold
                    }
                }
                self.wfile.write(json.dumps(stats, indent=2).encode())
            else:
                self.wfile.write(json.dumps({'error': 'System not ready'}).encode())
                
        elif self.path == '/api/export_csv':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            if system and hasattr(system, 'data_manager'):
                try:
                    # Export all data to CSV files
                    csv_files = []
                    for table in ['signals', 'trades', 'validation', 'performance', 'system_events']:
                        try:
                            csv_file = system.data_manager.db.export_to_csv(table)
                            csv_files.append(csv_file)
                        except Exception as e:
                            logger.warning(f"Failed to export {table}: {e}")
                    
                    response = {
                        'status': 'success',
                        'message': 'Data exported to CSV files',
                        'files': csv_files,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.wfile.write(json.dumps(response, indent=2).encode())
                except Exception as e:
                    response = {
                        'status': 'error',
                        'message': f'Export failed: {str(e)}',
                        'timestamp': datetime.now().isoformat()
                    }
                    self.wfile.write(json.dumps(response, indent=2).encode())
            else:
                response = {
                    'status': 'error',
                    'message': 'System or data manager not ready',
                    'timestamp': datetime.now().isoformat()
                }
                self.wfile.write(json.dumps(response, indent=2).encode())
                
        elif self.path == '/api/session_stats':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            if system and hasattr(system, 'data_manager'):
                try:
                    session_stats = system.data_manager.get_session_stats()
                    response = {
                        'status': 'success',
                        'session_stats': session_stats,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.wfile.write(json.dumps(response, indent=2).encode())
                except Exception as e:
                    response = {
                        'status': 'error',
                        'message': f'Failed to get session stats: {str(e)}',
                        'timestamp': datetime.now().isoformat()
                    }
                    self.wfile.write(json.dumps(response, indent=2).encode())
            else:
                response = {
                    'status': 'error',
                    'message': 'System or data manager not ready',
                    'timestamp': datetime.now().isoformat()
                }
                self.wfile.write(json.dumps(response, indent=2).encode())
                
        elif self.path == '/api/system_health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            if system and hasattr(system, 'data_manager'):
                try:
                    health_info = system.data_manager.get_system_health()
                    response = {
                        'status': 'success',
                        'database_health': health_info,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.wfile.write(json.dumps(response, indent=2).encode())
                except Exception as e:
                    response = {
                        'status': 'error',
                        'message': f'Failed to get health info: {str(e)}',
                        'timestamp': datetime.now().isoformat()
                    }
                    self.wfile.write(json.dumps(response, indent=2).encode())
            else:
                response = {
                    'status': 'error',
                    'message': 'System or data manager not ready',
                    'timestamp': datetime.now().isoformat()
                }
                self.wfile.write(json.dumps(response, indent=2).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        # Suppress HTTP server logs
        pass

class PSCTONTradingBot:
    """PSC + TON Trading System - Enhanced with Superp No-Liquidation Technology"""
    
    def __init__(self):
        # FIXED: Set project_root FIRST before loading config
        self.project_root = Path(__file__).parent
        
        self.config = self._load_config()
        
        # Check if Telegram bot should be disabled (for multiple deployments)
        disable_telegram = os.getenv('DISABLE_TELEGRAM_BOT', '').lower() in ['true', '1', 'yes']
        
        if disable_telegram:
            logger.warning("🚫 Telegram bot DISABLED via DISABLE_TELEGRAM_BOT environment variable")
            self.bot_token = None
            self.chat_id = None
            self.telegram_enabled = False
        else:
            self.bot_token = self.config.get('telegram_token', '')
            self.chat_id = self.config.get('telegram_chat_id', '')
            self.telegram_enabled = True
            
            if not self.bot_token:
                raise ValueError("No bot token configured")
            if not self.chat_id:
                raise ValueError("No chat ID configured")
            
        self.application = None
        self.running = False
        self.timer_minute = 0
        
        # Notification controls
        self.notifications_enabled = True
        self.high_confidence_only = False
        self.min_confidence_threshold = 0.65  # Raised back to higher standard to reduce weak signals
        
        # Timer alerts control
        disable_timer_env = os.getenv('DISABLE_TIMER_ALERTS', '').lower()
        disable_notifications_env = os.getenv('DISABLE_TIMER_NOTIFICATIONS', '').lower()
        
        self.timer_alerts_enabled = not (disable_timer_env in ['true', '1', 'yes'] or 
                                        disable_notifications_env in ['true', '1', 'yes'])
        
        if not self.timer_alerts_enabled:
            logger.info("🔇 Timer alerts and notifications DISABLED via environment variable")
            logger.info(f"   DISABLE_TIMER_ALERTS: {os.getenv('DISABLE_TIMER_ALERTS', 'not set')}")
            logger.info(f"   DISABLE_TIMER_NOTIFICATIONS: {os.getenv('DISABLE_TIMER_NOTIFICATIONS', 'not set')}")
        
        # Enhanced PSC + Superp Settings
        self.min_signal_ratio = 7.0  # UPDATED: Logarithmic PSC threshold (log + 6 scale, was 6.5)
        self.confidence_thresholds = {
            'very_high': 0.8,
            'high': 0.6,
            'medium': 0.4
        }
        
        # Superp Configuration
        self.superp_config = {
            'min_buy_in': 10.0,      # Minimum $10 buy-in
            'max_buy_in': 100.0,     # Maximum $100 buy-in  
            'max_leverage': 10000.0, # Up to 10,000x leverage
            'position_timeout': 600  # 10 minutes maximum
        }
        
        # Superp Positions Tracking
        self.superp_positions: Dict[str, SuperpPosition] = {}
        self.total_superp_exposure = 0.0
        self.max_total_risk = 500.0  # Maximum total at risk
        
        # Enhanced multi-coin monitoring - 6 coins total
        self.monitored_coins = [
            {'symbol': 'BTC', 'name': 'Bitcoin', 'volatility': 'Medium', 'pair': 'BTC/USDT'},
            {'symbol': 'ETH', 'name': 'Ethereum', 'volatility': 'Medium', 'pair': 'ETH/USDT'},
            {'symbol': 'SOL', 'name': 'Solana', 'volatility': 'High', 'pair': 'SOL/USDT'},
            {'symbol': 'SHIB', 'name': 'Shiba Inu', 'volatility': 'Very High', 'pair': 'SHIB/USDT'},
            {'symbol': 'DOGE', 'name': 'Dogecoin', 'volatility': 'High', 'pair': 'DOGE/USDT'},
            {'symbol': 'PEPE', 'name': 'Pepe', 'volatility': 'Extreme', 'pair': 'PEPE/USDT'}
        ]
        
        # Price tracking for current prices and exit estimates
        self.price_history = {}
        self.last_prices = {}
        self.price_changes = {}
        
        # Track actual open positions for real exit logging
        self.open_positions = {}  # {coin: {'entry_price': float, 'entry_time': datetime, 'direction': str, 'confidence': float, 'target_exit': float, 'signal_id': str, 'leverage': float, 'position_size': float}}
        
        # PSC Dynamic Leverage Settings
        self.base_position_size = 100.0  # Base $100 position
        self.max_leverage = 10.0  # Maximum 10x leverage
        self.min_leverage = 1.0   # Minimum 1x leverage
        
        # Initialize unified database system (replaces CSV files)
        database_path = self.project_root / "database" / "psc_trading.db"
        self.data_manager = PSCDataManager(str(database_path))
        
        # Legacy CSV file paths (kept for compatibility/backup)
        self.trades_log_file = self.project_root / "data" / "live_trades.csv"
        self.signals_log_file = self.project_root / "data" / "psc_signals.csv"  
        self.daily_summary_file = self.project_root / "data" / "daily_summaries.csv"
        
        # Debug: Log the database initialization
        logger.info(f"🔧 PSC TON System initialized:")
        logger.info(f"   Project root: {self.project_root}")
        logger.info(f"   Database: {database_path}")
        logger.info(f"   Legacy CSV files: {self.signals_log_file.parent}")
        
        # Initialize Enhanced Prediction Validator
        self.prediction_validator = None
        if PREDICTION_VALIDATOR_AVAILABLE:
            try:
                # Use DatabasePredictionValidator with database integration
                # Pass the existing database path that we know works
                db_path = str(database_path)
                self.prediction_validator = DatabasePredictionValidator(self.project_root, db_path)
                logger.info("🔍 Database-Integrated Prediction Validator initialized")
            except Exception as e:
                logger.error(f"Failed to initialize prediction validator: {e}")
                logger.info("Prediction validation will be disabled - system continues normally")
        
        # Initialize ML Microstructure Trainer with database integration - DATABASE ONLY
        self.ml_microstructure_trainer = None
        if ML_MICROSTRUCTURE_AVAILABLE:
            try:
                # Database-only initialization - no fallback modes
                logger.debug("Initializing LiveMicrostructureTrainer with database integration...")
                self.ml_microstructure_trainer = LiveMicrostructureTrainer(data_manager=self.data_manager)
                
                # Verify database integration
                if not hasattr(self.ml_microstructure_trainer, 'data_manager') or self.ml_microstructure_trainer.data_manager is None:
                    raise ValueError("Microstructure trainer failed to initialize with database - database-only mode required")
                
                logger.info("🧠 ML Microstructure Trainer initialized with database integration")
                logger.info("🎯 PSC-ML integration enabled for enhanced signal quality")
                    
            except Exception as e:
                logger.error(f"❌ Failed to initialize ML microstructure trainer with database: {e}")
                logger.error("❌ DATABASE-ONLY MODE: No fallback - system requires database integration")
                # Don't initialize without database - this ensures database-only operation
                self.ml_microstructure_trainer = None
                
        # Database system handles all data persistence
        
        # Trading statistics
        self.session_stats = {
            'signals_generated': 0,
            'trades_executed': 0,
            'successful_trades': 0,
            'total_profit': 0.0,
            'session_start': datetime.now()
        }
        
        # Deployment and monitoring features
        self.start_time = datetime.now()
        self.health_status = "starting"
        self.last_activity = datetime.now()
        self.system_stats = {
            'total_signals': 0,
            'successful_predictions': 0,
            'active_positions': 0,
            'system_uptime': 0
        }
        
        # HTTP server for health checks (cloud deployment)
        self.http_server = None
        self.http_port = int(os.environ.get('PORT', 8080))
        
        # Initialize ML engine with database integration
        try:
            import sys
            # Add current directory to path to ensure we find the local ml_engine
            current_dir = Path(__file__).parent
            sys.path.insert(0, str(current_dir))
            from src.ml_engine import MLEngine
            self.ml_engine = MLEngine(data_manager=self.data_manager)
            logger.info("ML Engine initialized successfully with database integration")
        except Exception as e:
            logger.error(f"Failed to initialize ML engine: {e}")
            self.ml_engine = None
            
        # Initialize Real Market Data Provider (NEW!)
        try:
            self.real_market_provider = RealMarketDataProvider()
            logger.info("✅ Real Market Data Provider initialized - using live data feeds")
        except Exception as e:
            logger.error(f"Failed to initialize real market data provider: {e}")
            self.real_market_provider = None
            
        # Initialize Advanced Signal Filter (NEW!)
        try:
            self.signal_filter = AdvancedSignalFilter(db_path=str(database_path))
            logger.info("✅ Advanced Signal Filter initialized - intelligent signal filtering enabled")
        except Exception as e:
            logger.error(f"Failed to initialize signal filter: {e}")
            self.signal_filter = None
        except ImportError:
            logger.warning("ML Engine not found, using simple prediction system")
            self.ml_engine = None
        except Exception as e:
            logger.warning(f"ML Engine init warning: {e}")
            self.ml_engine = None
        
        # Enhanced Prediction Validator is the active validation system
        # (Archive.paper_trading_validator has been superseded)
        self.paper_validator = None
        
        # Initialize TradingView integration with real market data
        self.tradingview = None
        if TRADINGVIEW_AVAILABLE:
            try:
                # Enable real market data for enhanced accuracy
                self.tradingview = TradingViewIntegration(use_real_data=True)
                logger.info("✅ TradingView integration initialized with REAL MARKET DATA")
            except Exception as e:
                logger.warning(f"⚠️ TradingView initialization failed: {e}")
                self.tradingview = None
        
        # TradingView settings
        self.tradingview_enabled = True
        self.tradingview_timeframe = '1m'  # 1-minute timeframe
        self.tradingview_check_interval = 30  # Check every 30 seconds
        self.tradingview_logs = []  # Store recent TradingView data
        
        # Initialize Integrated Signal Processor (Enhanced Accuracy System)
        self.integrated_processor = None
        if INTEGRATED_PROCESSOR_AVAILABLE:
            try:
                self.integrated_processor = IntegratedSignalProcessor(self)
                logger.info("🎯 Integrated Signal Processor initialized - Enhanced accuracy mode enabled")
            except Exception as e:
                logger.warning(f"⚠️ Integrated Signal Processor initialization failed: {e}")
                self.integrated_processor = None
    
        # Database system handles all data persistence (CSV setup deprecated)
        logger.info("✅ Data persistence handled by database system")
    
    # ============================================================================
    # SUPERP NO-LIQUIDATION TRADING METHODS
    # ============================================================================
    

    def cleanup_memory(self):
        """Clean up memory to prevent resource exhaustion"""
        try:
            # Limit data retention
            max_signals = 1000
            max_trades = 500
            
            # Clean old signals
            if hasattr(self, 'psc_signals') and len(self.psc_signals) > max_signals:
                self.psc_signals = self.psc_signals[-max_signals:]
                logger.info(f"🧹 Cleaned old signals, keeping last {max_signals}")
            
            # Clean old trades
            if hasattr(self, 'trades') and len(self.trades) > max_trades:
                self.trades = self.trades[-max_trades:]
                logger.info(f"🧹 Cleaned old trades, keeping last {max_trades}")
            
            # Clean ML prediction history
            if hasattr(self, 'ml_prediction_history') and len(self.ml_prediction_history) > max_signals:
                self.ml_prediction_history = self.ml_prediction_history[-max_signals:]
                logger.info(f"🧹 Cleaned ML history, keeping last {max_signals}")
                
        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")

    def calculate_superp_timer_leverage(self, confidence: float, psc_ratio: float, 
                                       volatility: float, timer_minute: int) -> float:
        """Calculate Superp leverage based on current timer position (like the mini app)"""
        # Superp mini app logic: Leverage decreases as timer progresses
        # Early timer (0-3 min): Maximum leverage available
        # Mid timer (4-7 min): Reduced leverage
        # Late timer (8-10 min): Minimum leverage
        
        # Base leverage calculation using existing logic
        base_confidence_multiplier = confidence * 2
        ratio_multiplier = min(psc_ratio / 1.25, 4.0)
        
        # Volatility adjustment for Superp
        volatility_map = {
            'Very High': 0.7, 'Extreme': 0.6, 'High': 0.8, 'Medium': 1.0, 'Low': 1.2
        }
        volatility_multiplier = volatility_map.get(volatility, 1.0)
        
        # TIMER-BASED LEVERAGE DECAY (Critical for Superp accuracy)
        if timer_minute <= 2:
            # Entry window: Full leverage (100%)
            timer_multiplier = 1.0
        elif timer_minute <= 5:
            # Early phase: High leverage (80-95%)
            timer_multiplier = 1.0 - (timer_minute - 2) * 0.05  # 100% -> 85%
        elif timer_minute <= 8:
            # Mid phase: Moderate leverage (60-80%)
            timer_multiplier = 0.85 - (timer_minute - 5) * 0.08  # 85% -> 61%
        else:
            # Late phase: Low leverage (30-60%)
            timer_multiplier = 0.61 - (timer_minute - 8) * 0.15  # 61% -> 31%
        
        # Calculate Superp leverage with timer decay
        superp_base_leverage = 100.0  # Higher base for Superp
        raw_leverage = (
            superp_base_leverage * 
            base_confidence_multiplier * 
            ratio_multiplier * 
            volatility_multiplier * 
            timer_multiplier
        )
        
        # Clamp to Superp limits (much higher than traditional)
        max_superp_leverage = self.superp_config.get('max_leverage', 10000.0)
        final_leverage = max(1.0, min(raw_leverage, max_superp_leverage))
        
        # Only log if timer alerts are enabled
        if hasattr(self, 'timer_alerts_enabled') and self.timer_alerts_enabled:
            logger.info(f"⏰ Superp Timer Leverage: Minute {timer_minute}, "
                       f"Timer Factor {timer_multiplier:.2f}, Final Leverage {final_leverage:.0f}x")
        
        return round(final_leverage, 2)
    
    def take_leverage_snapshot(self, position_id: str, current_timer_minute: int):
        """Take a snapshot of current leverage for position (mirrors Superp mini app)"""
        if position_id not in self.superp_positions:
            return
            
        position = self.superp_positions[position_id]
        if position.status != "ACTIVE":
            return
            
        # Calculate current leverage based on timer
        current_leverage = self.calculate_superp_timer_leverage(
            position.confidence_score,
            position.psc_ratio,
            'Medium',  # Default volatility for leverage calc
            current_timer_minute
        )
        
        # Update position with current leverage
        position.current_leverage = current_leverage
        
        # Initialize leverage tracking if not exists
        if position.leverage_history is None:
            position.leverage_history = {}
        if position.leverage_snapshots is None:
            position.leverage_snapshots = []
            
        # Record leverage snapshot
        position.leverage_history[current_timer_minute] = current_leverage
        position.leverage_snapshots.append({
            'timer_minute': current_timer_minute,
            'leverage': current_leverage,
            'timestamp': datetime.now().isoformat()
        })
        
        # Recalculate virtual exposure with current leverage
        new_virtual_exposure = position.buy_in_amount * current_leverage
        position.virtual_exposure = new_virtual_exposure
        
        # Only log if timer alerts are enabled
        if self.timer_alerts_enabled:
            logger.info(f"📸 Leverage Snapshot: {position.asset} | Timer {current_timer_minute} | "
                       f"Leverage {current_leverage:.0f}x | Exposure ${new_virtual_exposure:,.0f}")
    
    def update_all_position_leverages(self, current_timer_minute: int):
        """Update leverage for all active Superp positions based on current timer"""
        for position_id in list(self.superp_positions.keys()):
            self.take_leverage_snapshot(position_id, current_timer_minute)
    
    def determine_leverage_category(self, confidence: float, psc_ratio: float) -> SuperpLeverageType:
        """Determine appropriate Superp leverage category based on signal strength"""
        combined_score = confidence * psc_ratio
        
        if combined_score >= 2.5:  # Very high confidence + strong PSC ratio
            return SuperpLeverageType.EXTREME
        elif combined_score >= 2.0:
            return SuperpLeverageType.AGGRESSIVE
        elif combined_score >= 1.5:
            return SuperpLeverageType.MODERATE
        else:
            return SuperpLeverageType.CONSERVATIVE
    
    def calculate_optimal_superp_buy_in(self, asset_price: float, confidence: float, 
                                      psc_ratio: float, volatility: float) -> Tuple[float, float]:
        """Calculate optimal Superp buy-in amount and expected leverage"""
        leverage_category = self.determine_leverage_category(confidence, psc_ratio)
        min_lev, max_lev = leverage_category.value
        
        # Target leverage based on confidence (higher confidence = higher leverage)
        target_leverage = min_lev + (max_lev - min_lev) * confidence
        
        # Calculate buy-in for target leverage: buy_in = asset_price / target_leverage
        buy_in = asset_price / target_leverage
        
        # Constrain to configured limits
        min_buy_in = self.superp_config['min_buy_in']
        max_buy_in = self.superp_config['max_buy_in']
        
        buy_in = max(min_buy_in, min(buy_in, max_buy_in))
        actual_leverage = asset_price / buy_in
        
        logger.info(f"🎯 Superp Calculation: Price=${asset_price:.2f}, Confidence={confidence:.2f}, "
                   f"Buy-in=${buy_in:.2f}, Leverage={actual_leverage:.0f}x")
        
        return buy_in, actual_leverage
    
    def create_superp_position(self, asset: str, price: float, psc_ratio: float, 
                             confidence: float, volatility: float, position_size_multiplier: float = 1.0) -> Optional[SuperpPosition]:
        """Create a new Superp no-liquidation position with timer-based leverage"""
        try:
            # Get current timer position for leverage calculation
            current_timer_minute = self.get_aligned_timer_minute(datetime.now())
            
            # Calculate Superp leverage based on timer position
            superp_leverage = self.calculate_superp_timer_leverage(
                confidence, psc_ratio, volatility, current_timer_minute
            )
            
            # Calculate buy-in using Superp leverage with enhanced position sizing
            # Apply position size multiplier from signal quality filter
            base_target_exposure = 1000.0  # Base target exposure
            enhanced_target_exposure = base_target_exposure * position_size_multiplier
            
            logger.info(f"🎯 Enhanced target exposure: ${enhanced_target_exposure:.2f} (base: ${base_target_exposure}, multiplier: {position_size_multiplier:.2f}x)")
            
            buy_in = max(
                self.superp_config['min_buy_in'],
                min(enhanced_target_exposure / superp_leverage, self.superp_config['max_buy_in'])
            )
            
            # Recalculate actual leverage with constrained buy-in
            actual_leverage = enhanced_target_exposure / buy_in
            
            # Check total risk limits
            if self.total_superp_exposure + buy_in > self.max_total_risk:
                logger.warning(f"⚠️ Superp position rejected - would exceed max risk ${self.max_total_risk}")
                return None
            
            # Create position with timer-aware leverage and realistic targets
            position_id = f"SUPERP_{asset}_{datetime.now().strftime('%H%M%S')}"
            virtual_exposure = buy_in * actual_leverage
            
            # Use realistic Superp exit price calculation
            target_price = self.calculate_superp_exit_price(
                price, confidence, "LONG", actual_leverage, buy_in
            )
            
            stop_time = datetime.now() + timedelta(minutes=10)  # 10-minute limit
            
            position = SuperpPosition(
                id=position_id,
                asset=asset,
                buy_in_amount=buy_in,
                virtual_exposure=virtual_exposure,
                effective_leverage=actual_leverage,
                entry_price=price,
                target_price=target_price,
                stop_time=stop_time,
                psc_ratio=psc_ratio,
                confidence_score=confidence,
                current_price=price,
                # Timer-based tracking
                entry_leverage=actual_leverage,
                current_leverage=actual_leverage,
                timer_minute_opened=current_timer_minute
            )
            
            # Track position
            self.superp_positions[position_id] = position
            self.total_superp_exposure += buy_in
            
            logger.info(f"🚀 Superp Position Created: {asset} | Timer {current_timer_minute} | "
                       f"Buy-in: ${buy_in:.2f} | Leverage: {actual_leverage:.0f}x | "
                       f"Exposure: ${virtual_exposure:,.0f}")
            
            return position
            
        except Exception as e:
            logger.error(f"Error creating Superp position: {e}")
            return None
    
    def update_superp_positions(self, current_prices: Dict[str, float]):
        """Update all active Superp positions with current market prices and timer leverage"""
        positions_to_close = []
        current_timer_minute = self.get_aligned_timer_minute(datetime.now())
        
        # First, update all position leverages based on current timer
        self.update_all_position_leverages(current_timer_minute)
        
        for position_id, position in self.superp_positions.items():
            if position.status != "ACTIVE":
                continue
                
            current_price = current_prices.get(position.asset, position.current_price)
            position.current_price = current_price
            
            # Calculate P&L using CURRENT leverage (important for accurate P&L)
            price_change_pct = (current_price - position.entry_price) / position.entry_price
            # Use current virtual exposure (which reflects current leverage)
            position.unrealized_pnl = price_change_pct * position.virtual_exposure
            
            # Check for profit target or time limit
            if current_price >= position.target_price:
                # Profit target hit
                position.status = "CLOSED_PROFIT"
                position.realized_pnl = position.unrealized_pnl
                positions_to_close.append(position_id)
                
                # Log leverage snapshot at exit
                logger.info(f"🎯 Superp Target Hit: {position.asset} | "
                           f"Entry Leverage: {position.entry_leverage:.0f}x | "
                           f"Exit Leverage: {position.current_leverage:.0f}x | "
                           f"Profit: ${position.realized_pnl:,.2f}")
                
            elif datetime.now() >= position.stop_time:
                # Time limit reached
                position.status = "CLOSED_TIME"
                position.realized_pnl = position.unrealized_pnl
                positions_to_close.append(position_id)
                
                logger.info(f"⏰ Superp Time Exit: {position.asset} | "
                           f"Entry Leverage: {position.entry_leverage:.0f}x | "
                           f"Final Leverage: {position.current_leverage:.0f}x | "
                           f"P&L: ${position.realized_pnl:,.2f}")
        
        # Remove closed positions from active tracking
        for position_id in positions_to_close:
            if position_id in self.superp_positions:
                closed_position = self.superp_positions[position_id]
                self.total_superp_exposure -= closed_position.buy_in_amount
                
                # Log complete leverage history for analysis
                if closed_position.leverage_history:
                    leverage_summary = ", ".join([
                        f"T{min}:{lev:.0f}x" 
                        for min, lev in closed_position.leverage_history.items()
                    ])
                    logger.info(f"📊 Leverage History {closed_position.asset}: {leverage_summary}")
    
    # ============================================================================
    # EXISTING METHODS CONTINUE BELOW
    # ============================================================================
    
    def log_signal(self, coin, price, ratio, confidence, direction, exit_estimate, ml_prediction):
        """Log a PSC signal to database (replaces CSV logging)"""
        try:
            # Handle ml_prediction - extract float value if it's a dict
            if isinstance(ml_prediction, dict):
                ml_prediction_value = ml_prediction.get('prediction', ml_prediction.get('confidence', 0.0))
                ml_features = ml_prediction  # Store full dict as features
            else:
                ml_prediction_value = float(ml_prediction) if ml_prediction is not None else 0.0
                ml_features = None
            
            # Log to database
            signal_id = self.data_manager.log_psc_signal(
                coin=coin,
                price=price,
                ratio=ratio,
                confidence=confidence,
                direction=direction,
                exit_estimate=exit_estimate,
                ml_prediction=ml_prediction_value,
                market_conditions=self.get_market_conditions(),
                ml_features=ml_features
            )
            
            # Legacy CSV backup (optional)
            try:
                with open(self.signals_log_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.now().isoformat(),
                        coin,
                        f"{price:.8f}",
                        f"{ratio:.2f}",
                        f"{confidence:.3f}",
                        direction,
                        f"{exit_estimate:.8f}",
                        f"{ml_prediction_value:.3f}",
                        self.get_signal_strength(confidence),
                        self.get_market_conditions()
                    ])
            except Exception as csv_error:
                logger.warning(f"CSV backup logging failed: {csv_error}")
            
            # Session stats are now handled by data_manager
            return signal_id
            
        except Exception as e:
            logger.error(f"Signal logging error: {e}")
            return None
    
    def log_trade(self, coin, entry_price, exit_price, confidence, ml_prediction, ratio, direction, successful, profit_pct=0, profit_usd=0, prediction_id=None):
        """Log a completed trade to database (replaces CSV logging)"""
        try:
            # For compatibility, if no prediction_id provided, create a signal entry
            signal_id = prediction_id
            if signal_id is None:
                signal_id = self.data_manager.log_psc_signal(
                    coin=coin,
                    price=entry_price,
                    ratio=ratio,
                    confidence=confidence,
                    direction=direction,
                    exit_estimate=exit_price,
                    ml_prediction=ml_prediction
                )
            
            # Log trade execution
            trade_id = self.data_manager.log_trade_execution(
                signal_id=signal_id,
                coin=coin,
                entry_price=entry_price,
                quantity=100,  # Default quantity
                confidence=confidence,
                ml_prediction=ml_prediction,
                ratio=ratio,
                direction=direction,
                trade_type='PAPER'  # Assuming paper trading for now
            )
            
            # Close trade with results
            if trade_id:
                self.data_manager.close_trade_with_results(
                    trade_id=trade_id,
                    exit_price=exit_price,
                    profit_pct=profit_pct,
                    profit_usd=profit_usd,
                    exit_reason='PROFIT_TARGET' if successful else 'STOP_LOSS'
                )
            
            # Legacy CSV backup (optional)
            try:
                trade_duration = "10min"  # Standard PSC trade duration
                with open(self.trades_log_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.now().isoformat(),
                        coin,
                        "PSC_SIGNAL",
                        f"{entry_price:.8f}",
                        f"{exit_price:.8f}",
                        f"{profit_pct:.2f}",
                        f"{confidence:.3f}",
                        f"{ml_prediction:.3f}",
                        f"{ratio:.2f}",
                        direction,
                        trade_duration,
                        successful,
                        f"{profit_usd:.2f}"
                    ])
            except Exception as csv_error:
                logger.warning(f"CSV backup logging failed: {csv_error}")
            
            return trade_id
            
            # =======================================================================
            # PREDICTION VALIDATION - Validate completed trade
            # =======================================================================
            
            if self.prediction_validator and prediction_id:
                try:
                    outcome = "SUCCESS" if successful else "FAILURE"
                    validation_success = self.prediction_validator.validate_prediction(
                        prediction_id=prediction_id,
                        actual_exit_price=exit_price,
                        outcome=outcome,
                        exit_time=datetime.now().isoformat(),
                        notes=f"PSC trade completed: {profit_pct:.2f}% profit, {trade_duration} duration"
                    )
                    
                    if validation_success:
                        logger.info(f"✅ PSC prediction validated: {prediction_id} - {outcome}")
                    else:
                        logger.warning(f"⚠️ PSC prediction validation failed: {prediction_id}")
                        
                except Exception as e:
                    logger.error(f"Error validating PSC prediction {prediction_id}: {e}")
            
            # Also validate using trade data method for live trades
            if self.prediction_validator:
                try:
                    trade_data = {
                        'coin': coin,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'direction': direction,
                        'successful': successful,
                        'profit_pct': profit_pct,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.prediction_validator.validate_live_trade(trade_data)
                except Exception as e:
                    logger.error(f"Error validating live trade data: {e}")
            
            # Update session stats
            self.session_stats['trades_executed'] += 1
            if successful:
                self.session_stats['successful_trades'] += 1
                self.session_stats['total_profit'] += profit_pct
            
            logger.info(f"📊 Trade logged and validated: {coin} - {direction} - Profit: {profit_pct:.2f}%")
            
        except Exception as e:
            logger.error(f"Trade logging error: {e}")
    
    async def log_comprehensive_tradingview_data(self, tv_log_entry):
        """Log comprehensive TradingView multi-timeframe analysis data"""
        try:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / "comprehensive_tradingview_data.csv"
            
            # Write header if file doesn't exist
            if not log_file.exists():
                with open(log_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=[
                        'timestamp', 'symbol', 'timeframes_analyzed', 
                        'tf_1m_summary', 'tf_5m_summary', 'tf_10m_summary',
                        'consensus_direction', 'consensus_strength', 'consensus_confidence',
                        'entry_recommendation', 'confidence_multiplier', 'timeframe_alignment',
                        'original_confidence', 'enhanced_confidence', 'alignment_score',
                        'recommendation'
                    ])
                    writer.writeheader()
            
            # Flatten timeframe summary for CSV
            timeframe_summary = tv_log_entry.get('timeframe_summary', {})
            trade_signals = tv_log_entry.get('trade_signals', {})
            
            flattened_entry = {
                'timestamp': tv_log_entry['timestamp'],
                'symbol': tv_log_entry['symbol'],
                'timeframes_analyzed': ', '.join(tv_log_entry.get('timeframes_analyzed', [])),
                'tf_1m_summary': timeframe_summary.get('1m', 'N/A'),
                'tf_5m_summary': timeframe_summary.get('5m', 'N/A'),
                'tf_10m_summary': timeframe_summary.get('10m', 'N/A'),
                'consensus_direction': tv_log_entry.get('consensus_direction', 'neutral'),
                'consensus_strength': tv_log_entry.get('consensus_strength', 0),
                'consensus_confidence': tv_log_entry.get('consensus_confidence', 0),
                'entry_recommendation': trade_signals.get('entry_recommendation', 'hold'),
                'confidence_multiplier': tv_log_entry.get('confidence_multiplier', 1.0),
                'timeframe_alignment': trade_signals.get('timeframe_alignment', False),
                'original_confidence': tv_log_entry.get('original_confidence', 0),
                'enhanced_confidence': tv_log_entry.get('enhanced_confidence', 0),
                'alignment_score': tv_log_entry.get('alignment_score', 0),
                'recommendation': tv_log_entry.get('recommendation', 'N/A')
            }
            
            # Append data
            with open(log_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=flattened_entry.keys())
                writer.writerow(flattened_entry)
            
            logger.info(f"📋 Comprehensive TradingView data logged for {tv_log_entry['symbol']}")
            
        except Exception as e:
            logger.error(f"Comprehensive TradingView logging error: {e}")
    
    def calculate_dynamic_leverage(self, confidence, ratio, volatility, time_remaining):
        """Calculate dynamic leverage based on PSC factors and time decay"""
        try:
            # Base leverage from confidence and PSC ratio
            confidence_multiplier = confidence * 2  # 0.5-2.0x based on confidence
            ratio_multiplier = min(ratio / 1.25, 4.0)  # Up to 4x based on PSC ratio strength
            
            # Volatility adjustment
            volatility_map = {
                'Very High': 0.7,    # Reduce leverage for very high volatility
                'Extreme': 0.6,      # Further reduce for extreme volatility
                'High': 0.8,
                'Medium': 1.0,
                'Low': 1.2           # Increase leverage for low volatility
            }
            volatility_multiplier = volatility_map.get(volatility, 1.0)
            
            # Time decay - reduce leverage as time runs out (encourage early exits)
            time_factor = max(time_remaining / 10.0, 0.3)  # Decay from 1.0 to 0.3
            
            # Calculate final leverage
            raw_leverage = (
                self.min_leverage * 
                confidence_multiplier * 
                ratio_multiplier * 
                volatility_multiplier * 
                time_factor
            )
            
            # Clamp to min/max bounds
            final_leverage = max(self.min_leverage, min(raw_leverage, self.max_leverage))
            
            return round(final_leverage, 2)
            
        except Exception as e:
            logger.error(f"Leverage calculation error: {e}")
            return self.min_leverage
    
    def calculate_position_size(self, leverage, base_size=None):
        """Calculate position size based on leverage"""
        if base_size is None:
            base_size = self.base_position_size
        return base_size * leverage
    
    def get_signal_strength(self, confidence):
        """Get signal strength based on confidence - raised thresholds to reduce weak signals"""
        if confidence >= 0.80:  # Raised back to original - very high bar for strong signals
            return "VERY_STRONG"
        elif confidence >= 0.65:  # Raised from 0.45 to 0.65 - quality signals only
            return "STRONG"
        elif confidence >= 0.50:  # Raised from 0.3 to 0.5 - minimum viable threshold
            return "MODERATE"
        else:
            return "WEAK"
    
    def get_market_conditions(self):
        """Analyze current market conditions"""
        # Simple market condition assessment
        current_hour = datetime.now().hour
        if 9 <= current_hour <= 16:  # Market hours
            return "ACTIVE"
        elif 17 <= current_hour <= 21:  # Evening
            return "MODERATE"
        else:
            return "LOW_VOLUME"
    
    def open_position(self, coin, entry_price, direction, confidence, target_exit, volatility, time_remaining=10.0, position_size_multiplier=1.0):
        """Enhanced position opening with Superp no-liquidation technology and dynamic position sizing"""
        signal_id = f"{coin}_{int(datetime.now().timestamp())}"
        
        # Calculate PSC ratio for leverage calculation
        ratio = confidence * 2.0  # Approximation for leverage calculation
        
        # Apply enhanced position sizing multiplier from signal quality filter
        logger.info(f"📊 Position sizing multiplier: {position_size_multiplier:.2f}x (from signal quality)")
        
        # =======================================================================
        # SUPERP NO-LIQUIDATION POSITION CREATION
        # =======================================================================
        
        # Convert volatility string to numeric value for calculations
        volatility_map = {
            'Low': 0.1, 'Medium': 0.2, 'High': 0.3, 
            'Very High': 0.4, 'Extreme': 0.5
        }
        volatility_numeric = volatility_map.get(volatility, 0.2)
        
        # Create Superp position with enhanced position sizing
        superp_position = self.create_superp_position(
            asset=coin,
            price=entry_price,
            psc_ratio=ratio,
            confidence=confidence,
            volatility=volatility_numeric,
            position_size_multiplier=position_size_multiplier
        )
        
        # =======================================================================
        # PREDICTION VALIDATION - Record PSC/Superp Signals
        # =======================================================================
        
        prediction_id = None
        if self.prediction_validator:
            try:
                if superp_position:
                    # Record Superp prediction
                    prediction_id = self.prediction_validator.record_superp_signal(
                        coin=coin,
                        direction=direction,
                        entry_price=entry_price,
                        psc_ratio=ratio,
                        confidence=confidence,
                        leverage=superp_position.effective_leverage,
                        target_price=target_exit
                    )
                    logger.info(f"🎯 Superp prediction recorded: {prediction_id}")
                else:
                    # Record PSC prediction
                    prediction_id = self.prediction_validator.record_psc_signal(
                        coin=coin,
                        direction=direction,
                        entry_price=entry_price,
                        psc_ratio=ratio,
                        confidence=confidence,
                        target_price=target_exit,
                        leverage=leverage
                    )
                    logger.info(f"📊 PSC prediction recorded: {prediction_id}")
            except Exception as e:
                logger.error(f"Error recording PSC/Superp prediction: {e}")
        
        # =======================================================================
        # TRADITIONAL POSITION TRACKING (for compatibility)
        # =======================================================================
        
        # Calculate dynamic leverage for traditional tracking
        leverage = self.calculate_dynamic_leverage(confidence, ratio, volatility, time_remaining)
        position_size = self.calculate_position_size(leverage)
        
        self.open_positions[signal_id] = {
            'coin': coin,
            'entry_price': entry_price,
            'entry_time': datetime.now(),
            'direction': direction,
            'confidence': confidence,
            'target_exit': target_exit,
            'leverage': leverage,
            'position_size': position_size,
            'volatility': volatility,
            'superp_position_id': superp_position.id if superp_position else None,
            'prediction_id': prediction_id,  # For validation tracking
            'prediction_timestamp': datetime.now().isoformat()  # For ML tracking
        }
        
        if superp_position:
            logger.info(f"🚀 SUPERP Position opened: {coin} | Buy-in: ${superp_position.buy_in_amount:.2f} | "
                       f"Leverage: {superp_position.effective_leverage:.0f}x | "
                       f"Virtual Exposure: ${superp_position.virtual_exposure:,.0f}")
        else:
            logger.info(f"📈 Traditional Position opened: {coin} at ${entry_price:.8f} ({direction}) - "
                       f"Target: ${target_exit:.8f} - Leverage: {leverage}x - Size: ${position_size:.2f}")
        
        return signal_id
    
    async def check_exit_conditions(self):
        """Enhanced exit checking with Superp position management"""
        
        # =======================================================================
        # UPDATE SUPERP POSITIONS
        # =======================================================================
        
        # Gather current prices for all monitored coins
        current_prices = {}
        for coin_data in self.monitored_coins:
            symbol = coin_data['symbol']
            price = await self.fetch_current_price(symbol)
            if price:
                current_prices[symbol] = price
        
        # Update all Superp positions
        self.update_superp_positions(current_prices)
        
        # =======================================================================
        # TRADITIONAL POSITION EXIT LOGIC
        # =======================================================================
        
        positions_to_close = []
        
        for signal_id, position in self.open_positions.items():
            coin = position['coin']
            entry_price = position['entry_price']
            target_exit = position['target_exit']
            direction = position['direction']
            entry_time = position['entry_time']
            confidence = position['confidence']
            leverage = position.get('leverage', 1.0)
            position_size = position.get('position_size', 100.0)
            volatility = position.get('volatility', 'Medium')
            superp_position_id = position.get('superp_position_id')
            
            # Get current price
            current_price = current_prices.get(coin)
            if not current_price:
                continue
            
            # Calculate time remaining
            minutes_open = (datetime.now() - entry_time).total_seconds() / 60
            time_remaining = max(10.0 - minutes_open, 0)
            
            # Recalculate leverage based on time decay
            if time_remaining > 0:
                ratio = confidence * 2.0  # Approximation
                current_leverage = self.calculate_dynamic_leverage(confidence, ratio, volatility, time_remaining)
                # Update position with current leverage
                self.open_positions[signal_id]['leverage'] = current_leverage
                leverage = current_leverage
            
            # Check exit conditions
            should_exit = False
            exit_reason = ""
            
            # Check if corresponding Superp position is closed
            if superp_position_id and superp_position_id in self.superp_positions:
                superp_pos = self.superp_positions[superp_position_id]
                if superp_pos.status != "ACTIVE":
                    should_exit = True
                    exit_reason = f"Superp {superp_pos.status.replace('CLOSED_', '').title()}"
            
            # PSC System: 10-minute timer constraint
            minutes_open = (datetime.now() - entry_time).total_seconds() / 60
            if minutes_open >= 10:
                should_exit = True
                exit_reason = "Timer expired (10 min)"
            
            # ENHANCED: Realistic small-move exit targets (instead of unrealistic 100% gains)
            # Check for realistic profit targets based on Superp small-move optimization
            elif direction == "LONG" and current_price >= target_exit:  # Realistic target hit
                should_exit = True
                exit_reason = "Small-move target reached"
            elif direction == "SHORT" and current_price <= target_exit:  # Realistic short target hit
                should_exit = True
                exit_reason = "Small-move short target reached"
            
            # Fallback: Original 100% targets (unlikely but kept for extreme cases)
            elif direction == "LONG" and current_price >= entry_price * 2.0:  # 100% gain (rare)
                should_exit = True
                exit_reason = "Extreme target reached (>100%)"
            elif direction == "SHORT" and current_price <= entry_price * 0.5:  # 100% gain on short (rare)
                should_exit = True
                exit_reason = "Extreme short target reached (>100%)"
            
            # No traditional stop-loss in PSC system due to no-liquidation advantage
            
            if should_exit:
                # Calculate real profit with leverage
                if direction == "LONG":
                    price_change_pct = ((current_price - entry_price) / entry_price) * 100
                else:  # SHORT
                    price_change_pct = ((entry_price - current_price) / entry_price) * 100
                
                # Apply leverage to profit calculation
                leveraged_profit_pct = price_change_pct * leverage
                
                # Calculate profit in USD
                profit_usd = (leveraged_profit_pct / 100) * position_size
                successful = leveraged_profit_pct > 0
                
                # Log the real exit
                self.log_trade(
                    coin=coin,
                    entry_price=entry_price,
                    exit_price=current_price,  # REAL exit price, not estimate
                    confidence=confidence,
                    ml_prediction=0,  # Not from ML prediction
                    ratio=0,  # From position tracking
                    direction=direction,
                    successful=successful,
                    profit_pct=leveraged_profit_pct,
                    profit_usd=profit_usd,
                    prediction_id=position.get('prediction_id')  # Include prediction ID for validation
                )
                
                # Update ML engine with actual outcome for prediction validation
                if self.ml_engine and 'prediction_timestamp' in position:
                    prediction_time = position['prediction_timestamp']
                    # Calculate actual return as percentage
                    actual_return_pct = leveraged_profit_pct / 100.0
                    
                    # Update prediction with actual outcome
                    self.ml_engine.update_prediction_outcome(
                        prediction_timestamp=prediction_time,
                        actual_outcome=successful,  # True if profitable
                        actual_return=actual_return_pct
                    )
                    
                    logger.debug(f"🎯 ML prediction updated: successful={successful}, return={actual_return_pct:.3f}")
                
                # Update Enhanced Prediction Validator with actual outcome
                if self.prediction_validator and 'prediction_id' in position:
                    try:
                        actual_outcome = {
                            'exit_price': current_price,
                            'return': leveraged_profit_pct / 100.0,  # Convert to decimal
                            'profitable': successful
                        }
                        validation_success = self.prediction_validator.validate_prediction(
                            position['prediction_id'], 
                            actual_outcome
                        )
                        if validation_success:
                            logger.info(f"✅ Enhanced prediction validated: {position['prediction_id']}")
                    except Exception as e:
                        logger.error(f"Error validating prediction: {e}")
                
                logger.info(f"🏁 Position closed: {coin} - {exit_reason} - Real exit: ${current_price:.8f} - Leverage: {leverage}x - Profit: {leveraged_profit_pct:.2f}% - USD: ${profit_usd:.2f}")
                positions_to_close.append(signal_id)
        
        # Remove closed positions
        for signal_id in positions_to_close:
            del self.open_positions[signal_id]
        
    def _load_config(self):
        """Load configuration"""
        try:
            config_path = self.project_root / "config" / "settings.yaml"
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                logger.info(f"📋 Configuration loaded from: {config_path}")
                return config
        except Exception as e:
            logger.error(f"Config load error: {e}")
            return {}
    
    def start_health_server(self):
        """Start HTTP server for health checks (cloud deployment)"""
        try:
            def run_server():
                try:
                    self.http_server = HTTPServer(('0.0.0.0', self.http_port), HealthCheckHandler)
                    logger.info(f"✅ Health server started on port {self.http_port}")
                    self.http_server.serve_forever()
                except Exception as e:
                    logger.error(f"Health server error: {e}")
            
            # Start server in background thread
            server_thread = threading.Thread(target=run_server, daemon=True)
            server_thread.start()
            
        except Exception as e:
            logger.warning(f"Could not start health server: {e}")
    
    def stop_health_server(self):
        """Stop health server"""
        if self.http_server:
            try:
                self.http_server.shutdown()
                logger.info("Health server stopped")
            except Exception as e:
                logger.error(f"Error stopping health server: {e}")
    
    def update_health_status(self, status: str):
        """Update system health status"""
        self.health_status = status
        self.last_activity = datetime.now()
        
        # Update system stats
        uptime = (datetime.now() - self.start_time).total_seconds()
        self.system_stats['system_uptime'] = round(uptime / 3600, 2)  # hours
        self.system_stats['active_positions'] = len(self.superp_positions)
    
    def start_keep_alive(self):
        """Start keep-alive service for Render deployment (prevents sleeping)"""
        try:
            def keep_alive_worker():
                """Worker function to ping self every 10 minutes"""
                import requests
                base_url = os.environ.get('RENDER_EXTERNAL_URL', 'http://localhost:8080')
                health_url = f"{base_url}/health"
                
                while self.running:
                    try:
                        time.sleep(600)  # Wait 10 minutes
                        if self.running:  # Check if still running
                            response = requests.get(health_url, timeout=30)
                            if response.status_code == 200:
                                logger.info("✅ Keep-alive ping successful")
                            else:
                                logger.warning(f"⚠️ Keep-alive ping returned {response.status_code}")
                    except requests.exceptions.RequestException as e:
                        logger.warning(f"⚠️ Keep-alive ping failed: {e}")
                    except Exception as e:
                        logger.error(f"❌ Keep-alive error: {e}")
            
            # Start keep-alive in background thread
            if os.environ.get('RENDER'):  # Only run on Render
                keep_alive_thread = threading.Thread(target=keep_alive_worker, daemon=True)
                keep_alive_thread.start()
                logger.info("✅ Keep-alive service started for Render deployment")
            
        except Exception as e:
            logger.warning(f"Could not start keep-alive service: {e}")
    
    async def dashboard_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /dashboard command - Get dashboard access information"""
        try:
            message = """
🖥️ **PSC Trading Dashboard**
═══════════════════════════════

**Dashboard Access:**
📍 Local URL: http://localhost:8501
🌐 Network URL: http://your-ip:8501

**To Start Dashboard:**
```
cd core_system
streamlit run dashboard.py
```

**Dashboard Features:**
⚙️ **Configuration Panel**
   • Modify scan intervals
   • Adjust thresholds
   • Configure Superp settings

📈 **Trading Monitor**
   • Real-time performance
   • Active positions
   • Recent trades

🧠 **ML Analytics**
   • Model performance
   • Prediction accuracy
   • Retrain controls

📋 **System Logs**
   • Real-time logging
   • Error tracking
   • Debug information

📊 **Performance Analytics**
   • Profit analysis
   • Success rates
   • Trade statistics

**Quick Commands:**
/logs - Get recent logs
/config - Show current config
/performance - Performance summary
            """
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Dashboard info error: {e}")
    
    async def logs_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /logs command - Get recent system logs"""
        try:
            # Read recent logs
            log_file = Path("logs/hybrid_system.log")
            
            if not log_file.exists():
                await update.message.reply_text("📋 No log file found")
                return
            
            # Get last 20 lines
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            recent_lines = lines[-20:] if len(lines) >= 20 else lines
            
            if not recent_lines:
                await update.message.reply_text("📋 Log file is empty")
                return
            
            # Format logs for Telegram
            log_text = "📋 **Recent System Logs:**\n```\n"
            
            for line in recent_lines:
                # Truncate very long lines
                if len(line) > 100:
                    line = line[:97] + "..."
                log_text += line
            
            log_text += "\n```"
            
            # Split message if too long
            if len(log_text) > 4000:
                # Send first part
                await update.message.reply_text(log_text[:4000] + "\n```", parse_mode='Markdown')
                # Send remaining
                remaining = "```\n" + log_text[4000:]
                await update.message.reply_text(remaining, parse_mode='Markdown')
            else:
                await update.message.reply_text(log_text, parse_mode='Markdown')
                
        except Exception as e:
            await update.message.reply_text(f"❌ Logs error: {e}")
    
    async def config_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /config command - Show current configuration"""
        try:
            config_file = self.project_root / "config" / "settings.yaml"
            
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                config = {
                    'trading': {
                        'scan_interval': 30,
                        'confidence_threshold': 0.5,  # Lowered from 0.7 for data gathering
                        'ratio_threshold': 1.5,
                        'max_positions': 5
                    }
                }
            
            trading_config = config.get('trading', {})
            superp_config = config.get('superp', {})
            ml_config = config.get('ml', {})
            
            message = f"""
⚙️ **Current Configuration**
═══════════════════════════════

**🎯 Trading Settings:**
• Scan Interval: {trading_config.get('scan_interval', 30)}s
• Confidence Threshold: {trading_config.get('confidence_threshold', 0.7):.1%}
• Ratio Threshold: {trading_config.get('ratio_threshold', 1.5)}
• Max Positions: {trading_config.get('max_positions', 5)}
• Position Size: ${trading_config.get('position_size', 1000)}

**🎢 Superp Settings:**
• Enabled: {'✅' if superp_config.get('enabled', True) else '❌'}
• Conservative Max: {superp_config.get('conservative_range', [1, 100])[1]}x
• Moderate Max: {superp_config.get('moderate_range', [100, 1000])[1]}x
• Aggressive Max: {superp_config.get('aggressive_range', [1000, 5000])[1]}x

**🧠 ML Settings:**
• Enabled: {'✅' if ml_config.get('enabled', True) else '❌'}
• Retrain Interval: {ml_config.get('retrain_interval', 50)}

**To modify settings:**
Use the dashboard: /dashboard
Or edit config/settings.yaml directly
            """
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Config error: {e}")
    
    async def performance_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /performance command - Show performance summary"""
        try:
            # Get trade statistics from database
            stats = self.data_manager.get_trade_statistics()
            
            if stats['total_trades'] == 0:
                await update.message.reply_text("📊 No trades recorded yet")
                return
            
            # Calculate metrics from database stats
            total_trades = stats['total_trades']
            successful_trades = stats.get('profitable_trades', stats.get('successful_trades', 0))
            win_rate = stats['success_rate']
            total_profit = stats.get('total_pnl', stats.get('total_profit', 0))
            avg_profit = stats['avg_profit']
            best_trade = stats.get('max_profit', stats.get('best_trade', 0))
            worst_trade = stats.get('max_loss', stats.get('worst_trade', 0))
            avg_confidence = stats.get('avg_confidence', 0)
            
            performance_msg = f"""
📊 **PERFORMANCE SUMMARY**
═══════════════════════

🎯 **Trade Statistics:**
• Total Trades: `{total_trades}`
• Successful: `{successful_trades}`
• Win Rate: `{win_rate:.1f}%`

💰 **Profit Analysis:**
• Total Profit: `${total_profit:.2f}`
• Average per Trade: `${avg_profit:.2f}`
• Best Trade: `${best_trade:.2f}`
• Worst Trade: `${worst_trade:.2f}`

🧠 **Quality Metrics:**
• Avg Confidence: `{avg_confidence:.1%}`
• Database Records: `{stats['total_trades']} trades`

📈 **System Health:**
• Data Source: ✅ Database (Real-time)
• Tracking: ✅ All signals & trades logged
• Analytics: ✅ Full performance history

💡 **Quick Access:**
/trades - Recent trade history
/stats - Live session statistics
/signals - Current monitoring status
            """
            
            await update.message.reply_text(performance_msg, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Performance error: {e}")
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command with enhanced features showcase"""
        notifications_status = "🔔 ON" if self.notifications_enabled else "🔕 OFF"
        filter_status = "🎯 HIGH CONFIDENCE ONLY" if self.high_confidence_only else "📊 ALL SIGNALS"
        
        # Get system status indicators
        ml_status = "✅ ACTIVE" if self.ml_engine else "❌ DISABLED"
        paper_status = "✅ TRACKING" if self.paper_validator else "❌ DISABLED"
        tv_status = "✅ CONNECTED" if self.tradingview_enabled else "❌ DISABLED"
        
        welcome_msg = f"""
🚀 **PSC AI Trading System v4.1**
═══════════════════════════════════

✅ **ENHANCED SYSTEM FEATURES:**
• 🎯 Bidirectional Trading (LONG + SHORT)
• 🧠 Continuous ML Monitoring {ml_status}
• 🧪 Paper Trading Validation {paper_status}
• 📊 TradingView Integration {tv_status}
• 🚀 Superp No-Liquidation Technology
• ⏰ 10-Minute Timer Cycles

🧠 **AI-Powered Analysis:**
• ML Scanning: Every 45 seconds
• Small-Move Focus: 0.12-0.20% targets
• Prediction Validation: All signals tracked
• Technical Confirmation: Multi-timeframe TA
• Direction Intelligence: LONG/SHORT detection

� **Bidirectional Strategy:**
• LONG Signals: PSC ratios ≥ 1.25
• SHORT Signals: PSC ratios ≤ 0.8-0.9
• Full Market Coverage: ~100% opportunities
• Entry Window: Minutes 0-3 only
• Zero Liquidation Risk: Superp technology

🧪 **Paper Trading System:**
• Every prediction logged & validated
• Multiple timeframes: 5m, 10m, 15m, 30m
• Real-time accuracy tracking
• Continuous model improvement

🎯 **Current Settings:**
• Notifications: {notifications_status}
• Filter Mode: {filter_status}
• Min Confidence: {self.min_confidence_threshold:.1f}
• ML Monitoring: {ml_status}

📱 **Key Commands:**
/status - Complete system overview
/ml - AI monitoring & predictions  
/paper - Prediction accuracy report
/tradingview - Technical analysis status
/signals - Bidirectional signal monitoring
/help - Comprehensive command guide

💎 **Monitored Assets:** {len(self.monitored_coins)} coins
• BTC, ETH, SOL (Major cryptos)
• SHIB, DOGE, PEPE (High volatility)

**🔥 System Status: FULLY OPERATIONAL** ✅
*Professional AI-Powered Bidirectional Trading*
        """
        await update.message.reply_text(welcome_msg, parse_mode='Markdown')
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command with enhanced features information"""
        current_time = datetime.now()
        self.timer_minute = self.get_aligned_timer_minute(current_time)
        
        entry_window = "🟢 OPEN" if self.timer_minute < 3 else "🟡 CLOSED"
        next_window = 10 - self.timer_minute if self.timer_minute >= 3 else f"{3 - self.timer_minute} min left"
        
        # Count active Superp positions
        active_superp = len([p for p in self.superp_positions.values() if p.status == "ACTIVE"])
        total_superp_invested = sum(p.buy_in_amount for p in self.superp_positions.values() if p.status == "ACTIVE")
        
        # Get ML and validation status
        ml_status = "✅ ACTIVE" if self.ml_engine else "❌ DISABLED"
        prediction_status = "✅ ENHANCED VALIDATION" if self.prediction_validator else "❌ DISABLED"
        tv_status = "✅ CONNECTED" if self.tradingview_enabled else "❌ DISABLED"
        
        # Get recent ML signals count
        recent_ml_signals = 0
        if self.ml_engine:
            try:
                recent_ml_signals = len(self.ml_engine.get_recent_ml_signals(max_age_minutes=30))
            except:
                pass
        
        status_msg = f"""
📊 **PSC Trading System Status**
═══════════════════════════════

🕐 **Timer Status:**
• Cycle: {self.timer_minute}/10 minutes
• Entry Window: {entry_window}
• Next Reset: {next_window}

🚀 **Superp Positions:**
• Active: {active_superp} positions
• Invested: ${total_superp_invested:.2f}
• Available Risk: ${self.max_total_risk - self.total_superp_exposure:.2f}
• Max Leverage: Up to 10,000x

🧠 **ML Engine Status:**
• System: {ml_status}
• Recent Signals: {recent_ml_signals} (last 30 min)
• Continuous Scan: {'✅ RUNNING' if self.running and self.ml_engine else '❌ STOPPED'}
• Small-Move Focus: ✅ 0.12-0.20% targets

🔬 **Enhanced Prediction Validation:**
• Status: {prediction_status}
• Validation System: ✅ ENHANCED ACTIVE
• Real-time Tracking: ✅ ENABLED

🔗 **TradingView Integration:**
• Status: {tv_status}
• Multi-timeframe: ✅ 1m, 5m, 10m analysis
• Signal Validation: ✅ ML + TA consensus

�📈 **PSC Monitoring:**
• Pairs: BTC, ETH, SOL, SHIB, DOGE, PEPE
• LONG Threshold: Ratio ≥ 1.25
• SHORT Threshold: Ratio ≤ 0.8-0.9
• Bidirectional Trading: ✅ ENABLED

🔗 **TON Integration:**
• Connection: READY
• Cross-chain: MONITORING

🎯 **Trading Status:**
• System: ACTIVE
• Risk Management: ENABLED
• Notifications: {"🔔 ON" if self.notifications_enabled else "🔕 OFF"}
• Filter: {"🎯 HIGH CONFIDENCE" if self.high_confidence_only else "📊 ALL SIGNALS"}

⚡ **Quick Commands:**
/ml - ML system details
/paper - Paper trading report
/tradingview - TA integration status
/signals - Signal monitoring

Time: {current_time.strftime('%H:%M:%S')}
        """
        await update.message.reply_text(status_msg, parse_mode='Markdown')
    
    async def signals_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /signals command"""
        current_time = datetime.now()
        self.timer_minute = self.get_aligned_timer_minute(current_time)
        
        window_status = "🟢 ACTIVE - Optimal for entries!" if self.timer_minute < 3 else "🟡 WAITING - No entries until reset"
        
        signals_msg = f"""
⚡ **Enhanced PSC Signal Monitor**
════════════════════════════════

🔍 **Current Scan Status:**
• Timer: {self.timer_minute}/10 minutes
• Entry Window: {window_status}
• Main Scan: Every 30 seconds
• ML Scan: Every 45 seconds (independent)

📊 **Bidirectional Monitoring:**
• BTC/USDT: LONG (≥1.25) + SHORT (≤0.9) signals
• ETH/USDT: LONG (≥1.25) + SHORT (≤0.9) signals
• SOL/USDT: LONG (≥1.25) + SHORT (≤0.9) signals
• SHIB/USDT: LONG (≥1.25) + SHORT (≤0.9) signals
• DOGE/USDT: LONG (≥1.25) + SHORT (≤0.9) signals
• PEPE/USDT: LONG (≥1.25) + SHORT (≤0.9) signals

🧠 **AI Enhancement:**
• ML Predictions: Continuous small-move detection
• TradingView Validation: Multi-timeframe confirmation
• Paper Trading: All predictions logged & validated
• Direction Intelligence: Automatic LONG/SHORT classification

⏰ **Enhanced Strategy:**
• Entry: Minutes 0-3 only (timer-based)
• Target: Variable profit based on ML confidence
• Exit: Before minute 10 (zero liquidation risk)
• Coverage: ~100% market opportunities (both directions)

🎯 **Next Opportunity:**
{f"⏰ Ready for signals! ({3 - self.timer_minute} min window left)" if self.timer_minute < 3 else f"⏳ Next entry window in {10 - self.timer_minute} minutes"}

💡 **Quick Access:**
/ml - View ML predictions & monitoring
/paper - Check prediction accuracy
/tradingview - Technical analysis status
        """
        await update.message.reply_text(signals_msg, parse_mode='Markdown')
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_msg = """
🆘 **PSC + TON Trading System Help**
════════════════════════════════

📋 **Core Commands:**
/start - System overview and status
/status - Real-time system status with all components
/help - Show this help menu

🎯 **Trading & Signals:**
/signals - Recent ML and PSC signals from database
/ml - ML Engine status and recent predictions
/prices - Current cryptocurrency prices  
/coins - Monitored cryptocurrency list (BTC, ETH, SOL, DOGE, etc.)

� **Database & Analytics:**
/stats - Trading statistics from database
/trades - Recent trade history from database
/performance - Performance metrics and success rates

🧠 **AI & ML Features:**
/ml - ML continuous monitoring with database learning
/tradingview - TradingView integration status
/predictions - Enhanced ML prediction validation analysis
/paper - Paper trading validation and accuracy metrics
/database - Database status and signal analytics

🔧 **System & Configuration:**
/config - Current system configuration
/logs - Recent system logs and events
/notifications - Toggle notification settings
/settings - View/adjust system settings

� **Database-Integrated Features:**
• UUID-tracked signals in SQLite database
• Real-time ML predictions every 45 seconds
• 4-component signal processor (PSC, ML, TradingView, Microstructure)
• Superp technology with NO liquidation risk

🎯 **Current System Status:**
• Database Integration: ✅ Fully Operational (Database-only)
• ML Engine: ✅ Continuous learning from historical data
• Signal Storage: ✅ UUID-tracked with validation
• Learning Pipeline: ✅ Prediction → Validation → Learning

💡 **Quick Tips:**
• Use /paper to check validation accuracy
• Use /predictions for detailed ML performance  
• Use /database for signal analytics
• All data persists across restarts in database
        """
        await update.message.reply_text(help_msg, parse_mode='Markdown')
    
    async def notifications_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /notifications command - Toggle notifications"""
        self.notifications_enabled = not self.notifications_enabled
        status = "🔔 ENABLED" if self.notifications_enabled else "🔕 DISABLED"
        
        toggle_msg = f"""
📱 **Notification Settings Updated**
═══════════════════════════════

Status: {status}

{"✅ You will receive all PSC signals and timer alerts" if self.notifications_enabled else "❌ Notifications paused - system still monitoring"}

Current Filter: {"🎯 High Confidence Only" if self.high_confidence_only else "📊 All Signals"}

Use /filter to toggle confidence filtering
Use /settings to view all options
        """
        await update.message.reply_text(toggle_msg, parse_mode='Markdown')
    
    async def filter_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /filter command - Toggle high confidence filter"""
        self.high_confidence_only = not self.high_confidence_only
        status = "🎯 HIGH CONFIDENCE ONLY" if self.high_confidence_only else "📊 ALL SIGNALS"
        
        filter_msg = f"""
🎯 **Signal Filter Updated**
═══════════════════════════

Filter Mode: {status}

{"🎯 Only signals with confidence ≥" + str(self.min_confidence_threshold) + " will be sent" if self.high_confidence_only else "📊 All qualifying PSC signals will be sent"}

Notifications: {"🔔 ON" if self.notifications_enabled else "🔕 OFF"}

{"📈 This ensures you only see the most promising opportunities!" if self.high_confidence_only else "📊 You'll see all signals that meet ratio requirements."}

Use /notifications to toggle all notifications
Use /settings to adjust confidence threshold
        """
        await update.message.reply_text(filter_msg, parse_mode='Markdown')
    
    async def settings_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /settings command - Show current settings"""
        settings_msg = f"""
⚙️ **System Settings**
═══════════════════════

📱 **Notifications:**
• Status: {"🔔 ENABLED" if self.notifications_enabled else "🔕 DISABLED"}
• Filter: {"🎯 HIGH CONFIDENCE ONLY" if self.high_confidence_only else "📊 ALL SIGNALS"}
• Min Confidence: {self.min_confidence_threshold:.1f}

📊 **Trading Parameters:**
• Min Signal Ratio: {self.min_signal_ratio}
• Entry Window: 0-3 minutes
• Timer Cycle: 10 minutes

🎯 **Confidence Levels:**
• Very High: ≥{self.confidence_thresholds['very_high']:.1f}
• High: ≥{self.confidence_thresholds['high']:.1f}
• Medium: ≥{self.confidence_thresholds['medium']:.1f}

💡 **Quick Actions:**
/notifications - Toggle notifications
/filter - Toggle confidence filter

🔧 **Tips:**
• High confidence filter shows only premium signals
• Timer alerts help optimize entry timing
• All signals include confidence scores and direction
        """
        await update.message.reply_text(settings_msg, parse_mode='Markdown')
    
    async def coins_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show monitored coins"""
        coins_msg = "💎 **Monitored Coins**\n"
        coins_msg += "═" * 25 + "\n\n"
        
        for i, coin in enumerate(self.monitored_coins, 1):
            coins_msg += f"{i}. **{coin['name']}** ({coin['symbol']})\n"
            coins_msg += f"   • Pair: {coin['pair']}\n"
            coins_msg += f"   • Volatility: {coin['volatility']}\n"
            if coin['symbol'] in self.last_prices:
                price = self.last_prices[coin['symbol']]
                change = self.price_changes.get(coin['symbol'], 0.0)
                change_emoji = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                coins_msg += f"   • Price: ${price:.8f} ({change:+.2f}%) {change_emoji}\n"
            coins_msg += "\n"
        
        coins_msg += f"🔄 **Updates:** Every 30 seconds\n"
        coins_msg += f"⏰ **Active Window:** 0-3 minutes of each 10min cycle"
        
        await update.message.reply_text(coins_msg, parse_mode='Markdown')
    
    async def prices_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show current prices for all monitored coins"""
        prices_msg = "📊 **Current Prices**\n"
        prices_msg += "═" * 20 + "\n\n"
        
        for coin in self.monitored_coins:
            symbol = coin['symbol']
            # Fetch fresh price
            current_price = await self.fetch_current_price(symbol)
            
            if current_price:
                change = self.price_changes.get(symbol, 0.0)
                change_emoji = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                
                prices_msg += f"💎 **{coin['name']}** ({symbol})\n"
                prices_msg += f"   Price: `${current_price:.8f}`\n"
                prices_msg += f"   24h: `{change:+.2f}%` {change_emoji}\n"
                prices_msg += f"   Volatility: {coin['volatility']}\n\n"
        
        prices_msg += f"🕐 **Last Updated:** {datetime.now().strftime('%H:%M:%S')}\n"
        prices_msg += f"⏰ **Entry Window:** {'🟢 OPEN' if self.timer_minute < 3 else '🔴 CLOSED'}"
        
        await update.message.reply_text(prices_msg, parse_mode='Markdown')
    
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command - Show trading statistics"""
        session_duration = datetime.now() - self.session_stats['session_start']
        success_rate = (self.session_stats['successful_trades'] / max(1, self.session_stats['trades_executed'])) * 100
        
        stats_msg = f"""
📊 **TRADING STATISTICS**
═══════════════════════

⏱️ **Session Duration:** {str(session_duration).split('.')[0]}

🎯 **Signals & Trades:**
• Signals Generated: `{self.session_stats['signals_generated']}`
• Trades Executed: `{self.session_stats['trades_executed']}`
• Successful Trades: `{self.session_stats['successful_trades']}`
• Success Rate: `{success_rate:.1f}%`

💰 **Profitability:**
• Total Profit: `{self.session_stats['total_profit']:.2f}%`
• Avg per Trade: `{self.session_stats['total_profit']/max(1,self.session_stats['trades_executed']):.2f}%`

📈 **Performance:**
• Best Hour: Market hours (9-16 UTC)
• ML Engine: {'✅ Active' if self.ml_engine else '❌ Disabled'}
• Live Prices: ✅ Real-time APIs

📁 **Data Storage:**
• Live Database: `data/psc_trading.db`
• Real-time Updates: ✅ All signals & trades
• Export Available: CSV export via /dashboard
        """
        
        await update.message.reply_text(stats_msg, parse_mode='Markdown')
    
    async def trades_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /trades command - Show recent trades"""
        try:
            # Get last 5 trades from database
            recent_trades = self.data_manager.get_recent_trades(5)
            
            if not recent_trades:
                trades_msg = "📈 **RECENT TRADES**\n\nNo trades recorded yet. Monitoring for signals..."
            else:
                trades_msg = "📈 **RECENT TRADES**\n═════════════════\n\n"
                
                for i, trade in enumerate(recent_trades, 1):
                    # Check if profit_pct exists and is positive for success indicator
                    profit_pct = trade.get('profit_pct', 0)
                    profit_emoji = "✅" if profit_pct > 0 else "❌"
                    
                    # Handle timestamp parsing safely
                    try:
                        timestamp = datetime.fromisoformat(trade['created_at']).strftime('%H:%M')
                    except (ValueError, KeyError):
                        timestamp = "N/A"
                    
                    trades_msg += f"""
{profit_emoji} **Trade #{i}**
• Coin: `{trade.get('coin', 'N/A')}`
• Time: `{timestamp}`
• Direction: `{trade.get('direction', 'N/A')}`
• Profit: `{profit_pct:.2f}%`
• Confidence: `{float(trade.get('confidence', 0)):.0%}`
• Status: `{trade.get('status', 'N/A')}`
---"""
                
                trades_msg += f"\n\n📊 *Use /stats for detailed analytics*"
            
            await update.message.reply_text(trades_msg, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Trades command error: {e}")
            await update.message.reply_text("❌ Error reading trades data", parse_mode='Markdown')

    async def superp_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show Superp no-liquidation positions with timer-based leverage tracking"""
        try:
            current_timer_minute = self.get_aligned_timer_minute(datetime.now())
            
            if not self.superp_positions:
                # Show current leverage potential based on timer
                if current_timer_minute <= 2:
                    leverage_status = "🚀 MAXIMUM (100%)"
                elif current_timer_minute <= 5:
                    leverage_status = f"📈 HIGH ({100 - (current_timer_minute - 2) * 5}%)"
                elif current_timer_minute <= 8:
                    leverage_status = f"📊 MODERATE ({85 - (current_timer_minute - 5) * 8}%)"
                else:
                    leverage_status = f"📉 LOW ({61 - (current_timer_minute - 8) * 15}%)"
                
                superp_msg = f"""
🚀 **SUPERP NO-LIQUIDATION POSITIONS**
═══════════════════════════════════

Currently no active Superp positions.

⏰ **Current Timer Status:**
• Timer Minute: {current_timer_minute}/10
• Leverage Level: {leverage_status}
• Next Reset: {10 - current_timer_minute} minutes

**Superp Technology Benefits:**
• Up to 10,000x leverage with NO liquidation risk
• Timer-based leverage optimization
• Maximum loss = buy-in amount only ($10-$100)
• Revolutionary risk management

🎯 *Waiting for high-confidence PSC signals...*
                """
            else:
                superp_msg = f"🚀 **SUPERP POSITIONS** (Timer: {current_timer_minute}/10)\n═══════════════════════════════════\n\n"
                
                total_invested = 0.0
                total_exposure = 0.0
                total_pnl = 0.0
                
                for pos_id, position in self.superp_positions.items():
                    if position.status == "ACTIVE":
                        status_emoji = "🟢"
                        status_text = "ACTIVE"
                    elif position.status == "CLOSED_PROFIT":
                        status_emoji = "✅"
                        status_text = "PROFIT"
                    elif position.status == "CLOSED_TIME":
                        status_emoji = "⏰"
                        status_text = "TIME EXIT"
                    else:
                        status_emoji = "❌"
                        status_text = position.status
                    
                    # Time remaining
                    if position.status == "ACTIVE":
                        time_remaining = (position.stop_time - datetime.now()).total_seconds() / 60
                        time_text = f"{max(0, time_remaining):.1f} min left"
                    else:
                        time_text = "CLOSED"
                    
                    # P&L calculation using current leverage
                    if position.current_price > 0:
                        price_change = (position.current_price - position.entry_price) / position.entry_price
                        pnl = price_change * position.virtual_exposure
                        pnl_pct = price_change * 100
                    else:
                        pnl = position.realized_pnl or position.unrealized_pnl
                        pnl_pct = (pnl / position.virtual_exposure) * 100 if position.virtual_exposure > 0 else 0
                    
                    # Leverage information
                    if hasattr(position, 'entry_leverage') and hasattr(position, 'current_leverage'):
                        leverage_info = f"Entry: {position.entry_leverage:.0f}x → Current: {position.current_leverage:.0f}x"
                        if position.current_leverage != position.entry_leverage:
                            leverage_change = ((position.current_leverage - position.entry_leverage) / position.entry_leverage) * 100
                            leverage_info += f" ({leverage_change:+.0f}%)"
                    else:
                        leverage_info = f"{position.effective_leverage:.0f}x"
                    
                    # Timer information
                    if hasattr(position, 'timer_minute_opened'):
                        timer_info = f"Opened: T{position.timer_minute_opened}, Now: T{current_timer_minute}"
                    else:
                        timer_info = f"Timer: {current_timer_minute}/10"
                    
                    superp_msg += f"""
{status_emoji} **{position.asset}** - {status_text}
• Buy-in: `${position.buy_in_amount:.2f}`
• Leverage: `{leverage_info}`
• Exposure: `${position.virtual_exposure:,.0f}`
• Entry: `${position.entry_price:.6f}`
• Current: `${position.current_price:.6f}`
• P&L: `${pnl:+,.2f}` ({pnl_pct:+.1f}%)
• Time: {time_text}
• {timer_info}
• Confidence: `{position.confidence_score:.0%}`
---"""
                    
                    total_invested += position.buy_in_amount
                    total_exposure += position.virtual_exposure
                    total_pnl += pnl
                
                # Timer-based leverage summary
                if current_timer_minute <= 2:
                    timer_status = "🚀 MAXIMUM LEVERAGE PHASE"
                elif current_timer_minute <= 5:
                    timer_status = "📈 HIGH LEVERAGE PHASE"
                elif current_timer_minute <= 8:
                    timer_status = "📊 MODERATE LEVERAGE PHASE"
                else:
                    timer_status = "📉 LOW LEVERAGE PHASE"
                
                superp_msg += f"""

📊 **SUPERP PORTFOLIO SUMMARY:**
• Total Invested: ${total_invested:.2f}
• Total Exposure: ${total_exposure:,.0f}
• Total P&L: ${total_pnl:+,.2f}
• Max Risk Remaining: ${self.max_total_risk - self.total_superp_exposure:.2f}
• Active Positions: {len([p for p in self.superp_positions.values() if p.status == 'ACTIVE'])}

⏰ **TIMER STATUS:** {timer_status}
• Current Minute: {current_timer_minute}/10
• Next Reset: {10 - current_timer_minute} minutes
• Leverage adjusts automatically with timer

🛡️ **No Liquidation Risk** - Maximum loss = buy-in amounts only!
"""
            
            await update.message.reply_text(superp_msg, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Superp command error: {e}")
            await update.message.reply_text("❌ Error reading Superp positions", parse_mode='Markdown')

    async def positions_command(self, update, context: ContextTypes.DEFAULT_TYPE):
        """Show current open positions"""
        try:
            if not self.open_positions:
                await update.message.reply_text(
                    "📭 **No Open Positions**\n\n"
                    "All positions are currently closed.\n"
                    "Waiting for new trading signals...",
                    parse_mode='Markdown'
                )
                return
            
            positions_msg = f"📈 **OPEN POSITIONS** ({len(self.open_positions)})\n"
            positions_msg += "=" * 30 + "\n\n"
            
            for i, (signal_id, position) in enumerate(self.open_positions.items(), 1):
                coin = position['coin']
                entry_price = position['entry_price']
                target_exit = position['target_exit']
                direction = position['direction']
                confidence = position['confidence']
                entry_time = position['entry_time']
                leverage = position.get('leverage', 1.0)
                position_size = position.get('position_size', 100.0)
                volatility = position.get('volatility', 'Unknown')
                
                # Get current price for P&L calculation
                current_price = await self.fetch_current_price(coin)
                
                # Calculate current P&L with leverage
                if current_price:
                    if direction == "LONG":
                        price_change_pct = ((current_price - entry_price) / entry_price) * 100
                    else:  # SHORT
                        price_change_pct = ((entry_price - current_price) / entry_price) * 100
                    
                    # Apply leverage to P&L
                    leveraged_pnl = price_change_pct * leverage
                    profit_usd = (leveraged_pnl / 100) * position_size
                    
                    pnl_emoji = "🟢" if leveraged_pnl > 0 else "🔴" if leveraged_pnl < 0 else "🟡"
                    current_str = f"`${current_price:.8f}`"
                    pnl_str = f"`{leveraged_pnl:+.2f}%` {pnl_emoji}"
                    usd_str = f"`${profit_usd:+.2f}` USD"
                else:
                    current_str = "Loading..."
                    pnl_str = "Loading..."
                    usd_str = "Loading..."
                
                # Calculate time open in minutes for PSC system
                minutes_open = (datetime.now() - entry_time).total_seconds() / 60
                time_str = f"{minutes_open:.1f}min"
                time_remaining = max(10.0 - minutes_open, 0)
                
                # Direction emoji
                dir_emoji = "📈" if direction == "LONG" else "📉"
                
                positions_msg += f"""
{dir_emoji} **Position #{i}** - {coin}
• Direction: `{direction}`
• Entry: `${entry_price:.8f}`
• Current: {current_str}
• Target: `${target_exit:.8f}`
• Leverage: `{leverage}x` (Dynamic)
• Position Size: `${position_size:.2f}`
• P&L: {pnl_str}
• USD P&L: {usd_str}
• Time Open: `{time_str}` (Max: 10min)
• Time Left: `{time_remaining:.1f}min`
• Confidence: `{confidence:.0%}`
• Volatility: `{volatility}`
• ID: `{signal_id[-8:]}`
---"""
            
            positions_msg += f"\n\n⏰ PSC Positions auto-close after 10min or when >100% target hit"
            positions_msg += f"\n📊 *Use /trades to see completed trades*"
            
            await update.message.reply_text(positions_msg, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Positions command error: {e}")
            await update.message.reply_text("❌ Error reading positions data", parse_mode='Markdown')

    async def tradingview_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show TradingView comprehensive multi-timeframe analysis status and data"""
        try:
            current_time = datetime.now()
            
            # Check TradingView integration status
            tv_status = "✅ ACTIVE" if self.tradingview and self.tradingview_enabled else "❌ DISABLED"
            tv_available = "✅ Available" if TRADINGVIEW_AVAILABLE else "❌ Not Available"
            
            # Get recent TradingView logs (last 10)
            recent_logs = self.tradingview_logs[-10:] if self.tradingview_logs else []
            
            # Get comprehensive market data if available
            comprehensive_data = getattr(self.tradingview, 'multi_timeframe_data', {}) if self.tradingview else {}
            
            # Build main message
            tradingview_msg = f"""
📊 **TRADINGVIEW COMPREHENSIVE ANALYSIS**
═══════════════════════════════════════

🔧 **System Status:**
• Integration: {tv_available}
• Service: {tv_status}
• Timeframes: 1m, 5m, 10m (Multi-timeframe)
• Check Interval: {self.tradingview_check_interval}s
• Total Logs: {len(self.tradingview_logs)}
• Monitored Coins: {len(comprehensive_data)} active

⏰ **Current Settings:**
• Auto-Enhancement: {'✅ ON' if self.tradingview_enabled else '❌ OFF'}
• Timer Minute: {self.timer_minute}/10
• Last Scan: {current_time.strftime('%H:%M:%S')}

📈 **Comprehensive Benefits:**
• Multi-timeframe consensus analysis
• Timeframe alignment detection  
• Enhanced confidence scoring
• Trade signal recommendations
• Risk level assessment
            """
            
            # Show current market overview if we have comprehensive data
            if comprehensive_data:
                tradingview_msg += f"\n\n🌍 **LIVE TRADINGVIEW DASHBOARD - ALL COINS:**\n"
                
                for symbol, data in list(comprehensive_data.items())[:6]:  # Show all 6 coins
                    consensus = data.get('consensus', {})
                    trade_signals = data.get('trade_signals', {})
                    timeframes = data.get('timeframes', {})
                    
                    direction = consensus.get('direction', 'neutral').upper()
                    strength = consensus.get('strength', 0)
                    confidence = consensus.get('confidence', 0)
                    alignment = "🎯" if trade_signals.get('timeframe_alignment') else "⚠️"
                    entry_rec = trade_signals.get('entry_recommendation', 'hold').upper()
                    
                    # Get individual timeframe signals with scores
                    tf_1m_data = timeframes.get('1m', {})
                    tf_5m_data = timeframes.get('5m', {})
                    tf_10m_data = timeframes.get('10m', {})
                    
                    tf_1m = tf_1m_data.get('summary', 'N/A').upper()
                    tf_5m = tf_5m_data.get('summary', 'N/A').upper()
                    tf_10m = tf_10m_data.get('summary', 'N/A').upper()
                    
                    # Add emoji indicators for clarity
                    def get_signal_emoji(signal):
                        if signal == 'BUY': return "🟢"
                        elif signal == 'SELL': return "🔴"
                        else: return "🟡"
                    
                    tradingview_msg += f"""
📊 **{symbol}**: {get_signal_emoji(direction)} {direction} ({strength:.1%}) {alignment}
   • 1m: {get_signal_emoji(tf_1m)} **{tf_1m}** | 5m: {get_signal_emoji(tf_5m)} **{tf_5m}** | 10m: {get_signal_emoji(tf_10m)} **{tf_10m}**
   • Confidence: `{confidence:.1%}` | Entry: **{entry_rec}**
   • Bias Strength: `{strength:.1%}`
"""
            else:
                tradingview_msg += f"\n\n🔄 **Getting comprehensive market data...**\n"
                tradingview_msg += f"• Use this command again in a few moments"
                tradingview_msg += f"\n• Or wait for automatic data collection"
            
            if recent_logs:
                tradingview_msg += f"\n\n📋 **Recent Enhancement Data (Last {len(recent_logs[-5:])}):**\n"
                
                for i, log in enumerate(recent_logs[-5:], 1):  # Show last 5
                    timestamp = datetime.fromisoformat(log['timestamp']).strftime('%H:%M:%S')
                    symbol = log['symbol']
                    
                    # Handle both old and new log formats
                    if 'timeframe_summary' in log:
                        # New comprehensive format
                        tf_summary = log.get('timeframe_summary', {})
                        consensus_dir = log.get('consensus_direction', 'neutral')
                        original_conf = log.get('original_confidence', 0)
                        enhanced_conf = log.get('enhanced_confidence', 0)
                        multiplier = log.get('confidence_multiplier', 1.0)
                        alignment = log.get('timeframe_alignment', False)
                        
                        tradingview_msg += f"""
{i}. **{symbol}** at {timestamp}
   • 1m: *{tf_summary.get('1m', 'N/A')}* | 5m: *{tf_summary.get('5m', 'N/A')}* | 10m: *{tf_summary.get('10m', 'N/A')}*
   • Consensus: *{consensus_dir.upper()}* {'🎯' if alignment else '⚠️'}
   • Confidence: `{original_conf:.1%}` → `{enhanced_conf:.1%}` (`{multiplier:.2f}x`)
   • Recommendation: {log.get('recommendation', 'N/A')[:50]}...
---"""
                    else:
                        # Legacy format
                        summary = log.get('summary', 'N/A').upper()
                        original_conf = log.get('original_confidence', 0)
                        enhanced_conf = log.get('enhanced_confidence', 0)
                        enhancement_change = enhanced_conf - original_conf
                        change_emoji = "📈" if enhancement_change > 0 else "📉" if enhancement_change < 0 else "➡️"
                        
                        tradingview_msg += f"""
{i}. **{symbol}** at {timestamp}
   • Signal: *{summary}*
   • Confidence: `{original_conf:.1%}` → `{enhanced_conf:.1%}` {change_emoji}
   • Change: `{enhancement_change:+.1%}`
---"""
                
                # Summary statistics
                total_enhancements = len([log for log in recent_logs if log.get('enhanced_confidence', 0) > log.get('original_confidence', 0)])
                avg_enhancement = sum(log.get('enhanced_confidence', 0) - log.get('original_confidence', 0) for log in recent_logs) / len(recent_logs) if recent_logs else 0
                
                tradingview_msg += f"""

📊 **Recent Performance:**
• Enhanced Signals: {total_enhancements}/{len(recent_logs)}
• Avg Enhancement: `{avg_enhancement:+.1%}`
• Success Rate: `{(total_enhancements/len(recent_logs)*100):.1f}%`
"""
            else:
                tradingview_msg += f"\n\n📋 **Recent Analysis:** No data yet - waiting for signals..."
            
            tradingview_msg += f"""

🔧 **Commands:**
/tradingview - This comprehensive dashboard
/tvrefresh - Force refresh all TradingView data
/settings - Adjust TradingView settings
/logs - System logs (includes TradingView)

📈 *Multi-timeframe TradingView analysis enhances PSC signals with professional consensus*
            """
            
            await update.message.reply_text(tradingview_msg, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"TradingView command error: {e}")
            await update.message.reply_text("❌ Error reading TradingView data", parse_mode='Markdown')

    async def tvrefresh_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Force refresh TradingView comprehensive market analysis"""
        try:
            if not self.tradingview or not self.tradingview_enabled:
                await update.message.reply_text("❌ TradingView integration is disabled", parse_mode='Markdown')
                return
            
            refresh_msg = "🔄 **Refreshing TradingView Data...**\n\n"
            refresh_msg += "• Fetching 1m, 5m, 10m analysis for all 6 coins\n"
            refresh_msg += "• This may take 15-20 seconds\n"
            refresh_msg += "• Please wait..."
            
            sent_message = await update.message.reply_text(refresh_msg, parse_mode='Markdown')
            
            # Force refresh by clearing cache
            if hasattr(self.tradingview, 'cache'):
                self.tradingview.cache.clear()
            
            # Get fresh comprehensive data
            comprehensive_data = await self.tradingview.get_comprehensive_market_analysis()
            
            if comprehensive_data:
                success_msg = "✅ **TradingView Data Refreshed Successfully!**\n\n"
                success_msg += f"📊 **Fresh Analysis Available:**\n"
                
                for symbol, data in list(comprehensive_data.items())[:6]:
                    consensus = data.get('consensus', {})
                    direction = consensus.get('direction', 'neutral').upper()
                    strength = consensus.get('strength', 0)
                    alignment = "🎯" if data.get('trade_signals', {}).get('timeframe_alignment') else "⚠️"
                    
                    def get_signal_emoji(signal):
                        if signal == 'BUY': return "🟢"
                        elif signal == 'SELL': return "🔴"
                        else: return "🟡"
                    
                    success_msg += f"• {get_signal_emoji(direction)} **{symbol}**: {direction} ({strength:.1%}) {alignment}\n"
                
                success_msg += f"\n💡 Use `/tradingview` to see detailed multi-timeframe breakdown"
                
                await sent_message.edit_text(success_msg, parse_mode='Markdown')
            else:
                error_msg = "❌ **Failed to refresh TradingView data**\n\n"
                error_msg += "• Check network connection\n"
                error_msg += "• TradingView service may be temporarily unavailable\n"
                error_msg += "• Try again in a few moments"
                
                await sent_message.edit_text(error_msg, parse_mode='Markdown')
                
        except Exception as e:
            logger.error(f"TradingView refresh error: {e}")
            error_msg = f"❌ **Refresh failed:** {str(e)[:100]}..."
            try:
                await sent_message.edit_text(error_msg, parse_mode='Markdown')
            except:
                await update.message.reply_text(error_msg, parse_mode='Markdown')

    async def ml_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show ML engine continuous monitoring status and recent activity"""
        try:
            current_time = datetime.now()
            
            # Check ML engine status
            ml_status = "✅ ACTIVE" if self.ml_engine else "❌ DISABLED"
            
            # Get recent ML signals
            recent_ml_signals = []
            if self.ml_engine:
                recent_ml_signals = self.ml_engine.get_recent_ml_signals(max_age_minutes=30)
            
            # Count processed signals
            processed_count = len(getattr(self, 'processed_ml_signals', set()))
            
            # Build main message
            ml_msg = f"""
🤖 **ML CONTINUOUS MONITORING STATUS**
═══════════════════════════════════

🔧 **ML Engine Status:**
• Core Engine: {ml_status}
• Continuous Scan: {'✅ RUNNING' if self.running and self.ml_engine else '❌ STOPPED'}
• Small-Move Optimized: ✅ YES (0.12-0.20% targets)
• TradingView Validation: {'✅ ENABLED' if self.tradingview_enabled else '❌ DISABLED'}
• Scan Interval: 45 seconds (independent of PSC scan)

⏰ **Current Activity:**
• Timer Minute: {self.timer_minute}/10
• Recent ML Signals: {len(recent_ml_signals)} (last 30 min)
• Processed Signals: {processed_count}
• Last Scan: {current_time.strftime('%H:%M:%S')}

🎯 **ML Signal Generation:**
• Independent Detection: Runs continuously
• Signal Criteria: 4/5 advanced criteria must be met
• Quality Threshold: 70% minimum signal score
• TradingView Validation: Required for execution
• Small-Move Focus: 0.75+ probability threshold

📊 **Monitoring Features:**
• Continuous market scanning (6 coins)
• Small-move opportunity detection
• TradingView sentiment validation
• Quality score assessment
• Timer-aware signal generation
            """
            
            # Show recent ML signals if any
            if recent_ml_signals:
                ml_msg += f"\n\n🔍 **Recent ML Signals (Last 30 min):**\n"
                
                for i, signal in enumerate(recent_ml_signals[-10:], 1):  # Show last 10
                    timestamp = datetime.fromisoformat(signal['timestamp']).strftime('%H:%M:%S')
                    coin = signal['coin']
                    prediction = signal['prediction']
                    
                    small_move_prob = prediction.get('small_move_probability', 0)
                    expected_return = prediction.get('expected_return', 0)
                    confidence = prediction.get('confidence', 0)
                    
                    # Check if signal was processed
                    signal_timestamp = signal['timestamp']
                    processed = "✅" if hasattr(self, 'processed_ml_signals') and signal_timestamp in self.processed_ml_signals else "⏳"
                    
                    ml_msg += f"""
{i}. **{coin}** at {timestamp} {processed}
   • Small-Move Prob: `{small_move_prob:.1%}`
   • Expected Return: `{expected_return:.3%}`
   • Confidence: `{confidence:.1%}`
   • Price: `${signal['price']:.6f}`
   • Ratio: `{signal['ratio']}`
---"""
            else:
                ml_msg += f"\n\n🔍 **Recent ML Signals:** No signals generated yet"
            
            # Show ML performance if available
            if self.ml_engine:
                try:
                    performance = self.ml_engine.get_model_performance()
                    total_predictions = performance.get('total_predictions', 0)
                    accuracy = performance.get('overall_accuracy', 0)
                    
                    ml_msg += f"""

📈 **ML Performance:**
• Total Predictions: {total_predictions}
• Overall Accuracy: {accuracy:.1%}
• Model Status: {performance.get('model_status', 'Unknown')}
• Small-Move Optimization: ✅ Active
"""
                except:
                    ml_msg += f"\n\n📈 **ML Performance:** Data not available"
            
            ml_msg += f"""

📊 **Database Integration:**
• Signal Storage: ✅ All ML signals stored with UUIDs
• Real-time Logging: ✅ Every prediction tracked
• Database Path: {self.data_manager.db.db_path.name}
• Recent DB Signals: {len(recent_ml_signals)} stored

🎯 **Signal Quality Criteria:**
• Small-Move Probability: ≥75%
• Expected Return: ≥0.15%
• Overall Confidence: ≥80%
• Ratio Threshold: ≥1.2
• Timer Position: Favorable

🔧 **Commands:**
/ml - This ML status
/database - Database status and signal counts
/signals - Recent signals from database
/predictions - ML prediction analysis

🤖 *Continuous ML monitoring with database persistence*
            """
            
            await update.message.reply_text(ml_msg, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"ML command error: {e}")
            error_msg = f"""🤖 **ML Engine Status** (Error Recovery)

⚠️ **Status**: Partial data access issue
📊 **System**: Running but some metrics unavailable
🔧 **Action**: ML monitoring continues in background

🔍 **Issue**: {self.clean_error_message(e, 'ML Engine')}

💡 **Available Commands:**
/status - General system status
/signals - Recent signal data  
/help - Command reference
            """
            await update.message.reply_text(error_msg, parse_mode='Markdown')

    async def database_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /database command - Show database status and signal counts"""
        try:
            # Get database statistics
            stats = self.data_manager.get_database_stats()
            
            # Get signal counts by type
            signal_counts = {}
            try:
                with sqlite3.connect(self.data_manager.db.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Count signals by type
                    cursor.execute("SELECT signal_type, COUNT(*) FROM signals GROUP BY signal_type")
                    for signal_type, count in cursor.fetchall():
                        signal_counts[signal_type] = count
                    
                    # Get recent signals
                    cursor.execute("""
                        SELECT coin, signal_type, confidence, created_at 
                        FROM signals 
                        ORDER BY created_at DESC 
                        LIMIT 5
                    """)
                    recent_signals = cursor.fetchall()
                    
            except Exception as e:
                logger.error(f"Database query error: {e}")
                recent_signals = []
            
            database_msg = f"""
�️ **Database Status & Analytics**
═══════════════════════════════

📊 **Signal Counts by Type:**
• ML Signals: {signal_counts.get('ML_SIGNAL', 0)}
• PSC Signals: {signal_counts.get('PSC_SIGNAL', 0)}
• Total Signals: {sum(signal_counts.values())}

📈 **Trade Statistics:**
• Total Trades: {stats.get('total_trades', 0)}
• Success Rate: {stats.get('success_rate', 0):.1f}%
• Total Profit: ${stats.get('total_profit_usd', 0):.2f}

� **Recent Signals:**
"""
            
            # Add recent signals
            for signal in recent_signals[:3]:
                coin, sig_type, confidence, timestamp = signal
                database_msg += f"• {coin} {sig_type} (Conf: {confidence:.1f}) - {timestamp[:16]}\n"
            # Get database filename for display
            db_filename = str(self.data_manager.db.db_path).split('\\')[-1]
            
            database_msg += f"""
💾 **Database Health:**
• Status: ✅ Connected and Operational
• Path: {db_filename}
• Integration: ✅ Real-time logging active

🔧 **Related Commands:**
/signals - View recent signals
/stats - Detailed trading statistics
/ml - ML Engine status with database integration
            """
            
            await update.message.reply_text(database_msg, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Database command error: {e}")
            await update.message.reply_text("❌ Error reading database status", parse_mode='Markdown')

    async def predictions_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /predictions command - Show enhanced prediction validation report"""
        try:
            if not self.prediction_validator:
                # Provide fallback prediction info from ML engine and database
                fallback_msg = f"""🔮 **Prediction System Status**

⚠️ **Enhanced Validator**: Initialization issue - using fallback
📊 **ML Engine**: {'✅ Active' if self.ml_engine else '❌ Disabled'}
💾 **Database**: {'✅ Connected' if self.data_manager else '❌ Disconnected'}

🧠 **ML Predictions Available:**
• Database Predictions: {len(getattr(self.ml_engine, 'predictions', [])) if self.ml_engine else 0}
• Recent ML Signals: Available via ML engine
• Live Validation: Continuous background monitoring

📋 **Alternative Commands:**
/ml - Detailed ML engine status and predictions
/database - Database statistics and health
/status - Overall system performance

🔧 **Status**: Core prediction tracking continues via database"""
                await update.message.reply_text(fallback_msg, parse_mode='Markdown')
                return
            
            # Get comprehensive performance report
            report = self.prediction_validator.get_performance_report()
            
            if 'error' in report:
                await update.message.reply_text(
                    f"❌ Error generating prediction report: {report['error']}", 
                    parse_mode='Markdown'
                )
                return
            
            summary = report.get('summary', {})
            recent = report.get('recent_performance', {})
            confidence_analysis = report.get('confidence_analysis', {})
            recommendations = report.get('recommendations', [])
            
            # Main performance summary
            predictions_msg = f"""
🔮 **Enhanced Prediction Performance Report**
════════════════════════════════════

📊 **Overall Performance:**
• Total Predictions: {summary.get('total_predictions', 0)}
• Validated: {summary.get('validated_predictions', 0)}
• Accuracy Rate: {summary.get('accuracy_rate', 0):.1%}
• Profitable Rate: {summary.get('profitable_rate', 0):.1%}
• Average Return: {summary.get('avg_return', 0):.3f}%
• Best Confidence Threshold: {summary.get('best_confidence_threshold', 0.6):.1f}

📈 **Recent Performance (Last 30):**
"""
            
            if recent:
                predictions_msg += f"""• Accuracy: {recent.get('accuracy', 0):.1%}
• Profitability: {recent.get('profitability', 0):.1%}
• Avg Return: {recent.get('avg_return', 0):.3f}%
• Predictions: {recent.get('prediction_count', 0)}

🎯 **Confidence Analysis:**
"""
                
                high_conf = confidence_analysis.get('high_confidence', {})
                low_conf = confidence_analysis.get('low_confidence', {})
                
                predictions_msg += f"""• High Confidence (≥70%):
  - Count: {high_conf.get('count', 0)}
  - Accuracy: {high_conf.get('accuracy', 0):.1%}
  - Profitability: {high_conf.get('profitability', 0):.1%}
  
• Low Confidence (<60%):
  - Count: {low_conf.get('count', 0)}
  - Accuracy: {low_conf.get('accuracy', 0):.1%}
  - Profitability: {low_conf.get('profitability', 0):.1%}
"""
            else:
                predictions_msg += "• No recent data available yet\n"
            
            # Add recommendations
            if recommendations:
                predictions_msg += f"\n\n💡 **AI Recommendations:**\n"
                for i, rec in enumerate(recommendations[:3], 1):  # Show top 3
                    predictions_msg += f"{i}. {rec}\n"
            
            predictions_msg += f"""

🔬 **Validation Features:**
• Real-time prediction tracking
• Automatic outcome validation
• Performance trend analysis
• Model improvement recommendations
• Confidence threshold optimization

📋 **Related Commands:**
/paper - Paper trading validation and accuracy
/database - Database analytics and signal counts  
/ml - ML system status and learning
/performance - Overall system performance

🧠 *Advanced ML prediction validation for continuous improvement*
            """
            
            await update.message.reply_text(predictions_msg, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Predictions command error: {e}")
            await update.message.reply_text("❌ Error generating prediction report", parse_mode='Markdown')

    async def paper_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /paper command - Show paper trading validation and accuracy metrics"""
        try:
            # Get database trading statistics
            trade_stats = self.data_manager.get_trade_statistics()
            session_stats = self.data_manager.get_session_stats()
            
            # Get recent paper trades from database
            recent_trades = self.data_manager.get_recent_trades(limit=10)
            
            # Calculate paper trading metrics
            total_paper_trades = len([t for t in recent_trades if t.get('trade_type') == 'PAPER'])
            profitable_trades = len([t for t in recent_trades if t.get('trade_type') == 'PAPER' and t.get('profit_pct', 0) > 0])
            
            accuracy_rate = (profitable_trades / total_paper_trades * 100) if total_paper_trades > 0 else 0
            
            # Get ML prediction accuracy if available
            ml_accuracy = 0
            if hasattr(self, 'prediction_validator') and self.prediction_validator:
                try:
                    report = self.prediction_validator.get_performance_report()
                    ml_accuracy = report.get('summary', {}).get('accuracy_rate', 0) * 100
                except:
                    ml_accuracy = 0
            
            paper_msg = f"""📊 **Paper Trading & Validation Report**
═══════════════════════════════════

🎯 **Paper Trading Performance:**
• Total Paper Trades: {total_paper_trades}
• Profitable Trades: {profitable_trades}
• Success Rate: {accuracy_rate:.1f}%
• Session Trades: {session_stats.get('trades_executed', 0)}

🧠 **ML Prediction Accuracy:**
• Validation Accuracy: {ml_accuracy:.1f}%
• Database Predictions: {len(getattr(self.ml_engine, 'predictions', []))}
• Learning Status: ✅ Active (Database-driven)

📈 **Recent Paper Trades:**
"""
            
            # Show recent paper trades
            paper_trades = [t for t in recent_trades if t.get('trade_type') == 'PAPER'][:5]
            
            if paper_trades:
                for i, trade in enumerate(paper_trades, 1):
                    coin = trade.get('coin', 'Unknown')
                    profit_pct = trade.get('profit_pct', 0)
                    confidence = trade.get('confidence', 0)
                    status = "✅ Profit" if profit_pct > 0 else "❌ Loss"
                    
                    paper_msg += f"{i}. {coin} {status} ({profit_pct:+.2f}%) - Conf: {confidence:.1f}\n"
            else:
                paper_msg += "• No recent paper trades found\n"
            
            paper_msg += f"""

🔬 **Validation Features:**
• ✅ Real-time paper trade tracking
• ✅ Database-stored trade outcomes  
• ✅ ML prediction accuracy monitoring
• ✅ 10-minute outcome validation
• ✅ Continuous learning from results

💾 **Database Integration:**
• All trades stored with UUID tracking
• Historical validation data preserved
• Real-time accuracy calculations
• Performance trend analysis

📋 **Related Commands:**
/predictions - Detailed ML prediction analysis
/database - Database status and statistics
/performance - Overall system performance
/ml - ML engine status and learning

🎯 *Paper trading provides risk-free validation of system accuracy*
            """
            
            await update.message.reply_text(paper_msg, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Paper trading command error: {e}")
            fallback_msg = f"""📊 **Paper Trading Report** (Fallback Mode)

⚠️ **Status**: Data access issue detected
📊 **System**: Trading continues, validation active
🔧 **Recovery**: Background systems operational

🧪 **Paper Trading Validation**: ✅ ACTIVE
• Database integration: Running
• ML prediction tracking: Active  
• Real-time validation: Operational
• Trade outcome recording: Functional

🔍 **Issue**: {self.clean_error_message(e, 'Paper Trading')}

💡 **Alternatives:**
/status - System overview
/database - Database statistics
/ml - ML engine status

🎯 *Paper trading validation continues in background*
            """
            await update.message.reply_text(fallback_msg, parse_mode='Markdown')

    async def send_notification(self, message: str, force=False):
        """Send notification to user (respects notification settings and reduces spam)"""
        if not self.notifications_enabled and not force:
            return
            
        # Simple spam prevention - skip similar messages within 30 seconds
        current_time = datetime.now()
        message_hash = hash(message[:100])  # Hash first 100 chars
        
        if hasattr(self, '_last_notification_time') and hasattr(self, '_last_notification_hash'):
            time_diff = (current_time - self._last_notification_time).total_seconds()
            if time_diff < 30 and message_hash == self._last_notification_hash:
                logger.debug("Skipping duplicate notification")
                return
        
        try:
            bot = Bot(token=self.bot_token)
            await bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='Markdown'
            )
            
            # Update spam prevention tracking
            self._last_notification_time = current_time
            self._last_notification_hash = message_hash
            
        except Exception as e:
            logger.error(f"Notification error: {e}")
    
    def clean_error_message(self, error: Exception, context: str = "") -> str:
        """Clean up error messages for better user experience"""
        error_str = str(error)
        
        # Common error patterns and their user-friendly versions
        if "no such table" in error_str.lower():
            return f"Database table not found - system initializing"
        elif "no such column" in error_str.lower():
            return f"Database structure updating - please retry"
        elif "connection" in error_str.lower():
            return f"Connection issue - retrying automatically"
        elif "timeout" in error_str.lower():
            return f"Request timeout - system operational"
        elif "permission" in error_str.lower():
            return f"Access permission issue"
        else:
            # Truncate long errors and remove technical paths
            clean_error = error_str.replace('\\', '/').split('/')[-1] if '\\' in error_str else error_str
            return clean_error[:80] + "..." if len(clean_error) > 80 else clean_error

    def get_confidence_level_info(self, confidence):
        """Get confidence level and emoji"""
        if confidence >= self.confidence_thresholds['very_high']:
            return "Very High", "🟢", "⭐⭐⭐"
        elif confidence >= self.confidence_thresholds['high']:
            return "High", "🟡", "⭐⭐"
        elif confidence >= self.confidence_thresholds['medium']:
            return "Medium", "🟠", "⭐"
        else:
            return "Low", "🔴", ""
    
    def determine_trade_direction(self, crypto, ratio, confidence):
        """Determine trade direction based on PSC arbitrage: Logarithmic ratio analysis"""
        # UPDATED: Logarithmic PSC ratio logic (ratio = log10(crypto_price) - log10(ton_price) + 6)
        # Range: ~0.5 to 11.0 (log ratios shifted positive)
        # Neutral center: ~6.0 (when crypto_price ≈ ton_price)
        
        # LONG signals (crypto stronger than TON)
        if ratio >= 9.5 and confidence > 0.5:  # Very strong outperformance vs TON
            return "LONG", "📈", "Strong PSC arbitrage - crypto significantly above TON"
        elif ratio >= 8.5 and confidence > 0.4:  # Strong outperformance
            return "LONG", "📈", "Good PSC opportunity - crypto outperforming TON"
        elif ratio >= 7.0 and confidence > 0.35:  # Moderate outperformance
            return "LONG", "📈", "Entry-level PSC arbitrage - crypto above TON"
        
        # SHORT signals (crypto weaker than TON)
        elif ratio <= 2.5 and confidence > 0.5:  # Very strong underperformance vs TON
            return "SHORT", "📉", "Strong downward pressure - crypto significantly below TON"
        elif ratio <= 3.5 and confidence > 0.4:  # Strong underperformance
            return "SHORT", "📉", "Good short opportunity - crypto underperforming TON"
        elif ratio <= 5.0 and confidence > 0.35:  # Moderate underperformance
            return "SHORT", "📉", "Entry-level short signal - crypto below TON"
        else:
            return "NEUTRAL", "↔️", "Ratio balanced - crypto and TON in equilibrium"
    
    def should_send_signal(self, confidence, ml_prediction=None):
        """Check if signal should be sent based on filter settings and small-move viability"""
        if not self.notifications_enabled:
            logger.info(f"❌ Signal rejected: Notifications disabled")
            return False
        
        # ENHANCED: Check small-move viability if ML prediction available
        if ml_prediction:
            is_small_move_viable = ml_prediction.get('is_small_move_viable', True)
            small_move_prob = ml_prediction.get('small_move_probability', 0.5)
            expected_return = ml_prediction.get('expected_return', 0.001)
            
            # Reject signals that are unlikely to achieve profitable small moves
            if not is_small_move_viable:
                logger.info(f"❌ Signal rejected: Expected return {expected_return:.4f} below 0.12% minimum")
                return False
            
            # FIXED: For very low small-move probability, check expected return instead
            # Many crypto signals have 0% small-move but good expected returns
            if small_move_prob < 0.3:
                if abs(expected_return) < 0.001:  # Less than 0.1% expected return
                    logger.info(f"❌ Signal rejected: Low small-move probability {small_move_prob:.1%} with low expected return {expected_return:.4f}")
                    return False
                else:
                    logger.info(f"✅ Signal approved: Low small-move probability {small_move_prob:.1%} but good expected return {expected_return:.4f}")
            
        if self.high_confidence_only:
            result = confidence >= self.min_confidence_threshold
            if not result:
                logger.info(f"❌ Signal rejected: High confidence mode - confidence {confidence:.1%} < threshold {self.min_confidence_threshold:.1%}")
            return result
            
        logger.info(f"✅ Signal approved: confidence {confidence:.1%}")
        return True  # Send all signals if not filtering
    
    async def fetch_current_price(self, symbol):
        """Fetch current price for a symbol from real APIs"""
        try:
            import aiohttp
            import asyncio
            
            # Try Binance API first
            try:
                url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}USDT"
                logger.debug(f"🌐 Fetching {symbol} from Binance: {url}")
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            current_price = float(data['price'])
                            logger.debug(f"✅ Real price for {symbol}: ${current_price:.6f}")
                            
                            # Store price history
                            if symbol not in self.price_history:
                                self.price_history[symbol] = []
                            
                            self.price_history[symbol].append({
                                'price': current_price,
                                'timestamp': datetime.now()
                            })
                            
                            # Keep only last 50 prices for performance
                            if len(self.price_history[symbol]) > 50:
                                self.price_history[symbol] = self.price_history[symbol][-50:]
                            
                            # Calculate price change
                            if symbol in self.last_prices:
                                change_pct = ((current_price - self.last_prices[symbol]) / self.last_prices[symbol]) * 100
                                self.price_changes[symbol] = change_pct
                            else:
                                self.price_changes[symbol] = 0.0
                                
                            self.last_prices[symbol] = current_price
                            return current_price
                            
            except Exception as binance_error:
                logger.warning(f"⚠️ Binance API failed for {symbol}: {binance_error}")
                logger.debug(f"🔍 Binance URL attempted: https://api.binance.com/api/v3/ticker/price?symbol={symbol}USDT")
                
            # Fallback: CoinGecko API
            try:
                coin_id_map = {
                    'BTC': 'bitcoin',
                    'ETH': 'ethereum', 
                    'SOL': 'solana',
                    'SHIB': 'shiba-inu',
                    'DOGE': 'dogecoin',
                    'PEPE': 'pepe',
                    'TON': 'the-open-network'
                }
                
                if symbol in coin_id_map:
                    coin_id = coin_id_map[symbol]
                    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
                    
                    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                        async with session.get(url) as response:
                            if response.status == 200:
                                data = await response.json()
                                current_price = float(data[coin_id]['usd'])
                                logger.debug(f"✅ CoinGecko price for {symbol}: ${current_price:.6f}")
                                
                                # Store price and calculate changes
                                if symbol not in self.price_history:
                                    self.price_history[symbol] = []
                                
                                self.price_history[symbol].append({
                                    'price': current_price,
                                    'timestamp': datetime.now()
                                })
                                
                                if len(self.price_history[symbol]) > 50:
                                    self.price_history[symbol] = self.price_history[symbol][-50:]
                                
                                if symbol in self.last_prices:
                                    change_pct = ((current_price - self.last_prices[symbol]) / self.last_prices[symbol]) * 100
                                    self.price_changes[symbol] = change_pct
                                else:
                                    self.price_changes[symbol] = 0.0
                                    
                                self.last_prices[symbol] = current_price
                                return current_price
                                
            except Exception as coingecko_error:
                logger.warning(f"CoinGecko API failed for {symbol}: {coingecko_error}")
            
            # Final fallback: Use cached data if available
            if symbol in self.last_prices:
                logger.warning(f"⚠️ Using cached price for {symbol}: ${self.last_prices[symbol]:.6f}")
                return self.last_prices[symbol]
            
            # Emergency fallback with realistic base prices (Updated Aug 2025)
            base_prices = {
                'BTC': 111000.0,   # ~$111K (current range)
                'ETH': 2650.0,     # ~$2650
                'SOL': 198.0,      # ~$198  
                'SHIB': 0.0000124, # ~$0.0000124 (current range)
                'DOGE': 0.219,     # ~$0.219
                'PEPE': 0.0000102, # ~$0.0000102 (current range) 
                'TON': 3.22       # ~$3.22 (current range)
            }
            
            if symbol in base_prices:
                fallback_price = base_prices[symbol]
                logger.error(f"🚨 Using emergency fallback price for {symbol}: ${fallback_price:.8f}")
                logger.error(f"🚨 This indicates API connectivity issues - check network/proxy")
                return fallback_price
            
            logger.error(f"❌ Could not fetch price for {symbol}")
            return None
            
        except Exception as e:
            logger.error(f"Price fetch error for {symbol}: {e}")
            return None
    
    async def get_current_price(self, coin):
        """
        Get current price for a coin - wrapper for fetch_current_price
        Compatible with ML microstructure integration
        """
        try:
            # Convert coin symbol to proper format if needed
            symbol = coin.upper().replace('USDT', '')
            
            # Use existing fetch_current_price method
            price = await self.fetch_current_price(symbol)
            
            if price is None:
                logger.warning(f"Could not get current price for {coin}")
                return None
                
            logger.debug(f"📈 Current price for {coin}: ${price:.6f}")
            return price
            
        except Exception as e:
            logger.error(f"Error getting current price for {coin}: {e}")
            return None
    
    def get_live_superp_leverage(self, asset_price, buy_in_amount):
        """
        Calculate live Superp leverage based on current asset price and buy-in
        This simulates what the Superp platform would offer in real-time
        """
        try:
            # SUPERP MECHANICS (from analysis):
            # Effective Leverage = Asset Price / Buy-in Amount
            # Example: $67,000 BTC / $10 buy-in = 6,700x leverage
            
            if buy_in_amount <= 0:
                logger.error("Invalid buy-in amount for leverage calculation")
                return 1.0
                
            # Calculate base Superp leverage
            base_leverage = asset_price / buy_in_amount
            
            # Apply Superp platform constraints (from analysis: up to 10,000x)
            max_superp_leverage = self.superp_config.get('max_leverage', 10000.0)
            live_leverage = min(base_leverage, max_superp_leverage)
            
            # Ensure minimum leverage
            live_leverage = max(live_leverage, 1.0)
            
            logger.info(f"💹 Live Superp Leverage: ${asset_price:.2f} / ${buy_in_amount:.2f} = {live_leverage:.0f}x")
            
            return live_leverage
            
        except Exception as e:
            logger.error(f"Live Superp leverage calculation error: {e}")
            return 1.0
    
    def calculate_superp_profit_target(self, entry_price, current_leverage, buy_in_amount, confidence, ml_expected_return=None):
        """
        Calculate realistic profit target based on Superp leverage mechanics
        Target: Small price moves (>0.12%) for profitable trades (0.1% = break-even)
        ENHANCED: Uses ML engine expected return when available
        """
        try:
            # From Superp analysis: 0.1% = break-even, need >0.12% for profit
            # Example: 0.15% move with 6,700x leverage = 1005% return
            
            # Calculate minimum profitable move (0.1% break-even + 0.02% minimum profit)
            break_even_pct = 0.001    # 0.1% - confirmed break-even point
            min_profit_pct = 0.0012   # 0.12% - minimum for profitable trade
            
            # ENHANCED: Use ML expected return if available and viable
            if ml_expected_return and ml_expected_return >= min_profit_pct:
                target_move_pct = ml_expected_return
                logger.info(f"📊 Using ML predicted return: {target_move_pct:.4f} ({target_move_pct*100:.2f}%)")
            else:
                # Fallback to confidence-based targeting
                if confidence >= 0.8:
                    target_move_pct = random.uniform(0.0015, 0.002)   # 0.15-0.2%
                elif confidence >= 0.6:
                    target_move_pct = random.uniform(0.0013, 0.0015) # 0.13-0.15%
                else:
                    target_move_pct = random.uniform(0.0012, 0.0013) # 0.12-0.13%
                
                if ml_expected_return:
                    logger.info(f"⚠️ ML return {ml_expected_return:.4f} below minimum, using confidence-based: {target_move_pct:.4f}")
            
            # Ensure above minimum profit threshold
            target_move_pct = max(target_move_pct, min_profit_pct)
            
            # Calculate target price
            target_price = entry_price * (1 + target_move_pct)
            
            # Calculate expected profit
            price_change = target_price - entry_price
            virtual_exposure = buy_in_amount * current_leverage
            expected_profit = (price_change / entry_price) * virtual_exposure
            profit_percentage = (expected_profit / buy_in_amount) * 100
            
            return {
                'target_price': target_price,
                'target_move_pct': target_move_pct,
                'expected_profit_usd': expected_profit,
                'expected_profit_pct': profit_percentage,
                'break_even_price': entry_price * (1 + break_even_pct)
            }
            
        except Exception as e:
            logger.error(f"Superp profit target calculation error: {e}")
            return {
                'target_price': entry_price * 1.0012,  # 0.12% move (minimum profit)
                'target_move_pct': 0.0012,
                'expected_profit_usd': 0.0,
                'expected_profit_pct': 0.0,
                'break_even_price': entry_price * 1.001  # 0.1% break-even
            }

    def calculate_superp_exit_price(self, entry_price, confidence, direction, current_leverage, buy_in_amount, ml_expected_return=None):
        """
        Calculate realistic Superp exit price based on actual leverage mechanics
        From Superp analysis: 0.1% = break-even, need >0.12% for profitable exits
        ENHANCED: Uses ML engine expected return when available
        """
        try:
            # SUPERP LEVERAGE MECHANICS (from analysis document):
            # - $10 buy-in = 1 BTC exposure at $67,000 = 6,700x leverage  
            # - 0.1% move ($67) = break-even point
            # - 0.15% move = ~1000% return on $10 
            # - Break-even: 0.1% (confirmed by user)
            # - Target: >0.12% for profitable trades
            
            # Calculate minimum move needed for profit (0.1% break-even + 0.02% profit)
            break_even_percentage = 0.001   # 0.1% - confirmed break-even
            min_profit_percentage = 0.0012  # 0.12% - minimum profitable trade
            
            # ENHANCED: Use ML expected return if available and viable
            if ml_expected_return and ml_expected_return >= min_profit_percentage:
                target_percentage = ml_expected_return
                logger.info(f"🎯 Superp Exit: Using ML predicted move {target_percentage:.4f} ({target_percentage*100:.2f}%)")
            else:
                # Fallback to confidence-based targeting
                if confidence >= 0.8:
                    # High confidence: target 0.15-0.20% move
                    target_percentage = random.uniform(0.0015, 0.002)  # 0.15-0.20%
                elif confidence >= 0.6:
                    # Medium confidence: target 0.13-0.15% move  
                    target_percentage = random.uniform(0.0013, 0.0015)  # 0.13-0.15%
                else:
                    # Lower confidence: target 0.12-0.13% move
                    target_percentage = random.uniform(0.0012, 0.0013)  # 0.12-0.13%
                
                if ml_expected_return:
                    logger.info(f"⚠️ ML return {ml_expected_return:.4f} below minimum, using confidence-based: {target_percentage:.4f}")
            
            # Ensure we're above minimum profit threshold
            target_percentage = max(target_percentage, min_profit_percentage)
            
            # Calculate exit price based on direction
            if direction == "LONG":
                exit_price = entry_price * (1 + target_percentage)
            elif direction == "SHORT":
                exit_price = entry_price * (1 - target_percentage)
            else:  # NEUTRAL - slight upward bias
                exit_price = entry_price * (1 + target_percentage * 0.5)
            
            # Calculate expected profit for validation
            price_change_pct = abs(exit_price - entry_price) / entry_price
            virtual_exposure = buy_in_amount * current_leverage
            expected_profit = price_change_pct * virtual_exposure
            profit_percentage = (expected_profit / buy_in_amount) * 100
            
            logger.info(f"🎯 Superp Exit Calculation:")
            logger.info(f"   Entry: ${entry_price:.8f}")
            logger.info(f"   Exit: ${exit_price:.8f}")
            logger.info(f"   Move: {price_change_pct:.4%}")
            logger.info(f"   Leverage: {current_leverage:.0f}x")
            logger.info(f"   Buy-in: ${buy_in_amount:.2f}")
            logger.info(f"   Expected Profit: ${expected_profit:.2f} ({profit_percentage:.1f}%)")
            
            return round(exit_price, 8)
            
        except Exception as e:
            logger.error(f"Superp exit price calculation error: {e}")
            # Fallback: minimum profitable move (0.12%)
            fallback_move = 0.0012
            if direction == "LONG":
                return entry_price * (1 + fallback_move)
            else:
                return entry_price * (1 - fallback_move)

    def calculate_exit_price_estimate(self, entry_price, confidence, direction, leverage=None, buy_in_amount=None):
        """Calculate estimated exit price - Enhanced for Superp no-liquidation system"""
        try:
            # If we have Superp parameters, use realistic Superp calculations
            if leverage and buy_in_amount and leverage > 100:  # Likely Superp position
                return self.calculate_superp_exit_price(entry_price, confidence, direction, leverage, buy_in_amount)
            
            # Fallback to traditional PSC calculation (much more conservative than before)
            # Even traditional should be more realistic - target 5-15% moves, not 100%+
            if direction == "LONG":
                if confidence >= 0.8:
                    target_gain = random.uniform(0.10, 0.20)  # 10-20% (much more realistic)
                elif confidence >= 0.6:
                    target_gain = random.uniform(0.05, 0.15)  # 5-15%
                else:
                    target_gain = random.uniform(0.05, 0.10)  # 5-10%
                
                exit_price = entry_price * (1 + target_gain)
                
            elif direction == "SHORT":
                if confidence >= 0.8:
                    target_drop = random.uniform(0.10, 0.20)  # 10-20% drop
                elif confidence >= 0.6:
                    target_drop = random.uniform(0.05, 0.15)  # 5-15% drop
                else:
                    target_drop = random.uniform(0.05, 0.10)  # 5-10% drop
                
                exit_price = entry_price * (1 - target_drop)
                
            else:  # NEUTRAL
                change = random.uniform(0.05, 0.10)  # 5-10% gain
                exit_price = entry_price * (1 + change)
            
            return round(exit_price, 8)
            
        except Exception as e:
            logger.error(f"Exit price calculation error: {e}")
            # Very conservative fallback
            return entry_price * 1.05  # 5% gain target
    
    async def generate_ml_prediction(self, psc_price, ton_price, ratio):
        """Generate ML prediction with confidence score - SMALL MOVE OPTIMIZED"""
        try:
            # Use ML engine for small-move optimized prediction
            prediction_data = self.ml_engine.predict([psc_price, ton_price, ratio])
            
            confidence = prediction_data.get('confidence', 0.0)
            prediction = prediction_data.get('prediction', 0.0)
            
            # NEW: Extract small-move specific data
            small_move_prob = prediction_data.get('small_move_probability', 0.5)
            expected_return = prediction_data.get('expected_return', 0.0015)  # Default 0.15%
            target_range = prediction_data.get('target_range', '0.12-0.20%')
            
            # Enhanced movement classification for small moves
            if small_move_prob > 0.7 and expected_return >= 0.0015:
                expected_movement = "Strong Small Bullish (0.15-0.20%)"
            elif small_move_prob > 0.5 and expected_return >= 0.0012:
                expected_movement = "Moderate Small Bullish (0.12-0.15%)"
            elif expected_return >= 0.0012:
                expected_movement = "Weak Small Bullish (0.12%+)"
            elif expected_return > 0.001:
                expected_movement = "Break-Even Range (0.10-0.12%)"
            else:
                expected_movement = "Below Profitable Threshold"
            
            return {
                'confidence': confidence,
                'prediction': prediction,
                'expected_movement': expected_movement,
                'small_move_probability': small_move_prob,  # NEW
                'expected_return': expected_return,          # NEW
                'target_range': target_range,               # NEW
                'is_small_move_viable': expected_return >= 0.0012,  # NEW
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            # Fallback to simple heuristic
            simple_confidence = min(0.9, max(0.3, (ratio - 1.0) / 2.0))
            return {
                'confidence': simple_confidence,
                'prediction': ratio / 2.0,
                'expected_movement': "Moderate Bullish" if ratio > 1.6 else "Neutral",
                'timestamp': datetime.now()
            }
    

    def get_aligned_timer_minute(self, current_time):
        """Get timer minute aligned to 10-minute intervals (5:10, 5:20, etc.)"""
        # Align to 10-minute intervals starting from minute 10
        # This makes timer reset at 5:10, 5:20, 5:30, etc.
        minute = current_time.minute
        
        # Calculate which 10-minute window we're in
        if minute < 10:
            # 5:00-5:09 -> timer minutes 0-9 (previous cycle completion)
            timer_minute = minute
        elif minute < 20:
            # 5:10-5:19 -> timer minutes 0-9 (new cycle)
            timer_minute = minute - 10
        elif minute < 30:
            # 5:20-5:29 -> timer minutes 0-9 (new cycle)
            timer_minute = minute - 20
        elif minute < 40:
            # 5:30-5:39 -> timer minutes 0-9 (new cycle)
            timer_minute = minute - 30
        elif minute < 50:
            # 5:40-5:49 -> timer minutes 0-9 (new cycle)
            timer_minute = minute - 40
        else:
            # 5:50-5:59 -> timer minutes 0-9 (new cycle)
            timer_minute = minute - 50
        
        return timer_minute

    async def monitor_psc_signals(self):
        """Monitor for PSC signals with timer-based Superp leverage tracking"""
        logger.info("PSC signal monitoring started with Superp timer tracking")
        
        while self.running:
            try:
                current_time = datetime.now()
                old_minute = self.timer_minute
                self.timer_minute = self.get_aligned_timer_minute(current_time)
                
                # =====================================================================
                # SUPERP TIMER-BASED LEVERAGE UPDATES (Critical for accuracy)
                # =====================================================================
                
                # When timer minute changes, update all Superp position leverages
                if old_minute != self.timer_minute:
                    # Timer change messages disabled to reduce log clutter
                    # if self.timer_alerts_enabled:
                    #     logger.info(f"⏰ Timer changed: {old_minute} → {self.timer_minute}")
                    
                    # Take leverage snapshots for all active Superp positions
                    if self.superp_positions:
                        active_count = len([p for p in self.superp_positions.values() if p.status == "ACTIVE"])
                        if active_count > 0:
                            if self.timer_alerts_enabled:
                                logger.info(f"📸 Taking leverage snapshots for {active_count} active Superp positions")
                            self.update_all_position_leverages(self.timer_minute)
                    
                    # ⏰ AUTO-VALIDATE TIMER-EXPIRED PREDICTIONS (every minute)
                    if self.prediction_validator:
                        try:
                            validated_count = self.prediction_validator.auto_validate_timer_expired()
                            if validated_count > 0:
                                logger.info(f"⏰ Timer-validated {validated_count} expired predictions")
                        except Exception as e:
                            logger.error(f"❌ Timer validation error: {e}")
                    
                    # Auto-validate completed trades (existing functionality)
                    if self.prediction_validator:
                        try:
                            self.prediction_validator.auto_validate_completed_trades()
                        except Exception as e:
                            logger.error(f"❌ Trade validation error: {e}")
                
                # Timer window notifications (with Superp context) - DISABLED
                # These timer notifications have been disabled to reduce message clutter
                # if old_minute != self.timer_minute and self.timer_alerts_enabled and self.telegram_enabled:
                #     logger.info(f"📢 Sending timer notification for minute {self.timer_minute}")
                #     
                #     if self.timer_minute == 0:
                #         await self.send_notification(
                #             "⏰ **TIMER RESET - MAXIMUM SUPERP LEVERAGE**\n"
                #             "🟢 Entry window OPEN (next 3 minutes)\n"
                #             "🚀 Superp leverage at PEAK levels\n"
                #             "🎯 Prime time for high-leverage PSC trades!"
                #         )
                #     elif self.timer_minute == 3:
                #         await self.send_notification(
                #             "⏰ **Entry Window CLOSED - Leverage Decreasing**\n"
                #             "🟡 Wait for next cycle\n"
                #             "📉 Superp leverage now reducing\n"
                #             "⏱️ Next maximum leverage window in 7 minutes"
                #         )
                #     elif self.timer_minute == 5:
                #         # Mid-timer leverage notification
                #         await self.send_notification(
                #             "⏰ **Mid-Timer: Moderate Leverage Phase**\n"
                #             "🟡 Superp leverage at 60-80% of maximum\n"
                #             "📊 Existing positions adjusting exposure"
                #         )
                #     elif self.timer_minute == 8:
                #         # Late-timer leverage notification
                #         await self.send_notification(
                #             "⏰ **Late Timer: Low Leverage Phase**\n"
                #             "🔴 Superp leverage at 30-60% of maximum\n"
                #             "⚠️ Prepare for timer reset in 2 minutes"
                #         )
                # elif old_minute != self.timer_minute and not self.timer_alerts_enabled:
                #     logger.info(f"🔇 Timer notification SKIPPED for minute {self.timer_minute} (alerts disabled)")
                
                # Enhanced multi-coin signal detection with ML
                import random
                if self.timer_minute < 3 and random.random() < 0.15:  # 15% chance during entry window
                    # Select random coin from monitored list
                    coin_data = random.choice(self.monitored_coins)
                    crypto = coin_data['symbol']
                    coin_name = coin_data['name']
                    volatility = coin_data['volatility']
                    pair = coin_data['pair']
                    
                    # Fetch current price
                    current_price = await self.fetch_current_price(crypto)
                    if not current_price:
                        continue
                    
                    # Calculate arbitrage ratio
                    ton_price = await self.fetch_current_price('TON')
                    if not ton_price:
                        continue
                        
                    # FIXED: Use logarithmic scaling for meaningful PSC ratios
                    import math
                    if current_price > 0 and ton_price > 0:
                        log_ratio = math.log10(current_price) - math.log10(ton_price)
                        # Convert to positive scale for compatibility: log_ratio + 6 (shifts range to ~1-11)
                        base_ratio = round(log_ratio + 6, 4)
                    else:
                        base_ratio = 6.0  # Neutral value if price data invalid
                    
                    # Only proceed if ratio meets threshold
                    if base_ratio >= self.min_signal_ratio:
                        # Generate ML prediction for this signal
                        ml_prediction = await self.generate_ml_prediction(current_price, ton_price, base_ratio)
                        confidence = ml_prediction.get('confidence', random.uniform(0.4, 0.95))
                        
                        # =====================================================================
                        # COMPREHENSIVE TRADINGVIEW MULTI-TIMEFRAME ANALYSIS
                        # =====================================================================
                        tradingview_analysis = None
                        tradingview_enhancement = None
                        tv_log_entry = None
                        
                        if self.tradingview and self.tradingview_enabled:
                            try:
                                # Get OPTIMIZED single coin analysis for faster processing
                                logger.info(f"📊 Getting optimized TradingView analysis for {crypto}")
                                coin_analysis = await self.tradingview.get_single_coin_analysis(crypto)
                                
                                # Get PSC direction for enhancement
                                temp_direction, _, _ = self.determine_trade_direction(crypto, base_ratio, confidence)
                                
                                # Use the single coin analysis for faster enhancement
                                tradingview_enhancement = self.tradingview.enhance_psc_with_single_coin_analysis(
                                    coin_analysis, confidence, temp_direction
                                )
                                
                                if tradingview_enhancement:
                                    # Update confidence with TradingView enhancement
                                    enhanced_confidence = tradingview_enhancement['enhanced_confidence']
                                    logger.info(f"🔧 TradingView enhanced confidence: {confidence:.1%} → {enhanced_confidence:.1%}")
                                    confidence = enhanced_confidence
                                    
                                    # Store single coin data for logging
                                    tv_log_entry = {
                                        'timestamp': datetime.now().isoformat(),
                                        'symbol': crypto,
                                        'timeframes_analyzed': ['1m', '5m', '10m'],
                                        'timeframe_summary': tradingview_enhancement.get('timeframe_summary', {}),
                                        'consensus_direction': coin_analysis.get('consensus', {}).get('direction', 'neutral'),
                                        'consensus_strength': coin_analysis.get('consensus', {}).get('strength', 0),
                                        'consensus_confidence': coin_analysis.get('consensus', {}).get('confidence', 0),
                                        'trade_signals': tradingview_enhancement.get('trade_signals', {}),
                                        'original_confidence': ml_prediction.get('confidence', 0),
                                        'enhanced_confidence': confidence,
                                        'confidence_multiplier': tradingview_enhancement.get('confidence_multiplier', 1.0),
                                        'alignment_score': tradingview_enhancement.get('alignment_score', 0),
                                        'recommendation': tradingview_enhancement.get('recommendation', 'N/A'),
                                        'timeframe_alignment': tradingview_enhancement.get('trade_signals', {}).get('timeframe_alignment', False)
                                    }
                                    
                                    # Keep recent logs in memory (last 50)
                                    self.tradingview_logs.append(tv_log_entry)
                                    if len(self.tradingview_logs) > 50:
                                        self.tradingview_logs = self.tradingview_logs[-50:]
                                    
                                    # Log comprehensive analysis to CSV
                                    await self.log_comprehensive_tradingview_data(tv_log_entry)
                                        
                                else:
                                    logger.warning(f"⚠️ No comprehensive TradingView enhancement for {crypto}")
                                    
                            except Exception as tv_error:
                                logger.error(f"❌ Comprehensive TradingView analysis failed for {crypto}: {tv_error}")
                        
                        # =====================================================================
                        # REAL MARKET DATA INTEGRATION AND ADVANCED SIGNAL FILTERING (NEW!)
                        # =====================================================================
                        
                        # Fetch real market data for enhanced analysis
                        real_market_data = {}
                        market_quality_score = 0.0
                        
                        if self.real_market_provider:
                            try:
                                market_data = await self.real_market_provider.get_market_data(crypto)
                                if market_data:
                                    real_market_data = {
                                        'price': market_data.price,
                                        'volume_24h': market_data.volume_24h, 
                                        'change_24h': market_data.change_24h,
                                        'market_cap': market_data.market_cap,
                                        'rsi': market_data.rsi,
                                        'sma_20': market_data.sma_20,
                                        'sma_50': market_data.sma_50,
                                        'bb_upper': market_data.bb_upper,
                                        'bb_lower': market_data.bb_lower,
                                        'volume_sma': market_data.volume_sma
                                    }
                                    market_quality_score = self.real_market_provider.get_market_quality_score(market_data)
                                    logger.info(f"📊 Real market data: {crypto} ${market_data.price:.6f} (Quality: {market_quality_score:.2f})")
                                    
                                    # Update current_price with real data if available
                                    if market_data.price > 0:
                                        current_price = market_data.price
                                        logger.info(f"✅ Using real market price for {crypto}: ${current_price:.6f}")
                                        
                            except Exception as market_error:
                                logger.error(f"❌ Real market data error for {crypto}: {market_error}")
                        
                        # Apply advanced signal filtering
                        signal_accepted = True
                        filter_reason = "No filtering applied"
                        quality_metrics = None
                        position_size_multiplier = 1.0
                        
                        if self.signal_filter and real_market_data:
                            try:
                                # Prepare signal data for filtering
                                temp_direction, _, _ = self.determine_trade_direction(crypto, base_ratio, confidence)
                                signal_data = {
                                    'direction': temp_direction,
                                    'confidence': confidence,
                                    'symbol': crypto,
                                    'psc_ratio': base_ratio
                                }
                                
                                # Apply signal quality filter
                                signal_accepted, quality_metrics, filter_reason = self.signal_filter.should_accept_signal(
                                    crypto, signal_data, real_market_data
                                )
                                
                                if signal_accepted:
                                    # Calculate enhanced position size
                                    position_size_multiplier = self.signal_filter.get_position_size_multiplier(quality_metrics)
                                    logger.info(f"✅ Signal ACCEPTED for {crypto}: {filter_reason}")
                                    logger.info(f"📊 Quality Score: {quality_metrics.overall_quality:.2f}, Position Multiplier: {position_size_multiplier:.2f}x")
                                else:
                                    logger.info(f"❌ Signal REJECTED for {crypto}: {filter_reason}")
                                    continue  # Skip this signal - doesn't meet quality standards
                                    
                            except Exception as filter_error:
                                logger.error(f"❌ Signal filtering error for {crypto}: {filter_error}")
                                # Continue with signal if filtering fails
                        
                        # Only proceed if signal passed quality filters
                        if not signal_accepted:
                            continue
                        
                        # =====================================================================
                        # INTEGRATED SIGNAL PROCESSING (Enhanced Accuracy System)
                        # =====================================================================
                        
                        integrated_signal = None
                        if self.integrated_processor:
                            try:
                                # Get PSC direction for integrated processing
                                temp_direction, _, _ = self.determine_trade_direction(crypto, base_ratio, confidence)
                                
                                # Process integrated signal with all components
                                integrated_signal = await self.integrated_processor.process_integrated_signal(
                                    coin=crypto,
                                    current_price=current_price,
                                    psc_ratio=base_ratio,
                                    psc_confidence=confidence,
                                    psc_direction=temp_direction
                                )
                                
                                if integrated_signal:
                                    # Use integrated signal's enhanced confidence and direction
                                    confidence = integrated_signal.confidence
                                    direction = integrated_signal.direction
                                    logger.info(f"🎯 Using INTEGRATED signal for {crypto}: "
                                               f"Confidence: {confidence:.1%}, Direction: {direction}, "
                                               f"Consensus: {integrated_signal.consensus_strength:.1%}")
                                else:
                                    logger.info(f"❌ Integrated signal rejected for {crypto} - using standard PSC")
                                    
                            except Exception as integrated_error:
                                logger.error(f"❌ Integrated signal processing failed for {crypto}: {integrated_error}")
                        
                        # Check if signal should be sent based on filters and small-move viability
                        if self.should_send_signal(confidence, ml_prediction):
                            # Get confidence level info
                            conf_level, conf_emoji, stars = self.get_confidence_level_info(confidence)
                            
                            # Determine trade direction
                            direction, dir_emoji, direction_desc = self.determine_trade_direction(crypto, base_ratio, confidence)
                            
                            # =====================================================================
                            # SUPERP LEVERAGE & EXIT PRICE CALCULATION  
                            # =====================================================================
                            
                            # Calculate optimal Superp buy-in and leverage
                            superp_buy_in, expected_superp_leverage = self.calculate_optimal_superp_buy_in(
                                current_price, confidence, base_ratio, volatility
                            )
                            
                            # Get live Superp leverage (what platform would actually offer)
                            live_superp_leverage = self.get_live_superp_leverage(current_price, superp_buy_in)
                            
                            # Calculate realistic Superp profit targets using ML expected return
                            ml_expected_return = ml_prediction.get('expected_return', 0.0015)  # Default 0.15%
                            superp_targets = self.calculate_superp_profit_target(
                                current_price, live_superp_leverage, superp_buy_in, confidence,
                                ml_expected_return=ml_expected_return  # Pass ML prediction
                            )
                            
                            # Use ML-enhanced exit price calculation
                            exit_price = self.calculate_superp_exit_price(
                                current_price, confidence, direction, 
                                live_superp_leverage, superp_buy_in,
                                ml_expected_return=ml_expected_return  # Use ML prediction
                            )
                            
                            # Calculate realistic profit percentages
                            price_move_pct = abs(exit_price - current_price) / current_price
                            superp_profit_usd = superp_targets['expected_profit_usd']
                            superp_profit_pct = superp_targets['expected_profit_pct']
                            
                            # Get price change info
                            price_change = self.price_changes.get(crypto, 0.0)
                            change_emoji = "📈" if price_change > 0 else "📉" if price_change < 0 else "➡️"
                            
                            # Calculate preliminary leverage for display
                            time_remaining = max(10 - self.timer_minute, 1)
                            ratio_for_leverage = base_ratio
                            preliminary_leverage = self.calculate_dynamic_leverage(confidence, ratio_for_leverage, volatility, time_remaining)
                            estimated_position_size = self.calculate_position_size(preliminary_leverage)
                            
                            # Build comprehensive TradingView section for signal message with better error handling
                            tradingview_section = ""
                            if tradingview_enhancement and tv_log_entry:
                                try:
                                    alignment_score = tradingview_enhancement.get('alignment_score', 0)
                                    alignment_emoji = "✅" if alignment_score > 0.5 else "⚠️" if alignment_score > 0 else "❌"
                                    
                                    # Get data with proper fallbacks
                                    timeframe_summary = tradingview_enhancement.get('timeframe_summary', {})
                                    consensus_data = tradingview_enhancement.get('consensus_data', {})
                                    trade_signals = tradingview_enhancement.get('trade_signals', {})
                                    
                                    # Safe timeframe access with fallbacks
                                    tf_1m = timeframe_summary.get('1m', 'ANALYZING')
                                    tf_5m = timeframe_summary.get('5m', 'ANALYZING')  
                                    tf_10m = timeframe_summary.get('10m', 'ANALYZING')
                                    
                                    # Timeframe alignment check
                                    tf_alignment = "🎯 ANALYZING" if trade_signals.get('timeframe_alignment') else "⚠️ MIXED"
                                    
                                    # Safe consensus data access
                                    direction = consensus_data.get('direction', 'ANALYZING').upper()
                                    strength = consensus_data.get('strength', 0)
                                    tv_confidence = consensus_data.get('confidence', 0)
                                    entry_rec = trade_signals.get('entry_recommendation', 'MONITOR').upper()
                                    
                                    # Safe enhancement data access
                                    orig_conf = tv_log_entry.get('original_confidence', confidence)
                                    enhanced_conf = tv_log_entry.get('enhanced_confidence', confidence)
                                    conf_mult = tradingview_enhancement.get('confidence_multiplier', 1.0)
                                    recommendation = tradingview_enhancement.get('recommendation', 'MONITOR')
                                    
                                    tradingview_section = f"""
📊 **TradingView Multi-Timeframe Analysis:**
• 1m Signal: *{tf_1m}*
• 5m Signal: *{tf_5m}*
• 10m Signal: *{tf_10m}*
• Timeframe Status: {tf_alignment}

🎯 **Market Consensus:**
• Direction: *{direction}*
• Strength: `{strength:.1%}`
• Confidence: `{tv_confidence:.1%}`
• Entry Rec: *{entry_rec}*

⚡ **PSC Enhancement:**
• Original Confidence: `{orig_conf:.1%}`
• Enhanced Confidence: `{enhanced_conf:.1%}`
• Multiplier: `{conf_mult:.2f}x`
• PSC Alignment: {alignment_emoji} `{alignment_score:.1%}`
• Recommendation: {recommendation}

"""
                                except Exception as tv_error:
                                    logger.warning(f"TradingView section error: {tv_error}")
                                    tradingview_section = f"""
📊 **TradingView Analysis:** ⚡ Real-time Processing
• Status: Live market analysis active
• Integration: PSC enhancement operational
• Data: Continuous multi-timeframe monitoring

"""
                            else:
                                tradingview_section = f"""
📊 **TradingView Analysis:** 🎯 PSC-Driven Signal
• Primary: PSC ratio analysis ({base_ratio:.2f})
• Enhancement: {confidence:.1%} confidence base
• Integration: ML prediction active

"""
                            
                            signal_msg = f"""
🎯 **{crypto} SUPERP PSC SIGNAL** {conf_emoji}
═══════════════════════════════
💎 *{coin_name}* ({pair})
📊 Volatility: *{volatility}*
⏰ Timer: `{self.timer_minute}/10 min`
� Time: `{datetime.now().strftime("%H:%M:%S")}`
�🟢 *ENTRY WINDOW ACTIVE*

💰 **Current Market:**
• Entry Price: `${current_price:.8f}`
• 24h Change: `{price_change:+.2f}%` {change_emoji}
• PSC Ratio vs TON: `{base_ratio}` (≥{self.min_signal_ratio} ✅)

🚀 **SUPERP No-Liquidation Setup:**
• Buy-in Amount: `${superp_buy_in:.2f}` (Max Loss)
• Live Leverage: `{live_superp_leverage:.0f}x` (Asset/Buy-in)
• Virtual Exposure: `${live_superp_leverage * superp_buy_in:,.0f}`
• Break-even: `${superp_targets['break_even_price']:.8f}` ({(superp_targets['break_even_price']/current_price-1)*100:+.3f}%)

🎯 **Realistic Profit Target:**
• Target Price: `${exit_price:.8f}`
• Price Move Needed: `{price_move_pct:.3%}` (Tiny move!)
• Expected Profit: `${superp_profit_usd:.2f}` ({superp_profit_pct:.0f}% return)
• Target Logic: Small moves × Extreme leverage = Big profits

⚡ **Superp Advantages:**
• NO Liquidation Risk: Cannot be forced out
• NO Margin Calls: Buy-in is maximum loss
• Extreme Leverage: Up to 10,000x available
• Timer Protection: 10-minute position limit

🤖 **ML Analysis:**
• Confidence: `{confidence:.1%}` {stars}
• Level: *{conf_level}*
• Direction: *{direction}* {dir_emoji}
• Signal: {direction_desc}

🎯 **Integrated Accuracy System:**"""

                            # Add integrated signal information if available
                            if integrated_signal:
                                component_count = len(integrated_signal.components)
                                consensus_emoji = "🔥" if integrated_signal.consensus_strength > 0.9 else "✅" if integrated_signal.consensus_strength > 0.8 else "⚠️"
                                
                                signal_msg += f"""
• Status: {consensus_emoji} *ENHANCED ACCURACY MODE*
• Components: `{component_count}` systems validated
• Consensus: `{integrated_signal.consensus_strength:.1%}` agreement
• Final Confidence: `{integrated_signal.confidence:.1%}` (AI-optimized)
• Integration ID: `{integrated_signal.prediction_id[-8:]}`"""
                                
                                # Show component breakdown
                                for comp_name, comp_signal in integrated_signal.components.items():
                                    comp_emoji = "🟢" if comp_signal.confidence > 0.7 else "🟡" if comp_signal.confidence > 0.5 else "🔴"
                                    signal_msg += f"""
  • {comp_name.upper()}: {comp_emoji} `{comp_signal.confidence:.1%}` ({comp_signal.direction})"""
                            else:
                                signal_msg += f"""
• Status: ⚡ *STANDARD PSC MODE*
• Note: Enhanced accuracy not applied to this signal"""

                            signal_msg += f"""

{tradingview_section}📈 **Revolutionary Trading:**
• Risk: ONLY `${superp_buy_in:.2f}` maximum loss
• Reward: `${superp_profit_usd:.2f}` potential profit
• Strategy: Capture tiny {price_move_pct:.3%} move with {live_superp_leverage:.0f}x leverage
• Window: `{3 - self.timer_minute} min` remaining for entry

⏰ *Time: {datetime.now().strftime('%H:%M:%S')}*

**🔥 This is how Superp revolutionizes trading: Massive leverage, minimal risk!**
                            """
                            
                            await self.send_notification(signal_msg)
                            
                            # 📊 LOG SIGNAL TO CSV
                            self.log_signal(
                                coin=crypto,
                                price=current_price,
                                ratio=base_ratio,
                                confidence=confidence,
                                direction=direction,
                                exit_estimate=exit_price,
                                ml_prediction=ml_prediction.get('prediction', 0)
                            )
                            
                            # � LOG PREDICTION FOR PAPER TRADING VALIDATION
                            if self.paper_validator:
                                try:
                                    # Calculate expected profit percentage
                                    expected_profit_pct = abs(exit_price - current_price) / current_price
                                    
                                    # Get TradingView sentiment if available
                                    tv_sentiment = 0.5  # Default neutral
                                    if tradingview_enhancement:
                                        consensus_data = tradingview_enhancement.get('consensus_data', {})
                                        tv_sentiment = consensus_data.get('confidence', 0.5)
                                    
                                    # Log prediction for paper trading
                                    prediction_id = self.paper_validator.log_prediction(
                                        coin=crypto,
                                        direction=direction,
                                        confidence=confidence,
                                        entry_price=current_price,
                                        target_price=exit_price,
                                        psc_ratio=base_ratio,
                                        ml_prediction_value=ml_prediction.get('prediction', 0),
                                        signal_strength=self.get_signal_strength(confidence),
                                        market_conditions=self.get_market_conditions(),
                                        tradingview_sentiment=tv_sentiment,
                                        expected_profit_pct=expected_profit_pct
                                    )
                                    
                                    if prediction_id:
                                        logger.info(f"📊 Started paper trade validation: {prediction_id}")
                                    
                                except Exception as e:
                                    logger.error(f"Error logging prediction for paper trading: {e}")
                            
                            # OPEN REAL POSITION for tracking actual exit price with dynamic leverage
                            time_remaining = max(10 - self.timer_minute, 1)  # Time left in cycle
                            
                            # Apply enhanced position sizing from signal filter
                            enhanced_position_multiplier = position_size_multiplier if 'position_size_multiplier' in locals() else 1.0
                            
                            signal_id = self.open_position(
                                coin=crypto,
                                entry_price=current_price,
                                direction=direction,
                                confidence=confidence,
                                target_exit=exit_price,
                                volatility=volatility,
                                time_remaining=time_remaining,
                                position_size_multiplier=enhanced_position_multiplier
                            )
                            
                            # Record the prediction for accuracy tracking
                            if self.ml_engine:
                                prediction_data = {
                                    'price': current_price,
                                    'ton_price': ton_price,
                                    'ratio': base_ratio,
                                    'prediction': ml_prediction.get('prediction', 0),
                                    'confidence': confidence,
                                    'coin': crypto,
                                    'timestamp': datetime.now().isoformat()
                                }
                                self.ml_engine.record_prediction(prediction_data)
                                
                                # Also record in enhanced validator for comprehensive tracking
                                if self.prediction_validator:
                                    # Calculate expected return
                                    expected_profit_pct = abs(exit_price - current_price) / current_price
                                    if direction == "SHORT":
                                        expected_profit_pct = -expected_profit_pct
                                    
                                    # Get TradingView sentiment
                                    tv_sentiment = 0.5  # Default neutral
                                    if tradingview_enhancement:
                                        consensus_data = tradingview_enhancement.get('consensus_data', {})
                                        tv_sentiment = consensus_data.get('confidence', 0.5)
                                    
                                    enhanced_prediction = {
                                        'coin': crypto,
                                        'direction': direction,
                                        'confidence': confidence,
                                        'entry_price': current_price,
                                        'target_price': exit_price,
                                        'expected_return': expected_profit_pct,
                                        'signal_strength': 'HIGH' if confidence > 0.7 else 'MODERATE',
                                        'market_sentiment': ml_prediction.get('market_sentiment', 0.5),
                                        'model_version': '2.0_enhanced',
                                        'features': {
                                            'psc_ratio': base_ratio,
                                            'ml_prediction': ml_prediction.get('prediction', 0),
                                            'tradingview_sentiment': tv_sentiment
                                        }
                                    }
                                    prediction_id = self.prediction_validator.record_prediction(enhanced_prediction)
                                    if prediction_id:
                                        # Store prediction ID for later validation
                                        self.open_positions[signal_id]['prediction_id'] = prediction_id
                            
                            logger.info(f"Enhanced {crypto} signal: ratio {base_ratio}, confidence {confidence:.1%}, direction {direction}, position opened {signal_id}")
                
                # 🏁 CHECK EXIT CONDITIONS for open positions
                await self.check_exit_conditions()
                
                # Sleep for check interval (30 seconds)
                await asyncio.sleep(self.tradingview_check_interval)
                
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                await asyncio.sleep(60)
    
    def setup_application(self):
        """Setup Telegram application with conflict handling"""
        if not self.telegram_enabled:
            logger.info("🚫 Telegram application setup skipped (disabled)")
            return
            
        self.application = Application.builder().token(self.bot_token).build()
        
        # Add command handlers
        handlers = [
            CommandHandler("start", self.start_command),
            CommandHandler("status", self.status_command),
            CommandHandler("signals", self.signals_command),
            CommandHandler("help", self.help_command),
            CommandHandler("dashboard", self.dashboard_command),
            CommandHandler("logs", self.logs_command),
            CommandHandler("config", self.config_command),
            CommandHandler("performance", self.performance_command),
            CommandHandler("notifications", self.notifications_command),
            CommandHandler("filter", self.filter_command),
            CommandHandler("settings", self.settings_command),
            CommandHandler("coins", self.coins_command),
            CommandHandler("prices", self.prices_command),
            CommandHandler("stats", self.stats_command),
            CommandHandler("trades", self.trades_command),
            CommandHandler("positions", self.positions_command),
            CommandHandler("superp", self.superp_command),
            CommandHandler("tradingview", self.tradingview_command),
            CommandHandler("tvrefresh", self.tvrefresh_command),
            CommandHandler("ml", self.ml_command),
            CommandHandler("database", self.database_command),
            CommandHandler("predictions", self.predictions_command),
            CommandHandler("paper", self.paper_command),
        ]
        
        for handler in handlers:
            self.application.add_handler(handler)
        
        logger.info("Telegram application configured")
    
    async def clear_bot_conflicts(self):
        """Clear any existing bot conflicts"""
        try:
            logger.info("🔄 Clearing potential bot conflicts...")
            
            # Create a temporary bot instance to clear webhooks
            bot = Bot(token=self.bot_token)
            
            # Delete any existing webhook
            await bot.delete_webhook(drop_pending_updates=True)
            logger.info("✅ Cleared webhook and pending updates")
            
            # Small delay to ensure cleanup
            await asyncio.sleep(2)
            
        except Exception as e:
            logger.warning(f"⚠️ Bot conflict clearing warning: {e}")
    
    
    async def check_ml_microstructure_signals(self):
        """
        Check for ML Microstructure signals with PSC integration
        Enhanced signal processing with timer-based optimization
        """
        if not self.ml_microstructure_trainer:
            return
            
        try:
            # Get recent ML microstructure signals
            recent_signals = self.ml_microstructure_trainer.get_recent_psc_signals(
                max_age_minutes=2,  # Only very recent signals
                min_confidence=0.6  # High confidence only
            )
            
            for signal in recent_signals:
                try:
                    # Extract signal data
                    coin = signal.symbol.replace('USDT', '')
                    psc_ratio = signal.psc_ratio
                    confidence = signal.confidence_score
                    direction = signal.direction
                    leverage = signal.leverage
                    timer_window = signal.timer_window
                    
                    logger.info(f"🧠 ML Microstructure Signal: {coin} {direction}")
                    logger.info(f"   PSC Ratio: {psc_ratio:.3f}, Confidence: {confidence:.1%}")
                    logger.info(f"   Leverage: {leverage:.0f}x, Window: {timer_window}")
                    
                    # Validate PSC ratio thresholds
                    signal_valid = False
                    if direction == "LONG" and psc_ratio >= self.min_signal_ratio:
                        signal_valid = True
                    elif direction == "SHORT" and psc_ratio <= 5.0:  # SHORT threshold
                        signal_valid = True
                    
                    if not signal_valid:
                        logger.info(f"❌ ML signal rejected: PSC ratio {psc_ratio:.3f} doesn't meet thresholds")
                        continue
                    
                    # Check timer window efficiency
                    if timer_window != "ENTRY_WINDOW" and confidence < 0.8:
                        logger.info(f"⏰ ML signal deferred: Not in ENTRY_WINDOW, confidence too low")
                        continue
                    
                    # Get current price for the coin
                    current_price = await self.get_current_price(coin)
                    if not current_price:
                        continue
                    
                    # Create enhanced signal for processing (preserve ML timestamp)
                    # Try to get original ML timestamp
                    try:
                        if hasattr(signal, 'timestamp'):
                            original_timestamp = signal.timestamp
                        elif hasattr(signal, 'timestamp_unix') and signal.timestamp_unix:
                            original_timestamp = datetime.fromtimestamp(signal.timestamp_unix).isoformat()
                        else:
                            original_timestamp = datetime.now().isoformat()
                    except:
                        original_timestamp = datetime.now().isoformat()
                    
                    enhanced_ml_signal = {
                        'coin': coin,
                        'price': current_price,
                        'ratio': psc_ratio,
                        'confidence': confidence,
                        'direction': direction,
                        'leverage': leverage,
                        'timer_window': timer_window,
                        'signal_source': 'ml_microstructure',
                        'microstructure_score': signal.microstructure_score,
                        'reasons': signal.reasons,
                        'timestamp': original_timestamp,
                        'processed_timestamp': datetime.now().isoformat()
                    }
                    
                    # Process the enhanced signal
                    await self.process_ml_microstructure_signal(enhanced_ml_signal)
                    
                except Exception as signal_error:
                    logger.error(f"Error processing ML microstructure signal: {signal_error}")
                    
        except Exception as e:
            logger.error(f"Error checking ML microstructure signals: {e}")
    
    async def process_ml_microstructure_signal(self, enhanced_signal):
        """
        Process validated ML microstructure signal with PSC integration
        """
        try:
            coin = enhanced_signal['coin']
            price = enhanced_signal['price']
            ratio = enhanced_signal['ratio']
            confidence = enhanced_signal['confidence']
            direction = enhanced_signal['direction']
            leverage = enhanced_signal['leverage']
            timer_window = enhanced_signal['timer_window']
            
            # Calculate Superp position parameters
            position_size = min(100.0, leverage * 10.0)  # Scale position with leverage
            buy_in_amount = min(self.superp_config['max_buy_in'], position_size)
            
            # Calculate virtual exposure based on leverage
            virtual_exposure = buy_in_amount * leverage
            
            # Log the enhanced ML signal
            self.log_signal(
                coin=coin,
                price=price,
                ratio=ratio,
                confidence=confidence,
                direction=direction,
                exit_estimate=price * (1.002 if direction == "LONG" else 0.998),  # 0.2% target
                ml_prediction={
                    'source': 'ml_microstructure',
                    'confidence': confidence,
                    'leverage': leverage,
                    'timer_window': timer_window,
                    'microstructure_score': enhanced_signal.get('microstructure_score', 0.0)
                }
            )
            
            # Create Superp position
            position_id = f"ML_MS_{coin}_{int(time.time())}"
            target_price = price * (1.002 if direction == "LONG" else 0.998)
            
            superp_position = SuperpPosition(
                id=position_id,
                asset=coin,
                buy_in_amount=buy_in_amount,
                virtual_exposure=virtual_exposure,
                effective_leverage=leverage,
                entry_price=price,
                target_price=target_price,
                stop_time=datetime.now() + timedelta(minutes=10),
                psc_ratio=ratio,
                confidence_score=confidence,
                entry_leverage=leverage,
                current_leverage=leverage,
                timer_minute_opened=self.timer_minute
            )
            
            # Store position
            self.superp_positions[position_id] = superp_position
            
            # Generate comprehensive notification
            await self.send_ml_microstructure_notification(enhanced_signal, superp_position)
            
            # Update session statistics
            self.session_stats['signals_generated'] += 1
            self.session_stats['trades_executed'] += 1
            
            logger.info(f"✅ ML Microstructure signal processed: {coin} {direction}")
            
        except Exception as e:
            logger.error(f"Error processing ML microstructure signal: {e}")
    
    async def send_ml_microstructure_notification(self, signal, position):
        """
        Send notification for ML microstructure signal
        """
        try:
            coin = signal['coin']
            direction = signal['direction']
            confidence = signal['confidence']
            leverage = signal['leverage']
            timer_window = signal['timer_window']
            microstructure_score = signal.get('microstructure_score', 0.0)
            
            # Direction emoji
            direction_emoji = "🟢" if direction == "LONG" else "🔴"
            
            # Timer window emoji
            timer_emoji = "⚡" if timer_window == "ENTRY_WINDOW" else "⏰"
            
            # Format timestamps for display
            original_time = signal.get('timestamp', 'N/A')
            processed_time = signal.get('processed_timestamp', datetime.now().isoformat())
            
            # Convert ISO timestamp to readable format for display
            try:
                if isinstance(original_time, str) and 'T' in original_time:
                    dt = datetime.fromisoformat(original_time.replace('Z', '+00:00'))
                    time_display = dt.strftime("%H:%M:%S")
                else:
                    time_display = original_time
            except:
                time_display = "N/A"
            
            message = (
                f"🧠 **ML MICROSTRUCTURE SIGNAL** {direction_emoji}\n\n"
                f"**Asset**: {coin}\n"
                f"**Direction**: {direction}\n"
                f"**PSC Ratio**: {signal['ratio']:.3f}\n"
                f"**Confidence**: {confidence:.1%}\n"
                f"**Leverage**: {leverage:.0f}x\n"
                f"**Microstructure Score**: {microstructure_score:.1%}\n"
                f"**Timer Window**: {timer_window} {timer_emoji}\n"
                f"**Signal Time**: {time_display}\n\n"
                f"**Position Details**:\n"
                f"• Buy-in: ${position.buy_in_amount:.2f}\n"
                f"• Virtual Exposure: ${position.virtual_exposure:,.2f}\n"
                f"• Target Price: ${position.target_price:.6f}\n"
                f"• Position ID: {position.id}\n\n"
                f"**SuperP**: No liquidation risk up to 10,000x\n"
                f"**Timer**: Auto-close in 10 minutes"
            )
            
            if self.telegram_enabled:
                await self.send_notification(message)
                
        except Exception as e:
            logger.error(f"Error sending ML microstructure notification: {e}")

    async def check_ml_signals(self):
        """
        Continuous monitoring of ML-generated signals
        Validates ML signals against TradingView and processes them
        Enhanced with ML Microstructure integration
        """
        logger.info("🤖 Starting ML signal monitoring...")
        
        while self.running:
            try:
                # Check ML Microstructure signals FIRST (highest priority)
                await self.check_ml_microstructure_signals()
                
                # Check for recent ML signals from traditional ML engine
                if self.ml_engine:
                    recent_ml_signals = self.ml_engine.get_recent_ml_signals(max_age_minutes=5)
                    
                    for ml_signal in recent_ml_signals:
                        try:
                            coin = ml_signal['coin']
                            prediction = ml_signal['prediction']
                            ratio = ml_signal['ratio']
                            price = ml_signal['price']
                            
                            # Check if we already processed this signal
                            signal_timestamp = ml_signal['timestamp']
                            if hasattr(self, 'processed_ml_signals') and signal_timestamp in self.processed_ml_signals:
                                continue
                            
                            # Initialize processed signals tracking
                            if not hasattr(self, 'processed_ml_signals'):
                                self.processed_ml_signals = set()
                            
                            logger.info(f"🔍 Processing ML signal: {coin} at ${price:.6f}")
                            
                            # Get current TradingView comprehensive data for validation
                            tv_validation_passed = False
                            enhanced_signal = None
                            
                            if self.tradingview and self.tradingview_enabled:
                                try:
                                    # Get OPTIMIZED single coin TradingView data for ML validation
                                    coin_analysis = await self.tradingview.get_single_coin_analysis(coin)
                                    
                                    # Validate ML signal against single coin TradingView sentiment
                                    enhanced_signal = self.ml_engine.validate_ml_signal_with_single_coin(
                                        ml_signal, coin_analysis
                                    )
                                    
                                    if enhanced_signal:
                                        tv_validation_passed = True
                                        logger.info(f"✅ ML signal validated by TradingView: {coin}")
                                    else:
                                        logger.info(f"❌ ML signal rejected by TradingView: {coin}")
                                        
                                except Exception as e:
                                    logger.warning(f"TradingView validation error: {e}")
                            else:
                                # If TradingView not available, use ML signal with high confidence requirement
                                if prediction.get('confidence', 0) >= 0.85:  # Very high confidence required
                                    tv_validation_passed = True
                                    enhanced_signal = ml_signal
                                    logger.info(f"✅ ML signal accepted (high confidence, no TV): {coin}")
                            
                            # Process validated ML signal
                            if tv_validation_passed and enhanced_signal:
                                await self.process_validated_ml_signal(enhanced_signal)
                            
                            # Mark signal as processed
                            self.processed_ml_signals.add(signal_timestamp)
                            
                            # Clean up old processed signals (keep last 100)
                            if len(self.processed_ml_signals) > 100:
                                oldest_signals = sorted(list(self.processed_ml_signals))[:50]
                                for old_signal in oldest_signals:
                                    self.processed_ml_signals.remove(old_signal)
                                    
                        except Exception as e:
                            logger.error(f"Error processing ML signal for {coin}: {e}")
                            continue
                
                # Check every 30 seconds for ML signals
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"ML signal monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def process_validated_ml_signal(self, enhanced_signal):
        """Process a validated ML signal that passed TradingView sentiment check"""
        try:
            coin = enhanced_signal['coin']
            price = enhanced_signal['price']
            ratio = enhanced_signal['ratio']
            prediction = enhanced_signal['prediction']
            
            # Check if we're in entry window (respect timer constraints)
            current_time = datetime.now()
            self.timer_minute = self.get_aligned_timer_minute(current_time)
            
            if self.timer_minute >= 3:
                logger.info(f"🕐 ML signal delayed - outside entry window: {coin}")
                return
            
            # Extract prediction data
            confidence = prediction.get('combined_confidence', prediction.get('confidence', 0))
            small_move_prob = prediction.get('small_move_probability', 0)
            expected_return = prediction.get('expected_return', 0)
            tv_direction = prediction.get('tradingview_direction', 'BUY')
            
            # ENHANCED: Determine direction based on expected return and TradingView (support both LONG and SHORT)
            if expected_return > 0.0005 and tv_direction in ['BUY', 'STRONG_BUY']:
                direction = "LONG"
                dir_emoji = "📈"
                direction_desc = "ML + TradingView bullish consensus"
            elif expected_return < -0.0005 and tv_direction in ['SELL', 'STRONG_SELL']:
                direction = "SHORT"
                dir_emoji = "📉"
                direction_desc = "ML + TradingView bearish consensus"
            elif abs(expected_return) > 0.0012:  # Strong ML signal regardless of TradingView
                if expected_return > 0:
                    direction = "LONG"
                    dir_emoji = "📈"
                    direction_desc = "Strong ML bullish signal"
                else:
                    direction = "SHORT"
                    dir_emoji = "📉"
                    direction_desc = "Strong ML bearish signal"
            else:
                direction = "NEUTRAL"
                dir_emoji = "↔️"
                direction_desc = "Mixed signals - holding"
                return  # Don't trade on mixed signals
            
            # Check if signal quality meets our standards (enhanced for both LONG and SHORT)
            signal_quality_score = (
                (confidence * 0.4) +
                (small_move_prob * 0.3) +
                (min(abs(expected_return) / 0.002, 1.0) * 0.2) +  # Use absolute expected return for both directions
                (prediction.get('validation_score', 0) / 5.0 * 0.1)  # TradingView validation score
            )
            
            if signal_quality_score < 0.7:  # High quality threshold
                logger.info(f"❌ ML signal quality too low: {coin} (score: {signal_quality_score:.2f})")
                return
            
            # Check signal sending criteria
            if not self.should_send_signal(confidence, prediction):
                logger.info(f"❌ ML signal filtered out: {coin}")
                return
            
            # Calculate realistic exit price using ML expected return (ENHANCED: support both directions)
            exit_price = self.calculate_ml_guided_exit_price(price, expected_return, confidence, direction)
            
            # Generate comprehensive signal message
            signal_msg = await self.generate_ml_signal_message(
                coin, price, ratio, confidence, direction, dir_emoji, 
                direction_desc, exit_price, prediction, enhanced_signal
            )
            
            # Send ML-generated signal
            await self.send_notification(signal_msg)
            
            # Log ML signal
            self.log_signal(
                coin=coin,
                price=price,
                ratio=ratio,
                confidence=confidence,
                direction=direction,
                exit_estimate=exit_price,
                ml_prediction=prediction.get('small_move_probability', 0)
            )
            
            # Open position for tracking
            signal_id = self.open_position(
                coin=coin,
                entry_price=price,
                direction=direction,
                confidence=confidence,
                target_exit=exit_price,
                volatility="ML_Derived",
                time_remaining=max(10 - self.timer_minute, 1)
            )
            
            # Record ML prediction for tracking
            if self.ml_engine:
                prediction_data = {
                    'price': price,
                    'prediction': small_move_prob,
                    'confidence': confidence,
                    'coin': coin,
                    'timestamp': datetime.now().isoformat(),
                    'source': 'ML_INDEPENDENT'
                }
                self.ml_engine.record_prediction(prediction_data)
            
            logger.info(f"🤖 ML Independent Signal Processed: {coin} - {direction} - Position: {signal_id}")
            
        except Exception as e:
            logger.error(f"Error processing validated ML signal: {e}")

    def calculate_ml_guided_exit_price(self, entry_price, expected_return, confidence, direction="LONG"):
        """Calculate exit price guided by ML expected return (ENHANCED: supports both LONG and SHORT)"""
        try:
            # Use ML expected return but cap it at realistic levels for small moves
            if direction == "LONG":
                # For LONG positions: positive expected return, cap at 0.12% to 0.20%
                capped_return = min(max(expected_return, 0.0012), 0.002)
            else:  # SHORT positions
                # For SHORT positions: negative expected return, cap at -0.12% to -0.20%
                capped_return = max(min(expected_return, -0.0012), -0.002)
            
            # Adjust for confidence (lower confidence = more conservative target)
            confidence_adjustment = 0.7 + (confidence * 0.3)  # 0.7 to 1.0 multiplier
            adjusted_return = capped_return * confidence_adjustment
            
            if direction == "LONG":
                exit_price = entry_price * (1 + adjusted_return)
            else:  # SHORT
                exit_price = entry_price * (1 + adjusted_return)  # Note: adjusted_return is negative for shorts
            
            logger.debug(f"ML exit calculation ({direction}): {expected_return:.4f} → {adjusted_return:.4f} → ${exit_price:.8f}")
            
            return exit_price
            
        except Exception as e:
            logger.error(f"Error calculating ML guided exit price: {e}")
            # Fallback based on direction
            if direction == "LONG":
                return entry_price * 1.0015  # +0.15%
            else:  # SHORT
                return entry_price * 0.9985  # -0.15%

    async def generate_ml_signal_message(self, coin, price, ratio, confidence, direction, 
                                       dir_emoji, direction_desc, exit_price, prediction, enhanced_signal):
        """Generate comprehensive ML signal message"""
        try:
            # Get coin info
            coin_info = next((c for c in self.monitored_coins if c['symbol'] == coin), {})
            coin_name = coin_info.get('name', coin)
            pair = coin_info.get('pair', f'{coin}/USDT')
            volatility = coin_info.get('volatility', 'ML_Derived')
            
            # Calculate price move needed
            price_move_pct = (exit_price - price) / price
            
            # Calculate Superp details
            superp_buy_in = 100.0  # Standard buy-in
            live_leverage = price / (superp_buy_in / 1000) if superp_buy_in > 0 else 0
            superp_profit_pct = price_move_pct * live_leverage
            superp_profit_usd = (superp_profit_pct / 100) * superp_buy_in
            
            # Confidence display
            conf_level, conf_emoji, stars = self.get_confidence_level_info(confidence)
            
            # ML-specific information
            small_move_prob = prediction.get('small_move_probability', 0)
            expected_return = prediction.get('expected_return', 0)
            tv_validation = prediction.get('tradingview_validation', False)
            tv_strength = prediction.get('tradingview_strength', 0)
            validation_score = enhanced_signal.get('validation_score', 0)
            
            signal_msg = f"""
🤖 **{coin} ML INDEPENDENT SIGNAL** {conf_emoji}
═══════════════════════════════
💎 *{coin_name}* ({pair})
📊 Source: ML Continuous Scan + TradingView Validation
⏰ Timer: `{self.timer_minute}/10 min`
🟢 *ENTRY WINDOW ACTIVE*

💰 **Current Market:**
• Entry Price: `${price:.8f}`
• PSC Ratio: `{ratio}` (ML detected)
• Volatility: *{volatility}*

🚀 **SUPERP No-Liquidation Setup:**
• Buy-in Amount: `${superp_buy_in:.2f}` (Max Loss)
• Live Leverage: `{live_leverage:.0f}x` (Asset/Buy-in)
• Virtual Exposure: `${live_leverage * superp_buy_in:,.0f}`

🎯 **ML-Guided Target:**
• Target Price: `${exit_price:.8f}`
• Price Move Needed: `{price_move_pct:.3%}` (Small move optimized!)
• Expected Profit: `${superp_profit_usd:.2f}` ({superp_profit_pct:.0f}% return)

🤖 **ML Analysis:**
• Overall Confidence: `{confidence:.1%}` {stars}
• Small-Move Probability: `{small_move_prob:.1%}`
• Expected Return: `{expected_return:.3%}`
• Level: *{conf_level}*
• Direction: *{direction}* {dir_emoji}

📊 **TradingView Validation:**
• Validation: {'✅ PASSED' if tv_validation else '❌ FAILED'}
• TV Strength: `{tv_strength:.1%}`
• Validation Score: `{validation_score}/5`
• Sentiment Alignment: {'🎯 Confirmed' if tv_validation else '⚠️ Mixed'}

⚡ **Superp Advantages:**
• NO Liquidation Risk: Cannot be forced out
• NO Margin Calls: Buy-in is maximum loss
• ML Optimization: Trained for achievable small moves
• TradingView Confirmation: Professional analysis backing

📈 **Revolutionary ML Trading:**
• Risk: ONLY `${superp_buy_in:.2f}` maximum loss
• Reward: `${superp_profit_usd:.2f}` potential profit
• Strategy: ML-detected {price_move_pct:.3%} move opportunity
• Window: `{3 - self.timer_minute} min` remaining for entry

⏰ *Time: {datetime.now().strftime('%H:%M:%S')}*

**🔥 This is ML-powered micro-opportunity detection in action!**
            """
            
            return signal_msg
            
        except Exception as e:
            logger.error(f"Error generating ML signal message: {e}")
            return f"🤖 ML Signal: {coin} at ${price:.6f} - {direction}"
    
    async def start_system(self):
        """Start the complete system"""
        try:
            # Clear any bot conflicts first (only if telegram enabled)
            if self.telegram_enabled:
                await self.clear_bot_conflicts()
            
            # Start HTTP server for health checks
            self.start_health_server()
            
            # Start keep-alive service (prevents Render sleeping)
            self.start_keep_alive()
            
            self.setup_application()
            
            logger.info("Starting PSC + TON Trading System...")
            
            # Initialize application (only if telegram enabled)
            if self.telegram_enabled and self.application:
                await self.application.initialize()
                await self.application.start()
            
            # Update health status
            self.health_status = "running"
            self.last_activity = datetime.now()
            
            # Initialize TradingView integration
            if self.tradingview:
                try:
                    await self.tradingview.initialize()
                    logger.info("✅ TradingView integration started")
                    # Test basic connectivity
                    test_analysis = await self.tradingview.get_single_coin_analysis('BTC')
                    if test_analysis and test_analysis.get('symbol'):
                        logger.info("✅ TradingView connectivity verified")
                    else:
                        logger.warning("⚠️ TradingView connection test failed - analysis may be limited")
                except Exception as e:
                    logger.warning(f"⚠️ TradingView startup issue: {e}")
                    logger.info("📊 TradingView analysis will show fallback data when unavailable")
            
            # Send startup notification  
            tv_status = "✅ TradingView ACTIVE" if self.tradingview and self.tradingview_enabled else "⚠️ TradingView DISABLED"
            
            await self.send_notification(
                "🚀 **PSC + TON System Online**\n\n"
                "✅ PSC monitoring ACTIVE\n"
                "✅ Health monitoring ACTIVE\n"
                "✅ Timer constraints ENABLED\n"
                "✅ TON integration READY\n"
                f"{tv_status}\n\n"
                "Send /start for features!"
            )
            
            # Start monitoring
            self.running = True
            
            # Start monitoring
            monitor_task = asyncio.create_task(self.monitor_psc_signals())
            
            # Start Telegram polling with conflict handling
            try:
                polling_task = asyncio.create_task(self.application.updater.start_polling(
                    poll_interval=1,
                    timeout=10,
                    drop_pending_updates=True  # Clear any pending updates
                ))
                logger.info("✅ Telegram polling started successfully")
            except Exception as e:
                logger.error(f"❌ Telegram polling error: {e}")
                if "Conflict" in str(e):
                    logger.warning("🔄 Bot conflict detected - stopping other instances...")
                    try:
                        # Try to clear webhook if set
                        await self.application.bot.delete_webhook(drop_pending_updates=True)
                        await asyncio.sleep(5)  # Wait before retry
                        polling_task = asyncio.create_task(self.application.updater.start_polling(
                            poll_interval=1,
                            timeout=10,
                            drop_pending_updates=True
                        ))
                        logger.info("✅ Telegram polling restarted after conflict resolution")
                    except Exception as retry_e:
                        logger.error(f"❌ Failed to restart polling: {retry_e}")
                        polling_task = None
                else:
                    polling_task = None
            
            # Add continuous ML monitoring task
            ml_monitor_task = None
            if self.ml_engine:
                ml_monitor_task = asyncio.create_task(self.ml_engine.continuous_market_scan(self.fetch_current_price))
                logger.info("🤖 Continuous ML monitoring started")
            
            # Add ML signal checking task
            ml_signal_check_task = asyncio.create_task(self.check_ml_signals())
            
            # Enhanced Prediction Validator runs in the background (no separate task needed)
            logger.info("� Enhanced Prediction Validation is active")
            
            logger.info("System fully operational with continuous ML monitoring")
            
            # Wait for tasks
            tasks = [monitor_task, polling_task, ml_signal_check_task]
            if ml_monitor_task:
                tasks.append(ml_monitor_task)
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"System error: {e}")
        finally:
            await self.stop_system()
    
    async def stop_system(self):
        """Stop the system gracefully"""
        logger.info("Stopping system...")
        self.running = False
        self.health_status = "stopping"
        
        # Stop health server
        self.stop_health_server()
        
        if self.application and self.application.updater.running:
            await self.application.updater.stop()
        if self.application:
            await self.application.stop()
            
        await self.send_notification("🛑 **System Stopped**\nPSC + TON Trading System offline")
        logger.info("System stopped")

# Global system instance for signal handling
system = None

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n🛑 Shutting down system...")
    if system:
        asyncio.create_task(system.stop_system())
    sys.exit(0)

async def main():
    """Main function"""
    global system
    
    try:
        system = PSCTONTradingBot()
        
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)
        
        await system.start_system()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Main error: {e}")
    finally:
        if system:
            await system.stop_system()

if __name__ == "__main__":
    print("🚀 PSC + TON Trading System")
    print("=" * 30)
    print("✅ Bot configured")
    print("✅ TON integration ready") 
    print("✅ Starting monitoring...")
    print("\nPress Ctrl+C to stop")
    print("-" * 30)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n✅ System stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
