#!/usr/bin/env python3
"""
PSC Trading System - Unified Database Schema
Replaces CSV files with SQLite database for better data management
"""

import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any
import uuid
import logging

logger = logging.getLogger(__name__)

class PSCDatabase:
    """Unified database for PSC trading system"""
    
    def __init__(self, db_path: str = "data/psc_trading.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = None
        self.init_database()
    
    @property 
    def connection(self):
        """Return a persistent database connection (for compatibility)"""
        if self._connection is None:
            self._connection = sqlite3.connect(self.db_path)
        return self._connection
    
    def close_connection(self):
        """Close the database connection"""
        if self._connection:
            self._connection.close()
            self._connection = None
    
    def get_utc_timestamp(self) -> str:
        """Get consistent UTC timestamp"""
        return datetime.now(timezone.utc).isoformat()
    
    def init_database(self):
        """Initialize database with proper schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('PRAGMA foreign_keys = ON')
            
            # Signals table - ML predictions and PSC signals
            conn.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    coin TEXT NOT NULL,
                    signal_type TEXT NOT NULL, -- 'PSC_SIGNAL', 'ML_PREDICTION'
                    price REAL NOT NULL,
                    ratio REAL,
                    confidence REAL NOT NULL,
                    direction TEXT NOT NULL, -- 'LONG', 'SHORT'
                    exit_estimate REAL,
                    ml_prediction REAL,
                    signal_strength TEXT, -- 'WEAK', 'MEDIUM', 'STRONG'
                    market_conditions TEXT,
                    ml_features TEXT, -- JSON string of ML features
                    processed BOOLEAN DEFAULT FALSE,
                    created_at TEXT DEFAULT (datetime('now', 'utc'))
                )
            ''')
            
            # Trades table - actual executed trades
            conn.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id TEXT PRIMARY KEY,
                    signal_id TEXT,
                    timestamp TEXT NOT NULL,
                    coin TEXT NOT NULL,
                    trade_type TEXT NOT NULL, -- 'PAPER', 'LIVE'
                    side TEXT NOT NULL, -- 'BUY', 'SELL'
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    quantity REAL NOT NULL,
                    profit_pct REAL DEFAULT 0,
                    profit_usd REAL DEFAULT 0,
                    confidence REAL,
                    ml_prediction REAL,
                    ratio REAL,
                    direction TEXT,
                    trade_duration_minutes INTEGER,
                    successful BOOLEAN,
                    exit_reason TEXT, -- 'PROFIT_TARGET', 'STOP_LOSS', 'TIMEOUT'
                    status TEXT DEFAULT 'OPEN', -- 'OPEN', 'CLOSED', 'CANCELLED'
                    created_at TEXT DEFAULT (datetime('now', 'utc')),
                    closed_at TEXT,
                    FOREIGN KEY (signal_id) REFERENCES signals (id)
                )
            ''')
            
            # Validation table - prediction accuracy tracking
            conn.execute('''
                CREATE TABLE IF NOT EXISTS validation (
                    id TEXT PRIMARY KEY,
                    signal_id TEXT NOT NULL,
                    trade_id TEXT,
                    timestamp TEXT NOT NULL,
                    predicted_outcome TEXT NOT NULL, -- 'PROFIT', 'LOSS'
                    actual_outcome TEXT, -- 'PROFIT', 'LOSS', 'PENDING'
                    predicted_confidence REAL NOT NULL,
                    actual_profit_pct REAL,
                    accuracy_score REAL, -- 0-1 based on prediction vs actual
                    time_to_outcome_minutes INTEGER,
                    market_conditions_at_prediction TEXT,
                    market_conditions_at_outcome TEXT,
                    validation_status TEXT DEFAULT 'PENDING', -- 'PENDING', 'VALIDATED', 'EXPIRED'
                    validation_result TEXT, -- 'CORRECT', 'INCORRECT', 'TIMEOUT'
                    created_at TEXT DEFAULT (datetime('now', 'utc')),
                    FOREIGN KEY (signal_id) REFERENCES signals (id),
                    FOREIGN KEY (trade_id) REFERENCES trades (id)
                )
            ''')
            
            # Add validation_status column if it doesn't exist (migration)
            try:
                conn.execute("ALTER TABLE validation ADD COLUMN validation_status TEXT DEFAULT 'PENDING'")
                logger.info("✅ Added validation_status column to validation table")
            except sqlite3.OperationalError:
                # Column already exists, which is fine
                pass
                
            # Add validation_result column if it doesn't exist (migration)
            try:
                conn.execute("ALTER TABLE validation ADD COLUMN validation_result TEXT")
                logger.info("✅ Added validation_result column to validation table")
            except sqlite3.OperationalError:
                # Column already exists, which is fine
                pass
            
            # Performance table - daily/session summaries
            conn.execute('''
                CREATE TABLE IF NOT EXISTS performance (
                    id TEXT PRIMARY KEY,
                    date TEXT NOT NULL, -- YYYY-MM-DD
                    session_start TEXT NOT NULL,
                    session_end TEXT,
                    total_signals INTEGER DEFAULT 0,
                    total_trades INTEGER DEFAULT 0,
                    successful_trades INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 0,
                    total_profit_pct REAL DEFAULT 0,
                    total_profit_usd REAL DEFAULT 0,
                    avg_confidence REAL DEFAULT 0,
                    best_trade_pct REAL DEFAULT 0,
                    worst_trade_pct REAL DEFAULT 0,
                    max_drawdown_pct REAL DEFAULT 0,
                    ml_accuracy REAL DEFAULT 0,
                    signals_per_hour REAL DEFAULT 0,
                    avg_trade_duration_minutes REAL DEFAULT 0,
                    market_conditions_summary TEXT,
                    notes TEXT,
                    created_at TEXT DEFAULT (datetime('now', 'utc'))
                )
            ''')
            
            # System events table - for logging and debugging
            conn.execute('''
                CREATE TABLE IF NOT EXISTS system_events (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL, -- 'STARTUP', 'SHUTDOWN', 'ERROR', 'INFO'
                    component TEXT NOT NULL, -- 'PSC_SYSTEM', 'ML_ENGINE', 'TRADER'
                    message TEXT NOT NULL,
                    details TEXT, -- JSON string with additional data
                    severity TEXT DEFAULT 'INFO', -- 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
                    created_at TEXT DEFAULT (datetime('now', 'utc'))
                )
            ''')
            
            # Create indexes for better performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals (timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_signals_coin ON signals (coin)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades (timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_trades_signal_id ON trades (signal_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_validation_signal_id ON validation (signal_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_performance_date ON performance (date)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_events_timestamp ON system_events (timestamp)')
            
            conn.commit()
        
        logger.info(f"✅ Database initialized: {self.db_path}")
    
    def execute_query(self, query: str, params: tuple = ()) -> List[Dict]:
        """Execute a query and return results as list of dictionaries"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row  # Enable column access by name
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Database query error: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            return []
    
    def execute_update(self, query: str, params: tuple = ()) -> int:
        """Execute an update/insert query and return affected row count"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(query, params)
                conn.commit()
                return cursor.rowcount
        except Exception as e:
            logger.error(f"Database update error: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            return 0
    
    def log_signal(self, coin: str, price: float, ratio: float, confidence: float, 
                   direction: str, exit_estimate: float, ml_prediction: float,
                   signal_strength: str, market_conditions: str, 
                   ml_features: Dict = None) -> str:
        """Log a new signal and return signal ID"""
        signal_id = str(uuid.uuid4())
        timestamp = self.get_utc_timestamp()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO signals (
                    id, timestamp, coin, signal_type, price, ratio, confidence,
                    direction, exit_estimate, ml_prediction, signal_strength,
                    market_conditions, ml_features
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal_id, timestamp, coin, 'PSC_SIGNAL', price, ratio, confidence,
                direction, exit_estimate, ml_prediction, signal_strength,
                market_conditions, json.dumps(ml_features) if ml_features else None
            ))
            conn.commit()
        
        logger.info(f"📊 Signal logged: {signal_id} - {coin} {direction} @ {price}")
        return signal_id
    
    def log_ml_signal(self, coin: str, price: float, confidence: float, 
                      direction: str, ml_prediction: float, ml_features: Dict = None) -> str:
        """Log a new ML signal and return signal ID"""
        signal_id = str(uuid.uuid4())
        timestamp = self.get_utc_timestamp()
        
        # Extract values from ml_features if available
        ratio = ml_features.get('ratio', 0.0) if ml_features else 0.0
        exit_estimate = price * (1.02 if direction == 'LONG' else 0.98)  # Default 2% target
        signal_strength = self._get_signal_strength(confidence)
        market_conditions = ml_features.get('market_session', 'UNKNOWN') if ml_features else 'UNKNOWN'
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO signals (
                    id, timestamp, coin, signal_type, price, ratio, confidence,
                    direction, exit_estimate, ml_prediction, signal_strength,
                    market_conditions, ml_features
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal_id, timestamp, coin, 'ML_SIGNAL', price, ratio, confidence,
                direction, exit_estimate, ml_prediction, signal_strength,
                market_conditions, json.dumps(ml_features) if ml_features else None
            ))
            conn.commit()
        
        logger.info(f"🤖 ML signal logged: {signal_id} - {coin} {direction} @ {price}")
        return signal_id
    
    def _get_signal_strength(self, confidence: float) -> str:
        """Determine signal strength from confidence level"""
        if confidence >= 0.8:
            return 'STRONG'
        elif confidence >= 0.6:
            return 'MEDIUM'
        else:
            return 'WEAK'
    
    def log_trade(self, signal_id: str, coin: str, entry_price: float, 
                  quantity: float, confidence: float, ml_prediction: float,
                  ratio: float, direction: str, trade_type: str = 'PAPER') -> str:
        """Log a new trade and return trade ID"""
        trade_id = str(uuid.uuid4())
        timestamp = self.get_utc_timestamp()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO trades (
                    id, signal_id, timestamp, coin, trade_type, side, entry_price,
                    quantity, confidence, ml_prediction, ratio, direction, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_id, signal_id, timestamp, coin, trade_type,
                'BUY' if direction == 'LONG' else 'SELL', entry_price,
                quantity, confidence, ml_prediction, ratio, direction, 'OPEN'
            ))
            conn.commit()
        
        logger.info(f"💰 Trade logged: {trade_id} - {coin} {direction} @ {entry_price}")
        return trade_id
    
    def close_trade(self, trade_id: str, exit_price: float, profit_pct: float,
                    profit_usd: float, exit_reason: str = 'PROFIT_TARGET') -> bool:
        """Close a trade with results"""
        timestamp = self.get_utc_timestamp()
        successful = profit_pct > 0
        
        with sqlite3.connect(self.db_path) as conn:
            # Update trade
            cursor = conn.execute('''
                UPDATE trades 
                SET exit_price = ?, profit_pct = ?, profit_usd = ?, 
                    successful = ?, exit_reason = ?, status = 'CLOSED',
                    closed_at = ?
                WHERE id = ?
            ''', (exit_price, profit_pct, profit_usd, successful, exit_reason, timestamp, trade_id))
            
            if cursor.rowcount == 0:
                logger.error(f"❌ Trade not found: {trade_id}")
                return False
            
            conn.commit()
        
        logger.info(f"🏁 Trade closed: {trade_id} - {profit_pct:.2f}% profit")
        return True
    
    def log_validation(self, signal_id: str, predicted_outcome: str, 
                       predicted_confidence: float, actual_outcome: str = None,
                       actual_profit_pct: float = None, trade_id: str = None) -> str:
        """Log prediction validation"""
        validation_id = str(uuid.uuid4())
        timestamp = self.get_utc_timestamp()
        
        # Calculate accuracy score if we have actual outcome
        accuracy_score = None
        if actual_outcome and predicted_outcome:
            accuracy_score = 1.0 if predicted_outcome == actual_outcome else 0.0
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO validation (
                    id, signal_id, trade_id, timestamp, predicted_outcome,
                    actual_outcome, predicted_confidence, actual_profit_pct,
                    accuracy_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                validation_id, signal_id, trade_id, timestamp, predicted_outcome,
                actual_outcome, predicted_confidence, actual_profit_pct, accuracy_score
            ))
            conn.commit()
        
        logger.info(f"✅ Validation logged: {validation_id} - {predicted_outcome}")
        return validation_id
    
    def log_system_event(self, event_type: str, component: str, message: str,
                         details: Dict = None, severity: str = 'INFO'):
        """Log system events for debugging and monitoring"""
        event_id = str(uuid.uuid4())
        timestamp = self.get_utc_timestamp()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO system_events (
                    id, timestamp, event_type, component, message, details, severity
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                event_id, timestamp, event_type, component, message,
                json.dumps(details) if details else None, severity
            ))
            conn.commit()
    
    def get_daily_performance(self, date: str = None) -> Dict:
        """Get performance metrics for a specific date"""
        if date is None:
            date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get basic stats
            cursor = conn.execute('''
                SELECT 
                    COUNT(*) as total_signals,
                    AVG(confidence) as avg_confidence
                FROM signals 
                WHERE date(timestamp) = ?
            ''', (date,))
            signal_stats = cursor.fetchone()
            
            cursor = conn.execute('''
                SELECT 
                    COUNT(*) as total_trades,
                    COUNT(CASE WHEN successful = 1 THEN 1 END) as successful_trades,
                    SUM(profit_pct) as total_profit_pct,
                    SUM(profit_usd) as total_profit_usd,
                    MAX(profit_pct) as best_trade,
                    MIN(profit_pct) as worst_trade,
                    AVG(trade_duration_minutes) as avg_duration
                FROM trades 
                WHERE date(timestamp) = ? AND status = 'CLOSED'
            ''', (date,))
            trade_stats = cursor.fetchone()
            
            # Calculate success rate
            success_rate = 0
            if trade_stats['total_trades'] > 0:
                success_rate = (trade_stats['successful_trades'] / trade_stats['total_trades']) * 100
            
            return {
                'date': date,
                'total_signals': signal_stats['total_signals'] or 0,
                'total_trades': trade_stats['total_trades'] or 0,
                'successful_trades': trade_stats['successful_trades'] or 0,
                'success_rate': success_rate,
                'total_profit_pct': trade_stats['total_profit_pct'] or 0,
                'total_profit_usd': trade_stats['total_profit_usd'] or 0,
                'avg_confidence': signal_stats['avg_confidence'] or 0,
                'best_trade': trade_stats['best_trade'] or 0,
                'worst_trade': trade_stats['worst_trade'] or 0,
                'avg_duration': trade_stats['avg_duration'] or 0
            }
    
    def export_to_csv(self, table_name: str, output_file: str = None) -> str:
        """Export database table to CSV for compatibility"""
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"export_{table_name}_{timestamp}.csv"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(f"SELECT * FROM {table_name}")
            
            import csv
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write headers
                writer.writerow([description[0] for description in cursor.description])
                
                # Write data
                writer.writerows(cursor.fetchall())
        
        logger.info(f"📁 Exported {table_name} to {output_file}")
        return output_file
    
    def save_integrated_signal(self, signal_data: Dict) -> str:
        """Save integrated signal with component analysis to database"""
        signal_id = str(uuid.uuid4())
        timestamp = self.get_utc_timestamp()
        
        # Extract main signal data
        main_signal = signal_data.get('main_signal', {})
        component_signals = signal_data.get('component_signals', [])
        
        with sqlite3.connect(self.db_path) as conn:
            # Log main integrated signal
            conn.execute('''
                INSERT INTO signals (
                    id, timestamp, coin, signal_type, price, ratio, confidence,
                    direction, exit_estimate, ml_prediction, signal_strength,
                    market_conditions, ml_features
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal_id, timestamp, 
                main_signal.get('coin', 'UNKNOWN'),
                'INTEGRATED_SIGNAL',
                main_signal.get('price', 0),
                main_signal.get('ratio', 0),
                main_signal.get('confidence', 0),
                main_signal.get('direction', 'UNKNOWN'),
                main_signal.get('exit_estimate', 0),
                main_signal.get('ml_prediction', 0),
                main_signal.get('signal_strength', 'MEDIUM'),
                main_signal.get('market_conditions', ''),
                json.dumps({
                    'component_count': len(component_signals),
                    'consensus_score': signal_data.get('consensus_score', 0),
                    'quality_passed': signal_data.get('quality_passed', False),
                    'components': [cs.get('source', '') for cs in component_signals]
                })
            ))
            
            # Log system event for integrated signal
            event_id = str(uuid.uuid4())
            conn.execute('''
                INSERT INTO system_events (
                    id, timestamp, event_type, component, message, details, severity
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                event_id, timestamp,
                'INTEGRATED_SIGNAL_PROCESSED',
                'IntegratedSignalProcessor',
                f"Processed integrated signal for {main_signal.get('coin', 'UNKNOWN')}",
                json.dumps({
                    'signal_id': signal_id,
                    'consensus_score': signal_data.get('consensus_score', 0),
                    'component_count': len(component_signals),
                    'quality_passed': signal_data.get('quality_passed', False)
                }),
                'INFO'
            ))
            
            conn.commit()
        
        logger.info(f"🎯 Integrated signal saved: {signal_id}")
        return signal_id
    
    def update_accuracy_weights(self, component_accuracies: Dict[str, float]):
        """Update component accuracy weights in system events for tracking"""
        timestamp = self.get_utc_timestamp()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO system_events (
                    id, timestamp, event_type, component, message, details, severity
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                str(uuid.uuid4()), timestamp, 
                'ACCURACY_WEIGHTS_UPDATE',
                'AccuracyOptimizer',
                'Updated component accuracy weights',
                json.dumps(component_accuracies),
                'INFO'
            ))
            conn.commit()
        
        logger.info(f"⚖️ Accuracy weights updated: {component_accuracies}")
    
    def get_latest_accuracy_weights(self) -> Dict[str, float]:
        """Get the most recent accuracy weights from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT details FROM system_events 
                WHERE event_type = 'ACCURACY_WEIGHTS_UPDATE'
                ORDER BY timestamp DESC 
                LIMIT 1
            ''')
            
            row = cursor.fetchone()
            if row and row[0]:
                try:
                    return json.loads(row[0])
                except:
                    pass
            
            # Return default weights if none found
            return {
                'psc': 0.25,
                'ml_engine': 0.25,
                'tradingview': 0.25,
                'ml_microstructure': 0.25
            }
    
    def get_component_accuracy_history(self, component: str, limit: int = 10) -> List[Dict]:
        """Get accuracy history for a specific component"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            cursor = conn.execute('''
                SELECT timestamp, details FROM system_events 
                WHERE event_type = 'ACCURACY_WEIGHTS_UPDATE'
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            
            history = []
            for row in cursor.fetchall():
                try:
                    details = json.loads(row['details'])
                    if component in details:
                        history.append({
                            'timestamp': row['timestamp'],
                            'accuracy': details[component]
                        })
                except:
                    continue
            
            return history
    
    def get_database_stats(self) -> Dict:
        """Get overall database statistics"""
        with sqlite3.connect(self.db_path) as conn:
            stats = {}
            
            tables = ['signals', 'trades', 'validation', 'performance', 'system_events']
            for table in tables:
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                stats[f"{table}_count"] = cursor.fetchone()[0]
            
            # Database file size
            stats['db_size_mb'] = self.db_path.stat().st_size / (1024 * 1024)
            
            return stats

# Example usage and testing
if __name__ == "__main__":
    # Test the database
    db = PSCDatabase("test_psc.db")
    
    # Log a sample signal
    signal_id = db.log_signal(
        coin="TONUSDT",
        price=2.345,
        ratio=1.15,
        confidence=0.85,
        direction="LONG",
        exit_estimate=2.380,
        ml_prediction=0.78,
        signal_strength="STRONG",
        market_conditions="BULLISH"
    )
    
    # Log a trade based on that signal
    trade_id = db.log_trade(
        signal_id=signal_id,
        coin="TONUSDT",
        entry_price=2.345,
        quantity=100,
        confidence=0.85,
        ml_prediction=0.78,
        ratio=1.15,
        direction="LONG"
    )
    
    # Close the trade
    db.close_trade(
        trade_id=trade_id,
        exit_price=2.380,
        profit_pct=1.49,
        profit_usd=3.50
    )
    
    # Log validation
    db.log_validation(
        signal_id=signal_id,
        predicted_outcome="PROFIT",
        predicted_confidence=0.85,
        actual_outcome="PROFIT",
        actual_profit_pct=1.49,
        trade_id=trade_id
    )
    
    # Get today's performance
    performance = db.get_daily_performance()
    print("Daily Performance:", performance)
    
    # Get database stats
    stats = db.get_database_stats()
    print("Database Stats:", stats)
