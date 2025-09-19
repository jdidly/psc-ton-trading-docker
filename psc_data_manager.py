#!/usr/bin/env python3
"""
PSC System Database Integration
Replaces CSV file operations with database operations
"""

import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import json

from psc_database import PSCDatabase

logger = logging.getLogger(__name__)

class PSCDataManager:
    """
    Data management layer for PSC trading system
    Handles all data operations with proper timekeeping and correlation
    """
    
    def __init__(self, db_path: str = "data/psc_trading.db"):
        self.db = PSCDatabase(db_path)
        self.session_stats = {
            'signals_generated': 0,
            'trades_executed': 0,
            'successful_trades': 0,
            'session_start': self.db.get_utc_timestamp()
        }
        
        # Log system startup
        self.db.log_system_event(
            event_type='STARTUP',
            component='PSC_DATA_MANAGER',
            message='PSC Data Manager initialized',
            details={'db_path': str(db_path)},
            severity='INFO'
        )
        
        logger.info("✅ PSC Data Manager initialized with database")
    
    def _serialize_ml_features(self, ml_features):
        """Helper to serialize ML features with datetime handling"""
        if ml_features is None:
            return None
        
        # Convert datetime objects to ISO format strings
        serializable_features = {}
        for key, value in ml_features.items():
            if isinstance(value, datetime):
                serializable_features[key] = value.isoformat()
            else:
                serializable_features[key] = value
        
        return serializable_features
    
    def log_psc_signal(self, coin: str, price: float, ratio: float, 
                       confidence: float, direction: str, exit_estimate: float,
                       ml_prediction: float, market_conditions: str = None,
                       ml_features: Dict = None) -> str:
        """
        Log a PSC signal - replaces old CSV signal logging
        Returns signal_id for correlation with trades
        """
        try:
            # Determine signal strength based on confidence
            if confidence >= 0.8:
                signal_strength = "STRONG"
            elif confidence >= 0.6:
                signal_strength = "MEDIUM"
            else:
                signal_strength = "WEAK"
            
            # Default market conditions if not provided
            if market_conditions is None:
                market_conditions = self._assess_market_conditions()
            
            signal_id = self.db.log_signal(
                coin=coin,
                price=price,
                ratio=ratio,
                confidence=confidence,
                direction=direction,
                exit_estimate=exit_estimate,
                ml_prediction=ml_prediction,
                signal_strength=signal_strength,
                market_conditions=market_conditions,
                ml_features=self._serialize_ml_features(ml_features)
            )
            
            # Update session stats
            self.session_stats['signals_generated'] += 1
            
            logger.info(f"📊 PSC Signal logged: {coin} {direction} @ {price:.8f} (confidence: {confidence:.3f})")
            return signal_id
            
        except Exception as e:
            logger.error(f"❌ Error logging PSC signal: {e}")
            self.db.log_system_event(
                event_type='ERROR',
                component='PSC_DATA_MANAGER',
                message=f'Failed to log PSC signal: {e}',
                details={'coin': coin, 'price': price, 'confidence': confidence},
                severity='ERROR'
            )
            return None
    
    def log_ml_signal(self, coin: str, price: float, confidence: float, 
                      direction: str, ml_prediction: float, ml_features: Dict = None) -> str:
        """
        Log an ML-generated signal - optimized for ML engine output
        Returns signal_id for correlation with validation
        """
        try:
            # Use the dedicated ML signal logging method
            signal_id = self.db.log_ml_signal(
                coin=coin,
                price=price,
                confidence=confidence,
                direction=direction,
                ml_prediction=ml_prediction,
                ml_features=ml_features or {}
            )
            
            self.session_stats['signals_generated'] += 1
            logger.info(f"🤖 ML signal logged: {coin} {direction} @${price:.6f} (ID: {signal_id})")
            return signal_id
            
        except Exception as e:
            logger.error(f"Error logging ML signal: {e}")
            self.db.log_system_event(
                event_type='ERROR',
                component='PSC_DATA_MANAGER',
                message=f'Failed to log ML signal: {e}',
                details={'coin': coin, 'price': price, 'confidence': confidence},
                severity='ERROR'
            )
            return None
    
    def log_trade_execution(self, signal_id: str, coin: str, entry_price: float,
                           quantity: float, confidence: float, ml_prediction: float,
                           ratio: float, direction: str, trade_type: str = 'PAPER') -> str:
        """
        Log trade execution - replaces old CSV trade logging  
        Returns trade_id for tracking
        """
        try:
            trade_id = self.db.log_trade(
                signal_id=signal_id,
                coin=coin,
                entry_price=entry_price,
                quantity=quantity,
                confidence=confidence,
                ml_prediction=ml_prediction,
                ratio=ratio,
                direction=direction,
                trade_type=trade_type
            )
            
            # Update session stats
            self.session_stats['trades_executed'] += 1
            
            logger.info(f"💰 Trade executed: {coin} {direction} @ {entry_price:.8f}")
            return trade_id
            
        except Exception as e:
            logger.error(f"❌ Error logging trade: {e}")
            self.db.log_system_event(
                event_type='ERROR',
                component='PSC_DATA_MANAGER',
                message=f'Failed to log trade execution: {e}',
                details={'signal_id': signal_id, 'coin': coin, 'entry_price': entry_price},
                severity='ERROR'
            )
            return None
    
    def close_trade_with_results(self, trade_id: str, exit_price: float,
                                profit_pct: float, profit_usd: float,
                                exit_reason: str = 'PROFIT_TARGET') -> bool:
        """
        Close trade with results - replaces old CSV profit logging
        """
        try:
            success = self.db.close_trade(
                trade_id=trade_id,
                exit_price=exit_price,
                profit_pct=profit_pct,
                profit_usd=profit_usd,
                exit_reason=exit_reason
            )
            
            if success and profit_pct > 0:
                self.session_stats['successful_trades'] += 1
            
            logger.info(f"🏁 Trade closed: {trade_id} - {profit_pct:.2f}% profit")
            return success
            
        except Exception as e:
            logger.error(f"❌ Error closing trade: {e}")
            self.db.log_system_event(
                event_type='ERROR',
                component='PSC_DATA_MANAGER',
                message=f'Failed to close trade: {e}',
                details={'trade_id': trade_id, 'profit_pct': profit_pct},
                severity='ERROR'
            )
            return False
    
    def log_prediction_validation(self, signal_id: str, predicted_outcome: str,
                                 predicted_confidence: float, actual_outcome: str = None,
                                 actual_profit_pct: float = None, trade_id: str = None) -> str:
        """
        Log ML prediction validation - new functionality
        """
        try:
            validation_id = self.db.log_validation(
                signal_id=signal_id,
                predicted_outcome=predicted_outcome,
                predicted_confidence=predicted_confidence,
                actual_outcome=actual_outcome,
                actual_profit_pct=actual_profit_pct,
                trade_id=trade_id
            )
            
            logger.info(f"✅ Prediction validated: {predicted_outcome} -> {actual_outcome}")
            return validation_id
            
        except Exception as e:
            logger.error(f"❌ Error logging validation: {e}")
            return None
    
    def get_session_stats(self) -> Dict:
        """Get current session statistics"""
        # Get today's database performance
        today_performance = self.db.get_daily_performance()
        
        # Combine with session stats
        return {
            **self.session_stats,
            'daily_performance': today_performance,
            'session_duration_hours': self._get_session_duration_hours(),
            'signals_per_hour': self._calculate_signals_per_hour(),
            'success_rate': self._calculate_session_success_rate()
        }
    
    def get_system_health(self) -> Dict:
        """Get system health metrics for dashboard"""
        try:
            db_stats = self.db.get_database_stats()
            session_stats = self.get_session_stats()
            
            return {
                'status': 'healthy',
                'database': {
                    'connected': True,
                    'size_mb': db_stats['db_size_mb'],
                    'total_signals': db_stats['signals_count'],
                    'total_trades': db_stats['trades_count'],
                    'total_validations': db_stats['validation_count']
                },
                'session': session_stats,
                'timestamp': self.db.get_utc_timestamp()
            }
        except Exception as e:
            logger.error(f"❌ Error getting system health: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': self.db.get_utc_timestamp()
            }
    
    def export_data_for_analysis(self, table_name: str = None) -> Dict[str, str]:
        """Export database data to CSV for analysis"""
        exported_files = {}
        
        tables_to_export = [table_name] if table_name else ['signals', 'trades', 'validation', 'performance']
        
        for table in tables_to_export:
            try:
                filename = self.db.export_to_csv(table)
                exported_files[table] = filename
                logger.info(f"📁 Exported {table} to {filename}")
            except Exception as e:
                logger.error(f"❌ Error exporting {table}: {e}")
                exported_files[table] = f"Error: {e}"
        
        return exported_files
    
    
    def _get_signal_strength(self, confidence: float) -> str:
        """Determine signal strength from confidence level"""
        if confidence >= 0.8:
            return "STRONG"
        elif confidence >= 0.6:
            return "MEDIUM"
        else:
            return "WEAK"
    
    def _assess_market_conditions(self) -> str:
        """Simple market condition assessment - can be enhanced"""
        # This is a placeholder - you can enhance with actual market analysis
        return "NORMAL"
    
    def _get_session_duration_hours(self) -> float:
        """Calculate session duration in hours"""
        start_time = datetime.fromisoformat(self.session_stats['session_start'].replace('Z', '+00:00'))
        current_time = datetime.now(timezone.utc)
        duration = (current_time - start_time).total_seconds() / 3600
        return round(duration, 2)
    
    def _calculate_signals_per_hour(self) -> float:
        """Calculate signals generated per hour"""
        duration_hours = self._get_session_duration_hours()
        if duration_hours > 0:
            return round(self.session_stats['signals_generated'] / duration_hours, 2)
        return 0
    
    def _calculate_session_success_rate(self) -> float:
        """Calculate session success rate"""
        if self.session_stats['trades_executed'] > 0:
            return round((self.session_stats['successful_trades'] / self.session_stats['trades_executed']) * 100, 2)
        return 0
    
    def get_recent_trades(self, limit: int = 5) -> list:
        """Get recent trades for display (Telegram bot compatibility)"""
        return self.db.get_recent_trades(limit)
    
    def get_all_trades_for_performance(self) -> list:
        """Get all trades for performance analysis"""
        return self.db.get_all_trades_for_performance()
    
    def get_trade_statistics(self) -> Dict:
        """Get comprehensive trade statistics"""
        return self.db.get_trade_statistics()
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        return self.db.get_database_stats()
    
    def cleanup_and_close(self):
        """Cleanup when shutting down"""
        self.db.log_system_event(
            event_type='SHUTDOWN',
            component='PSC_DATA_MANAGER',
            message='PSC Data Manager shutting down',
            details=self.get_session_stats(),
            severity='INFO'
        )
        logger.info("🔄 PSC Data Manager shutdown complete")
    
    def execute_query(self, query: str, params: tuple = None):
        """Execute a database query - for ML engine compatibility"""
        try:
            return self.db.execute_query(query, params)
        except Exception as e:
            logger.error(f"Database query error: {e}")
            return []
    
    def get_ml_signals(self, limit: int = 100):
        """Get ML signals for learning - for ML engine compatibility"""
        try:
            query = """
            SELECT coin, price, confidence, direction, ml_prediction, 
                   ml_features, timestamp, signal_id
            FROM signals 
            WHERE ml_features IS NOT NULL 
            ORDER BY timestamp DESC 
            LIMIT ?
            """
            return self.execute_query(query, (limit,))
        except Exception as e:
            logger.error(f"Error getting ML signals: {e}")
            return []
    
    def get_ml_predictions(self, limit: int = 100):
        """Get ML predictions for validation - for ML engine compatibility"""
        try:
            query = """
            SELECT signal_id, coin, confidence, predicted_outcome, 
                   predicted_confidence, actual_outcome, actual_profit_pct, 
                   timestamp
            FROM prediction_validations 
            ORDER BY timestamp DESC 
            LIMIT ?
            """
            return self.execute_query(query, (limit,))
        except Exception as e:
            logger.error(f"Error getting ML predictions: {e}")
            return []

# Integration helper functions for easy migration from CSV
class PSCLegacyCompatibility:
    """Helper class to maintain compatibility with existing CSV-based code"""
    
    def __init__(self, data_manager: PSCDataManager):
        self.data_manager = data_manager
    
    def log_signal_csv_compatible(self, coin: str, price: float, ratio: float,
                                 confidence: float, direction: str, exit_estimate: float,
                                 ml_prediction: float, signal_strength: str, market_conditions: str):
        """CSV-compatible signal logging"""
        return self.data_manager.log_psc_signal(
            coin=coin,
            price=price,
            ratio=ratio,
            confidence=confidence,
            direction=direction,
            exit_estimate=exit_estimate,
            ml_prediction=ml_prediction,
            market_conditions=market_conditions
        )
    
    def log_trade_csv_compatible(self, coin: str, entry_price: float, exit_price: float,
                                confidence: float, ml_prediction: float, ratio: float,
                                direction: str, successful: bool, profit_pct: float = 0,
                                profit_usd: float = 0, prediction_id: str = None):
        """CSV-compatible trade logging"""
        # For compatibility, create a signal ID if none provided
        if prediction_id is None:
            signal_id = self.data_manager.log_psc_signal(
                coin=coin,
                price=entry_price,
                ratio=ratio,
                confidence=confidence,
                direction=direction,
                exit_estimate=exit_price,
                ml_prediction=ml_prediction
            )
        else:
            signal_id = prediction_id
        
        # Log trade execution
        trade_id = self.data_manager.log_trade_execution(
            signal_id=signal_id,
            coin=coin,
            entry_price=entry_price,
            quantity=100,  # Default quantity
            confidence=confidence,
            ml_prediction=ml_prediction,
            ratio=ratio,
            direction=direction
        )
        
        # Close trade immediately with results
        if trade_id:
            self.data_manager.close_trade_with_results(
                trade_id=trade_id,
                exit_price=exit_price,
                profit_pct=profit_pct,
                profit_usd=profit_usd,
                exit_reason='PROFIT_TARGET' if successful else 'STOP_LOSS'
            )
        
        return trade_id

# Example usage for testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test the data manager
    data_manager = PSCDataManager("test_integration.db")
    
    # Test signal logging
    signal_id = data_manager.log_psc_signal(
        coin="TONUSDT",
        price=2.345,
        ratio=1.15,
        confidence=0.85,
        direction="LONG",
        exit_estimate=2.380,
        ml_prediction=0.78
    )
    
    # Test trade execution
    trade_id = data_manager.log_trade_execution(
        signal_id=signal_id,
        coin="TONUSDT",
        entry_price=2.345,
        quantity=100,
        confidence=0.85,
        ml_prediction=0.78,
        ratio=1.15,
        direction="LONG"
    )
    
    # Test trade closure
    data_manager.close_trade_with_results(
        trade_id=trade_id,
        exit_price=2.380,
        profit_pct=1.49,
        profit_usd=3.50
    )
    
    # Test validation logging
    data_manager.log_prediction_validation(
        signal_id=signal_id,
        predicted_outcome="PROFIT",
        predicted_confidence=0.85,
        actual_outcome="PROFIT",
        actual_profit_pct=1.49,
        trade_id=trade_id
    )
    
    # Get system health
    health = data_manager.get_system_health()
    print("System Health:", json.dumps(health, indent=2))
    
    # Export data
    exported = data_manager.export_data_for_analysis()
    print("Exported files:", exported)
    
    # Cleanup
    data_manager.cleanup_and_close()
