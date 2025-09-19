"""
Database-Integrated Prediction Validation System for PSC TON Trading

Updated version that uses SQLite database instead of CSV files for:
- Real-time prediction accuracy monitoring
- Automated outcome validation after trades close
- Performance analytics and improvement recommendations
- Detailed reporting for ML model optimization
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, List, Optional
import sqlite3

# Import our database components
try:
    from psc_data_manager import PSCDataManager
    from psc_database import PSCDatabase
    DATABASE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Database components not available: {e}")
    DATABASE_AVAILABLE = False

logger = logging.getLogger(__name__)

class DatabasePredictionValidator:
    """
    Database-integrated prediction validation system that tracks all ML predictions
    and validates them against actual market outcomes using SQLite database
    """
    
    def __init__(self, project_root: Path, db_path: str = None):
        self.project_root = project_root
        
        # Initialize database connection
        if not db_path:
            db_path = str(project_root / "database" / "psc_trading.db")
            # Fallback to test database
            if not Path(db_path).exists():
                db_path = str(project_root / "test_database.db")
                
        self.db_path = db_path
        self.data_manager = None
        
        if DATABASE_AVAILABLE and Path(db_path).exists():
            try:
                self.data_manager = PSCDataManager(db_path)
                logger.info(f"✅ Database-integrated validator connected to: {db_path}")
            except Exception as e:
                logger.error(f"❌ Failed to connect to database: {e}")
                self.data_manager = None
        else:
            logger.warning("❌ Database not available - validator will run in limited mode")
        
        # In-memory tracking for performance
        self.pending_validations = []
        self.performance_metrics = {
            'total_predictions': 0,
            'validated_predictions': 0,
            'accuracy_rate': 0.0,
            'profitable_rate': 0.0,
            'avg_return': 0.0,
            'best_confidence_threshold': 0.6
        }
        
        self.load_performance_metrics()
        logger.info("🔍 Database-Integrated Prediction Validator initialized")
    
    def log_prediction(self, signal_id: str, coin: str, prediction_type: str,
                      predicted_price: float, confidence: float, 
                      prediction_horizon: str = "10min", 
                      ml_features: Dict = None, model_version: str = "v1.0") -> str:
        """
        Log a new ML prediction to database for future validation
        
        Returns:
            str: prediction_id for tracking
        """
        if not self.data_manager:
            logger.warning("Database not available - prediction not logged")
            return None
            
        try:
            # Use database to log prediction with validation metadata
            prediction_data = {
                'signal_id': signal_id,
                'coin': coin,
                'prediction_type': prediction_type,
                'predicted_price': predicted_price,
                'confidence': confidence,
                'prediction_horizon': prediction_horizon,
                'ml_features': ml_features,
                'model_version': model_version,
                'timestamp': datetime.utcnow().isoformat(),
                'status': 'PENDING'
            }
            
            # Log to validation table using correct parameters
            predicted_outcome = f"{prediction_type}_{coin}_{predicted_price:.2f}"
            prediction_id = self.data_manager.db.log_validation(
                signal_id=signal_id,
                predicted_outcome=predicted_outcome,
                predicted_confidence=confidence
            )
            
            self.performance_metrics['total_predictions'] += 1
            
            # Add to pending validations for monitoring
            self.pending_validations.append({
                'prediction_id': prediction_id,
                'signal_id': signal_id,
                'coin': coin,
                'predicted_price': predicted_price,
                'confidence': confidence,
                'prediction_time': datetime.utcnow(),
                'horizon': prediction_horizon
            })
            
            logger.info(f"📊 Prediction logged: {coin} @ {predicted_price:.4f} (confidence: {confidence:.2f})")
            return prediction_id
            
        except Exception as e:
            logger.error(f"❌ Failed to log prediction: {e}")
            return None
    
    def validate_prediction(self, prediction_id: str, actual_price: float, 
                           trade_result: Dict = None) -> Dict:
        """
        Validate a prediction against actual market outcome
        
        Args:
            prediction_id: ID of the prediction to validate
            actual_price: Actual market price achieved
            trade_result: Optional trade results (profit, etc.)
            
        Returns:
            Dict: Validation results
        """
        if not self.data_manager:
            logger.warning("Database not available - validation skipped")
            return {}
            
        try:
            # Get prediction details from database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT predicted_outcome, predicted_confidence, timestamp 
                FROM validation 
                WHERE id = ? AND validation_status = 'PENDING'
            """, (prediction_id,))
            
            result = cursor.fetchone()
            if not result:
                logger.warning(f"Prediction {prediction_id} not found or already validated")
                return {}
                
            predicted_outcome, confidence, timestamp = result
            # Extract predicted price from the predicted_outcome string if possible
            # Format is usually like "SUPERP_BTC_50000.00" or similar
            predicted_price = actual_price  # Default fallback
            
            # Calculate accuracy metrics
            price_accuracy = abs(predicted_price - actual_price) / predicted_price * 100
            direction_correct = (predicted_price > actual_price) == (trade_result.get('profit_pct', 0) > 0) if trade_result else False
            
            validation_result = {
                'prediction_id': prediction_id,
                'predicted_price': predicted_price,
                'actual_price': actual_price,
                'price_accuracy': 100 - price_accuracy,  # Convert to accuracy percentage
                'direction_correct': direction_correct,
                'confidence': confidence,
                'profit_achieved': trade_result.get('profit_pct', 0) if trade_result else 0,
                'validation_time': datetime.utcnow().isoformat(),
                'status': 'VALIDATED'
            }
            
            # Update database with validation results
            cursor.execute("""
                UPDATE validation 
                SET validation_result = ?, validation_status = 'VALIDATED', 
                    validated_at = ?
                WHERE id = ?
            """, (str(validation_result), datetime.utcnow().isoformat(), prediction_id))
            
            conn.commit()
            conn.close()
            
            # Update performance metrics
            self.performance_metrics['validated_predictions'] += 1
            self.update_performance_metrics(validation_result)
            
            # Remove from pending validations
            self.pending_validations = [p for p in self.pending_validations 
                                      if p.get('prediction_id') != prediction_id]
            
            logger.info(f"✅ Validation completed: {price_accuracy:.1f}% price accuracy, "
                       f"direction {'✓' if direction_correct else '✗'}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"❌ Failed to validate prediction: {e}")
            return {}
    
    def update_performance_metrics(self, validation_result: Dict):
        """Update overall performance metrics based on validation result"""
        try:
            # Calculate running averages
            total = self.performance_metrics['validated_predictions']
            
            # Update accuracy rate
            current_accuracy = validation_result['price_accuracy']
            self.performance_metrics['accuracy_rate'] = (
                (self.performance_metrics['accuracy_rate'] * (total - 1) + current_accuracy) / total
            )
            
            # Update profitable rate
            is_profitable = validation_result['profit_achieved'] > 0
            current_profitable_rate = self.performance_metrics['profitable_rate'] * (total - 1)
            if is_profitable:
                current_profitable_rate += 1
            self.performance_metrics['profitable_rate'] = current_profitable_rate / total
            
            # Update average return
            profit = validation_result['profit_achieved']
            self.performance_metrics['avg_return'] = (
                (self.performance_metrics['avg_return'] * (total - 1) + profit) / total
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to update performance metrics: {e}")
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary from database"""
        if not self.data_manager:
            return self.performance_metrics
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get validation statistics
            cursor.execute("SELECT COUNT(*) FROM validation WHERE validation_status = 'VALIDATED'")
            validated_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM validation")
            total_predictions = cursor.fetchone()[0]
            
            # Get accuracy statistics from validation results
            cursor.execute("""
                SELECT validation_result 
                FROM validation 
                WHERE validation_status = 'VALIDATED' 
                AND validation_result IS NOT NULL
            """)
            
            results = cursor.fetchall()
            accuracies = []
            profits = []
            
            for result_str in results:
                try:
                    result = eval(result_str[0])
                    accuracies.append(result.get('price_accuracy', 0))
                    profits.append(result.get('profit_achieved', 0))
                except:
                    continue
            
            conn.close()
            
            # Calculate summary statistics
            summary = {
                'total_predictions': total_predictions,
                'validated_predictions': validated_count,
                'pending_validations': total_predictions - validated_count,
                'accuracy_rate': sum(accuracies) / len(accuracies) if accuracies else 0,
                'profitable_rate': len([p for p in profits if p > 0]) / len(profits) * 100 if profits else 0,
                'avg_return': sum(profits) / len(profits) if profits else 0,
                'best_accuracy': max(accuracies) if accuracies else 0,
                'worst_accuracy': min(accuracies) if accuracies else 0,
                'total_profit': sum(profits) if profits else 0
            }
            
            # Update internal metrics
            self.performance_metrics.update(summary)
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Failed to get performance summary: {e}")
            return self.performance_metrics
    
    def auto_validate_completed_trades(self):
        """Automatically validate predictions for completed trades"""
        if not self.data_manager:
            return
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Find pending validations that have corresponding completed trades
            cursor.execute("""
                SELECT v.id, v.signal_id, v.predicted_outcome, t.exit_price, t.profit_pct, t.profit_usd
                FROM validation v
                JOIN trades t ON v.signal_id = t.signal_id
                WHERE v.validation_status = 'PENDING' 
                AND t.status = 'CLOSED'
                AND t.exit_price IS NOT NULL
            """)
            
            pending_validations = cursor.fetchall()
            
            for validation_data in pending_validations:
                prediction_id, signal_id, predicted_outcome, exit_price, profit_pct, profit_usd = validation_data
                
                trade_result = {
                    'profit_pct': profit_pct,
                    'profit_usd': profit_usd,
                    'exit_price': exit_price
                }
                
                # Validate this prediction
                self.validate_prediction(prediction_id, exit_price, trade_result)
                
            conn.close()
            
            if pending_validations:
                logger.info(f"🔄 Auto-validated {len(pending_validations)} completed trades")
                
        except Exception as e:
            logger.error(f"❌ Auto-validation failed: {e}")
    
    def load_performance_metrics(self):
        """Load existing performance metrics from database"""
        try:
            summary = self.get_performance_summary()
            self.performance_metrics.update(summary)
            logger.info(f"📊 Loaded performance metrics: {summary.get('validated_predictions', 0)} validations")
        except Exception as e:
            logger.warning(f"Could not load performance metrics: {e}")
    
    def record_superp_signal(self, coin: str, direction: str, entry_price: float,
                           psc_ratio: float, confidence: float, leverage: float,
                           target_price: float = None) -> str:
        """Record a Superp signal as a prediction for validation"""
        try:
            # Calculate target if not provided (higher target for Superp due to leverage)
            target_multiplier = min(1.0 + (leverage * 0.002), 1.05)  # Max 5% target
            
            if target_price is None:
                if direction.upper() in ['LONG', 'BUY']:
                    target_price = entry_price * target_multiplier
                    stop_loss = entry_price * 0.95  # 5% stop for Superp
                else:
                    target_price = entry_price * (2 - target_multiplier)
                    stop_loss = entry_price * 1.05
            else:
                if direction.upper() in ['LONG', 'BUY']:
                    stop_loss = entry_price * 0.95
                else:
                    stop_loss = entry_price * 1.05
            
            # Create ml_features dictionary with all the SuperP data
            ml_features = {
                'direction': direction,
                'entry_price': entry_price,
                'target_price': target_price,
                'stop_loss': stop_loss,
                'psc_ratio': psc_ratio,
                'leverage': leverage,
                'signal_strength': "SUPERP_SIGNAL",
                'market_conditions': "superp_trading",
                'expected_profit_pct': (target_multiplier - 1) * 100,
                'signal_type': "SUPERP"
            }
            
            return self.log_prediction(
                signal_id=f"SUPERP_{coin}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                coin=coin,
                prediction_type="SUPERP",
                predicted_price=target_price,  # Use target price as predicted price
                confidence=confidence,
                prediction_horizon="10min",  # SuperP typical horizon
                ml_features=ml_features,
                model_version="superp_v1.0"
            )
        except Exception as e:
            logger.error(f"Error recording Superp signal: {e}")
            return ""

    def record_prediction(self, coin: str = None, direction: str = None, confidence: float = None, 
                         entry_price: float = None, target_price: float = None, stop_loss: float = None,
                         psc_ratio: float = 1.0, ml_prediction_value: float = 0.5,
                         signal_strength: str = "unknown", market_conditions: str = "unknown",
                         expected_profit_pct: float = 0.0, prediction_id: str = None,
                         signal_type: str = "ML", leverage: float = 1.0, 
                         prediction_data: Dict = None) -> str:
        """
        Record a new prediction for validation tracking
        Supports both new parameter format and legacy prediction_data dict
        
        Args:
            signal_type: Type of signal ("ML", "PSC", "SUPERP", "HYBRID")
            leverage: Leverage used for the position (for PSC/Superp signals)
            prediction_data: Legacy dict format for backward compatibility
        """
        try:
            # Handle legacy prediction_data dict format
            if prediction_data is not None:
                coin = prediction_data.get('coin', 'UNKNOWN')
                direction = prediction_data.get('direction', prediction_data.get('recommendation', 'HOLD'))
                confidence = prediction_data.get('confidence', 0.0)
                entry_price = prediction_data.get('entry_price', 0.0)
                target_price = prediction_data.get('target_price', 0.0)
                expected_profit_pct = prediction_data.get('expected_return', 0.0)
                signal_strength = prediction_data.get('signal_strength', 'MODERATE')
                signal_type = prediction_data.get('signal_type', 'ML')
                leverage = prediction_data.get('leverage', 1.0)
            
            # Generate prediction ID
            if prediction_id is None:
                prediction_id = f"{signal_type.lower()}_{coin}_{int(datetime.now().timestamp())}"
            
            # Default values
            if not coin:
                coin = "UNKNOWN"
            if not direction:
                direction = "HOLD"
            if confidence is None:
                confidence = 0.0
            if not target_price and entry_price:
                target_price = entry_price * 1.001  # Default 0.1% target
            
            # Use log_prediction method for database storage
            return self.log_prediction(
                signal_id=prediction_id,
                coin=coin,
                prediction_type=signal_type,
                predicted_price=target_price or entry_price or 0.0,
                confidence=confidence,
                prediction_horizon="10min",
                ml_features={
                    'direction': direction,
                    'entry_price': entry_price,
                    'target_price': target_price,
                    'stop_loss': stop_loss,
                    'psc_ratio': psc_ratio,
                    'leverage': leverage,
                    'signal_strength': signal_strength,
                    'market_conditions': market_conditions,
                    'expected_profit_pct': expected_profit_pct,
                    'signal_type': signal_type
                },
                model_version=f"{signal_type.lower()}_v1.0"
            )
            
        except Exception as e:
            logger.error(f"Error recording prediction: {e}")
            return ""

    def export_validation_report(self) -> str:
        """Export comprehensive validation report"""
        if not self.data_manager:
            return "Database not available"
            
        try:
            # Export validation data to CSV
            csv_file = self.data_manager.db.export_to_csv('validation')
            
            # Generate summary report
            summary = self.get_performance_summary()
            
            report = f"""
# ML Prediction Validation Report
Generated: {datetime.now().isoformat()}

## Summary Statistics
- Total Predictions: {summary['total_predictions']}
- Validated Predictions: {summary['validated_predictions']}
- Pending Validations: {summary['pending_validations']}
- Average Accuracy: {summary['accuracy_rate']:.2f}%
- Profitable Rate: {summary['profitable_rate']:.2f}%
- Average Return: {summary['avg_return']:.2f}%
- Total Profit: ${summary['total_profit']:.2f}

## Data Export
Validation data exported to: {csv_file}

## Recommendations
"""
            
            if summary['accuracy_rate'] > 80:
                report += "✅ Model accuracy is excellent (>80%)\n"
            elif summary['accuracy_rate'] > 60:
                report += "⚠️ Model accuracy is acceptable (60-80%) - consider improvements\n"
            else:
                report += "❌ Model accuracy is low (<60%) - immediate improvements needed\n"
                
            if summary['profitable_rate'] > 70:
                report += "✅ High profitable trade rate (>70%)\n"
            else:
                report += "⚠️ Consider improving trade selection criteria\n"
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Failed to export validation report: {e}")
            return f"Error generating report: {e}"

# Compatibility wrapper for existing code
class EnhancedPredictionValidator(DatabasePredictionValidator):
    """Compatibility wrapper that maintains the original interface"""
    
    def __init__(self, project_root: Path):
        # Initialize with database integration
        super().__init__(project_root)
        
        # Legacy compatibility - create data directory
        self.data_dir = project_root / "data"
        self.data_dir.mkdir(exist_ok=True)
        
        logger.info("🔄 Legacy prediction validator upgraded to database integration")
