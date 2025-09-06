"""
Enhanced Prediction Validation System for PSC TON Trading

This module provides comprehensive prediction tracking and validation:
- Real-time prediction accuracy monitoring
- Automated outcome validation after trades close
- Performance analytics and improvement recommendations
- Detailed reporting for ML model optimization
"""

import csv
import json
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import logging
import pandas as pd
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class EnhancedPredictionValidator:
    """
    Advanced prediction validation system that tracks all ML predictions
    and validates them against actual market outcomes
    """
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.data_dir = project_root / "data"
        
        # Validation tracking files
        self.predictions_file = self.data_dir / "ml_predictions.csv"
        self.validation_file = self.data_dir / "prediction_validation.csv"
        self.performance_file = self.data_dir / "ml_performance_tracking.csv"
        
        # In-memory tracking
        self.pending_validations = []
        self.performance_metrics = {
            'total_predictions': 0,
            'validated_predictions': 0,
            'accuracy_rate': 0.0,
            'profitable_rate': 0.0,
            'avg_return': 0.0,
            'best_confidence_threshold': 0.6
        }
        
        self.setup_validation_files()
        self.load_performance_metrics()
        
        logger.info("🔍 Enhanced Prediction Validator initialized")
    
    def setup_validation_files(self):
        """Setup CSV files for validation tracking"""
        
        # Enhanced ML Predictions tracking
        if not self.predictions_file.exists():
            with open(self.predictions_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'prediction_id', 'coin', 'direction', 'confidence',
                    'entry_price', 'target_price', 'expected_return', 'signal_strength',
                    'market_sentiment', 'model_version', 'features_used'
                ])
        
        # Validation results tracking
        if not self.validation_file.exists():
            with open(self.validation_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'prediction_id', 'timestamp', 'coin', 'direction', 'confidence',
                    'entry_price', 'exit_price', 'actual_return', 'profitable',
                    'accuracy_score', 'validation_time', 'time_to_outcome'
                ])
        
        # Performance metrics tracking
        if not self.performance_file.exists():
            with open(self.performance_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'date', 'total_predictions', 'validated_count', 'accuracy_rate',
                    'profitable_rate', 'avg_return', 'best_confidence_threshold',
                    'model_improvements'
                ])
    
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
            
            timestamp = datetime.now().isoformat()
            
            # Enhanced prediction data with signal type
            pred_data = {
                'timestamp': timestamp,
                'coin': coin,
                'prediction_id': prediction_id,
                'direction': direction,
                'confidence': confidence,
                'entry_price': entry_price,
                'target_price': target_price,
                'stop_loss': stop_loss if stop_loss else 0.0,
                'psc_ratio': psc_ratio,
                'ml_prediction_value': ml_prediction_value,
                'signal_strength': signal_strength,
                'market_conditions': market_conditions,
                'expected_profit_pct': expected_profit_pct,
                'signal_type': signal_type,
                'leverage': leverage,
                'tradingview_sentiment': json.dumps({
                    "psc_ratio": psc_ratio,
                    "ml_prediction": ml_prediction_value,
                    "signal_strength": signal_strength,
                    "leverage": leverage,
                    "signal_type": signal_type
                })
            }
            
            # Add to pending validations
            self.pending_validations.append({
                'prediction_id': prediction_id,
                'coin': coin,
                'direction': direction,
                'entry_time': timestamp,
                'entry_price': entry_price,
                'target_price': target_price,
                'stop_loss': stop_loss if stop_loss else 0.0,
                'confidence': confidence,
                'signal_type': signal_type,
                'leverage': leverage
            })
            
            # Log to CSV using new format
            self._write_prediction_to_csv(pred_data)
            
            # Update performance metrics
            self.performance_metrics['total_predictions'] += 1
            self._save_performance_metrics()
            
            logger.info(f"✅ {signal_type} prediction recorded: {prediction_id} for {coin} {direction}")
            return prediction_id
            
        except Exception as e:
            logger.error(f"❌ Error recording prediction: {e}")
            return ""
    
    def _write_prediction_to_csv(self, prediction_data: Dict):
        """Write prediction data to CSV file"""
        try:
            with open(self.predictions_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    prediction_data['timestamp'],
                    prediction_data['coin'], 
                    prediction_data['prediction_id'],
                    prediction_data['direction'],
                    prediction_data['confidence'],
                    prediction_data['entry_price'],
                    prediction_data['target_price'],
                    prediction_data['stop_loss'],
                    prediction_data['psc_ratio'],
                    prediction_data['ml_prediction_value'],
                    prediction_data['signal_strength'],
                    prediction_data['market_conditions'],
                    prediction_data['tradingview_sentiment'],
                    prediction_data['expected_profit_pct']
                ])
        except Exception as e:
            logger.error(f"Error writing prediction to CSV: {e}")
    
    def record_legacy_prediction(self, prediction_data: Dict):
        """Legacy method - calls new record_prediction with dict"""
        return self.record_prediction(prediction_data=prediction_data)
    
    def record_psc_signal(self, coin: str, direction: str, entry_price: float, 
                         psc_ratio: float, confidence: float, target_price: float = None,
                         leverage: float = 1.0) -> str:
        """Record a PSC signal as a prediction for validation"""
        try:
            # Calculate target if not provided (2% default for PSC)
            if target_price is None:
                if direction.upper() in ['LONG', 'BUY']:
                    target_price = entry_price * 1.02
                    stop_loss = entry_price * 0.98
                else:
                    target_price = entry_price * 0.98
                    stop_loss = entry_price * 1.02
            else:
                # Calculate stop loss opposite to target
                if direction.upper() in ['LONG', 'BUY']:
                    stop_loss = entry_price * 0.98
                else:
                    stop_loss = entry_price * 1.02
            
            return self.record_prediction(
                coin=coin,
                direction=direction,
                confidence=confidence,
                entry_price=entry_price,
                target_price=target_price,
                stop_loss=stop_loss,
                psc_ratio=psc_ratio,
                ml_prediction_value=min(confidence * psc_ratio / 2.0, 1.0),  # Convert to ML equivalent
                signal_strength="PSC_SIGNAL",
                market_conditions="psc_trading",
                expected_profit_pct=2.0,  # Standard PSC target
                signal_type="PSC",
                leverage=leverage
            )
        except Exception as e:
            logger.error(f"Error recording PSC signal: {e}")
            return ""
    
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
            
            return self.record_prediction(
                coin=coin,
                direction=direction,
                confidence=confidence,
                entry_price=entry_price,
                target_price=target_price,
                stop_loss=stop_loss,
                psc_ratio=psc_ratio,
                ml_prediction_value=min(confidence * psc_ratio / 3.0, 1.0),  # Conservative for Superp
                signal_strength="SUPERP_SIGNAL",
                market_conditions="superp_trading",
                expected_profit_pct=(target_multiplier - 1) * 100,
                signal_type="SUPERP",
                leverage=leverage
            )
        except Exception as e:
            logger.error(f"Error recording Superp signal: {e}")
            return ""
    
    def validate_live_trade(self, trade_data: Dict) -> bool:
        """Validate a completed live trade against its prediction"""
        try:
            # Extract trade details
            coin = trade_data.get('coin', '')
            entry_price = float(trade_data.get('entry_price', 0))
            exit_price = float(trade_data.get('exit_price', 0))
            direction = trade_data.get('direction', '')
            successful = trade_data.get('successful', False)
            profit_pct = float(trade_data.get('profit_pct', 0))
            timestamp = trade_data.get('timestamp', datetime.now().isoformat())
            
            # Try to find matching prediction
            prediction_id = None
            for pending in self.pending_validations:
                if (pending['coin'] == coin and 
                    abs(pending['entry_price'] - entry_price) < 0.01 and
                    pending['direction'].upper() in [direction.upper(), 
                                                   'BUY' if direction.upper() == 'LONG' else 'SELL']):
                    prediction_id = pending['prediction_id']
                    break
            
            # If no direct match, create a validation for this trade
            if not prediction_id:
                # Generate a prediction ID based on trade data
                trade_timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).timestamp()
                prediction_id = f"trade_{coin}_{int(trade_timestamp)}"
                
                logger.info(f"Creating retroactive prediction validation for trade: {prediction_id}")
            
            # Determine outcome
            outcome = "SUCCESS" if successful else "FAILURE"
            
            # Validate the prediction
            return self.validate_prediction(
                prediction_id=prediction_id,
                actual_exit_price=exit_price,
                outcome=outcome,
                exit_time=timestamp,
                notes=f"Live trade validation: {profit_pct:.2f}% profit"
            )
            
        except Exception as e:
            logger.error(f"Error validating live trade: {e}")
            return False
    
    def validate_prediction(self, prediction_id: str, actual_outcome: Dict):
        """Validate a prediction against actual market outcome"""
        try:
            # Find pending validation
            pending = None
            for i, p in enumerate(self.pending_validations):
                if p['prediction_id'] == prediction_id:
                    pending = self.pending_validations.pop(i)
                    break
            
            if not pending:
                logger.warning(f"No pending validation found for {prediction_id}")
                return False
            
            # Extract actual outcome
            exit_price = actual_outcome.get('exit_price', 0.0)
            actual_return = actual_outcome.get('return', 0.0)
            profitable = actual_outcome.get('profitable', False)
            
            # Calculate accuracy
            predicted_profitable = pending['expected_return'] > 0
            accuracy_score = 1.0 if predicted_profitable == profitable else 0.0
            
            # Time tracking
            validation_time = datetime.now().isoformat()
            entry_time = datetime.fromisoformat(pending['timestamp'])
            time_to_outcome = (datetime.now() - entry_time).total_seconds() / 60  # minutes
            
            # Record validation result
            with open(self.validation_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    prediction_id, pending['timestamp'], pending['coin'], 
                    pending['direction'], pending['confidence'], pending['entry_price'],
                    exit_price, actual_return, profitable, accuracy_score,
                    validation_time, time_to_outcome
                ])
            
            # Update performance metrics
            self.performance_metrics['validated_predictions'] += 1
            self.update_performance_metrics()
            
            logger.info(f"✅ Prediction validated: {prediction_id} - Accuracy: {accuracy_score}")
            return True
            
        except Exception as e:
            logger.error(f"Error validating prediction {prediction_id}: {e}")
            return False
    
    def update_performance_metrics(self):
        """Update overall performance metrics"""
        try:
            # Load validation data
            if not self.validation_file.exists():
                return
            
            df = pd.read_csv(self.validation_file)
            if len(df) == 0:
                return
            
            # Calculate metrics
            total_validated = len(df)
            accuracy_rate = df['accuracy_score'].mean()
            profitable_rate = df['profitable'].mean()
            avg_return = df['actual_return'].mean()
            
            # Find best confidence threshold
            best_threshold = 0.6
            best_accuracy = 0.0
            for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
                high_conf = df[df['confidence'] >= threshold]
                if len(high_conf) > 5:  # Need at least 5 samples
                    threshold_accuracy = high_conf['accuracy_score'].mean()
                    if threshold_accuracy > best_accuracy:
                        best_accuracy = threshold_accuracy
                        best_threshold = threshold
            
            # Update metrics
            self.performance_metrics.update({
                'validated_predictions': total_validated,
                'accuracy_rate': accuracy_rate,
                'profitable_rate': profitable_rate,
                'avg_return': avg_return,
                'best_confidence_threshold': best_threshold
            })
            
            # Save daily performance
            today = datetime.now().strftime('%Y-%m-%d')
            with open(self.performance_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    today, self.performance_metrics['total_predictions'],
                    total_validated, accuracy_rate, profitable_rate, avg_return,
                    best_threshold, "auto_update"
                ])
            
            logger.info(f"📊 Performance updated: {accuracy_rate:.1%} accuracy, {profitable_rate:.1%} profitable")
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def get_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        try:
            self.update_performance_metrics()
            
            report = {
                'summary': self.performance_metrics.copy(),
                'recommendations': [],
                'recent_performance': {},
                'confidence_analysis': {}
            }
            
            # Load recent data for analysis
            if self.validation_file.exists():
                df = pd.read_csv(self.validation_file)
                
                if len(df) >= 10:
                    # Recent performance (last 30 predictions)
                    recent = df.tail(30)
                    report['recent_performance'] = {
                        'accuracy': recent['accuracy_score'].mean(),
                        'profitability': recent['profitable'].mean(),
                        'avg_return': recent['actual_return'].mean(),
                        'prediction_count': len(recent)
                    }
                    
                    # Confidence analysis
                    high_conf = df[df['confidence'] >= 0.7]
                    low_conf = df[df['confidence'] < 0.6]
                    
                    report['confidence_analysis'] = {
                        'high_confidence': {
                            'count': len(high_conf),
                            'accuracy': high_conf['accuracy_score'].mean() if len(high_conf) > 0 else 0.0,
                            'profitability': high_conf['profitable'].mean() if len(high_conf) > 0 else 0.0
                        },
                        'low_confidence': {
                            'count': len(low_conf),
                            'accuracy': low_conf['accuracy_score'].mean() if len(low_conf) > 0 else 0.0,
                            'profitability': low_conf['profitable'].mean() if len(low_conf) > 0 else 0.0
                        }
                    }
                    
                    # Generate recommendations
                    report['recommendations'] = self.generate_recommendations(df)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {'error': str(e)}
    
    def generate_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate improvement recommendations based on performance"""
        recommendations = []
        
        accuracy = df['accuracy_score'].mean()
        profitability = df['profitable'].mean()
        
        if accuracy < 0.4:
            recommendations.append("🔴 Low accuracy detected. Consider retraining models with more recent data.")
        elif accuracy < 0.6:
            recommendations.append("🟡 Moderate accuracy. Fine-tune confidence thresholds and feature selection.")
        else:
            recommendations.append("🟢 Good accuracy performance. Continue current strategy.")
        
        if profitability < 0.4:
            recommendations.append("💰 Low profitability. Review risk management and position sizing.")
        
        # High confidence analysis
        high_conf = df[df['confidence'] >= 0.7]
        if len(high_conf) > 5:
            high_acc = high_conf['accuracy_score'].mean()
            if high_acc > accuracy + 0.1:
                recommendations.append(f"⭐ High-confidence predictions perform {high_acc:.1%} better. Consider raising minimum confidence threshold.")
        
        # Recent performance trend
        if len(df) >= 20:
            recent = df.tail(10)
            earlier = df.tail(20).head(10)
            
            recent_acc = recent['accuracy_score'].mean()
            earlier_acc = earlier['accuracy_score'].mean()
            
            if recent_acc < earlier_acc - 0.1:
                recommendations.append("📉 Performance declining. Models may need retraining.")
            elif recent_acc > earlier_acc + 0.1:
                recommendations.append("📈 Performance improving. Current model updates are effective.")
        
        return recommendations
    
    def load_performance_metrics(self):
        """Load existing performance metrics"""
        try:
            if self.performance_file.exists():
                df = pd.read_csv(self.performance_file)
                if len(df) > 0:
                    latest = df.iloc[-1]
                    self.performance_metrics.update({
                        'total_predictions': int(latest['total_predictions']),
                        'validated_predictions': int(latest['validated_count']),
                        'accuracy_rate': float(latest['accuracy_rate']),
                        'profitable_rate': float(latest['profitable_rate']),
                        'avg_return': float(latest['avg_return']),
                        'best_confidence_threshold': float(latest['best_confidence_threshold'])
                    })
        except Exception as e:
            logger.warning(f"Could not load performance metrics: {e}")

    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old validation data to prevent files from growing too large"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Clean validation file
            if self.validation_file.exists():
                df = pd.read_csv(self.validation_file)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                recent_df = df[df['timestamp'] > cutoff_date]
                recent_df.to_csv(self.validation_file, index=False)
                
                removed_count = len(df) - len(recent_df)
                if removed_count > 0:
                    logger.info(f"🧹 Cleaned up {removed_count} old validation records")
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")

    def _save_performance_metrics(self):
        """Save current performance metrics to file"""
        try:
            metrics_file = self.project_root / "data" / "performance_metrics.json"
            metrics_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(metrics_file, 'w') as f:
                json.dump(self.performance_metrics, f, indent=2, default=str)
                
        except Exception as e:
            logger.warning(f"Could not save performance metrics: {e}")
