#!/usr/bin/env python3
"""
ML Prediction Optimization - Reduce Database Load
Only store high-quality predictions to prevent database bloat
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class PredictionOptimizer:
    """Optimizes ML predictions to reduce database load while maintaining learning quality"""
    
    def __init__(self, min_confidence=0.20, min_expected_return=0.001, dedup_window_minutes=5):
        self.min_confidence = min_confidence
        self.min_expected_return = min_expected_return
        self.dedup_window_minutes = dedup_window_minutes
        self.recent_predictions = {}  # coin -> list of recent predictions
        
    def should_store_prediction(self, prediction: Dict) -> bool:
        """
        Determine if a prediction should be stored in database
        
        Criteria for storage:
        1. Confidence above minimum threshold
        2. Expected return above minimum threshold
        3. Not duplicate of recent prediction for same coin
        4. Signal strength indicates actionable opportunity
        """
        coin = prediction.get('symbol', 'UNKNOWN')
        confidence = prediction.get('confidence', 0.0)
        expected_return = abs(prediction.get('expected_return', 0.0))
        signal = prediction.get('signal', 'HOLD')
        
        # Filter 1: Minimum confidence threshold
        if confidence < self.min_confidence:
            logger.debug(f"🗑️ Skipping {coin} prediction - low confidence: {confidence:.3f}")
            return False
            
        # Filter 2: Minimum expected return threshold
        if expected_return < self.min_expected_return:
            logger.debug(f"🗑️ Skipping {coin} prediction - low return: {expected_return:.4f}")
            return False
            
        # Filter 3: Skip HOLD signals (not actionable)
        if signal == 'HOLD':
            logger.debug(f"🗑️ Skipping {coin} prediction - HOLD signal")
            return False
            
        # Filter 4: Deduplication - avoid storing similar predictions for same coin
        if self._is_duplicate_prediction(coin, prediction):
            logger.debug(f"🗑️ Skipping {coin} prediction - duplicate within {self.dedup_window_minutes}min")
            return False
            
        # Passed all filters - store this prediction
        self._add_to_recent(coin, prediction)
        logger.info(f"✅ Storing {coin} prediction - Conf: {confidence:.3f}, Return: {expected_return:.4f}")
        return True
        
    def _is_duplicate_prediction(self, coin: str, prediction: Dict) -> bool:
        """Check if this prediction is too similar to recent ones for the same coin"""
        if coin not in self.recent_predictions:
            return False
            
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(minutes=self.dedup_window_minutes)
        
        # Clean old predictions
        self.recent_predictions[coin] = [
            p for p in self.recent_predictions[coin] 
            if p['timestamp'] > cutoff_time
        ]
        
        # Check for similar predictions
        current_signal = prediction.get('signal', 'HOLD')
        current_confidence = prediction.get('confidence', 0.0)
        
        for recent in self.recent_predictions[coin]:
            # Same signal direction and similar confidence (within 5%)
            if (recent['signal'] == current_signal and 
                abs(recent['confidence'] - current_confidence) < 0.05):
                return True
                
        return False
        
    def _add_to_recent(self, coin: str, prediction: Dict):
        """Add prediction to recent predictions cache"""
        if coin not in self.recent_predictions:
            self.recent_predictions[coin] = []
            
        self.recent_predictions[coin].append({
            'timestamp': datetime.now(),
            'signal': prediction.get('signal', 'HOLD'),
            'confidence': prediction.get('confidence', 0.0)
        })
        
    def get_stats(self) -> Dict:
        """Get optimization statistics"""
        total_coins = len(self.recent_predictions)
        total_recent = sum(len(predictions) for predictions in self.recent_predictions.values())
        
        return {
            'coins_tracked': total_coins,
            'recent_predictions': total_recent,
            'min_confidence': self.min_confidence,
            'min_expected_return': self.min_expected_return,
            'dedup_window_minutes': self.dedup_window_minutes
        }

# Global optimizer instance
prediction_optimizer = PredictionOptimizer(
    min_confidence=0.25,  # Only store predictions with 25%+ confidence  
    min_expected_return=0.002,  # Only store predictions with 0.2%+ expected return
    dedup_window_minutes=5  # Avoid duplicates within 5 minutes
)