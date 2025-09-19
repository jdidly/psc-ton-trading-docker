"""
🎯 Integrated Signal Processor - Enhanced Accuracy System
Integrates all components (PSC, ML, TradingView, Microstructure) for maximum prediction accuracy
FULLY INTEGRATED WITH PSC DATABASE SYSTEM
"""

import asyncio
import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ComponentSignal:
    """Individual component signal data"""
    component: str
    confidence: float
    direction: str
    strength: float
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class IntegratedSignal:
    """Final integrated signal with all component data"""
    coin: str
    direction: str
    confidence: float
    consensus_strength: float
    entry_price: float
    target_price: float
    components: Dict[str, ComponentSignal]
    integration_timestamp: datetime
    prediction_id: str

class AccuracyOptimizer:
    """Real-time accuracy optimization and component weight management with database integration"""
    
    def __init__(self, data_manager=None):
        self.data_manager = data_manager  # PSC Database integration
        
        self.component_weights = {
            'psc': 0.35,           # PSC base signal
            'ml': 0.30,            # ML prediction
            'tradingview': 0.25,   # Technical analysis
            'microstructure': 0.10 # Microstructure (lighter weight initially)
        }
        self.accuracy_history = {}
        self.performance_history = {}
        self.min_samples = 20  # Minimum samples before optimization
        self.optimization_interval = 50  # Reoptimize every 50 predictions
        self.prediction_count = 0
        
        # Initialize component tracking
        for component in self.component_weights.keys():
            self.accuracy_history[component] = []
            self.performance_history[component] = []
        
        # Load weights from database if available
        self.load_weights()
        
        logger.info(f"🔧 AccuracyOptimizer initialized with database integration: {'✅' if data_manager else '❌'}")
    
    def load_weights(self):
        """Load component weights from database"""
        if not self.data_manager:
            return
        
        try:
            # For now, use a simple system event query
            # In the future, could create a dedicated weights table
            logger.info("📊 Component weights loaded from defaults (database ready for persistence)")
        except Exception as e:
            logger.warning(f"Could not load weights from database: {e}")
    
    def save_weights(self):
        """Save component weights to database"""
        if not self.data_manager:
            return
        
        try:
            # Save weights as system event for persistence
            self.data_manager.db.log_system_event(
                event_type='ACCURACY_WEIGHTS',
                component='INTEGRATED_PROCESSOR',
                message='Component weights updated by accuracy optimizer',
                details={
                    'component_weights': self.component_weights.copy(),
                    'prediction_count': self.prediction_count,
                    'accuracy_history_sizes': {k: len(v) for k, v in self.accuracy_history.items()}
                },
                severity='INFO'
            )
            logger.info(f"💾 Saved component weights to database")
        except Exception as e:
            logger.error(f"Failed to save weights to database: {e}")
    
    def update_component_accuracy(self, prediction_id: str, components: Dict[str, ComponentSignal], outcome: bool):
        """Update accuracy metrics for each component based on actual outcome"""
        try:
            for component_name, signal in components.items():
                if component_name not in self.accuracy_history:
                    self.accuracy_history[component_name] = []
                
                # Calculate component accuracy score
                direction_correct = (
                    (signal.direction == "LONG" and outcome) or
                    (signal.direction == "SHORT" and outcome)
                )
                
                # Weight by confidence (higher confidence predictions matter more)
                accuracy_score = signal.confidence if direction_correct else (1.0 - signal.confidence)
                
                self.accuracy_history[component_name].append(accuracy_score)
                
                # Keep rolling window of last 100 predictions
                if len(self.accuracy_history[component_name]) > 100:
                    self.accuracy_history[component_name] = self.accuracy_history[component_name][-100:]
            
            self.prediction_count += 1
            
            # Optimize weights periodically
            if self.prediction_count % self.optimization_interval == 0:
                self.optimize_weights()
                
        except Exception as e:
            logger.error(f"Error updating component accuracy: {e}")
    
    def optimize_weights(self):
        """Dynamically optimize component weights based on recent performance"""
        try:
            if not all(len(self.accuracy_history.get(comp, [])) >= self.min_samples 
                      for comp in self.component_weights.keys()):
                return  # Need minimum samples
            
            # Calculate recent accuracy for each component
            recent_accuracy = {}
            for component in self.component_weights.keys():
                if component in self.accuracy_history:
                    recent_scores = self.accuracy_history[component][-self.min_samples:]
                    recent_accuracy[component] = np.mean(recent_scores)
                else:
                    recent_accuracy[component] = 0.5  # Default neutral
            
            # Apply performance-based weight adjustment
            total_accuracy = sum(recent_accuracy.values())
            if total_accuracy > 0:
                for component in self.component_weights.keys():
                    # Smooth weight adjustment (don't change too dramatically)
                    new_weight = recent_accuracy[component] / total_accuracy
                    self.component_weights[component] = (
                        0.7 * self.component_weights[component] + 0.3 * new_weight
                    )
                
                # Ensure weights sum to 1.0
                total_weight = sum(self.component_weights.values())
                for component in self.component_weights.keys():
                    self.component_weights[component] /= total_weight
            
            logger.info(f"🔧 Optimized component weights: {self.component_weights}")
            
        except Exception as e:
            logger.error(f"Error optimizing weights: {e}")
    
    def get_component_weights(self) -> Dict[str, float]:
        """Get current component weights"""
        return self.component_weights.copy()

class IntegratedSignalProcessor:
    """
    Main integrated signal processor that combines all system components
    for maximum accuracy without breaking existing functionality
    """
    
    def __init__(self, trading_bot):
        self.trading_bot = trading_bot
        
        # Database integration for enhanced accuracy tracking
        self.data_manager = trading_bot.data_manager if trading_bot else None
        
        # Initialize accuracy optimizer with database persistence
        self.accuracy_optimizer = AccuracyOptimizer(data_manager=self.data_manager)
        
        self.min_confidence_threshold = 0.65  # Higher threshold for integrated signals
        self.min_consensus_threshold = 0.75   # Require 75% component agreement
        self.integration_enabled = True
        
        # Component availability flags
        self.components_available = {
            'psc': True,  # Always available (core system)
            'ml': hasattr(trading_bot, 'ml_engine') and trading_bot.ml_engine is not None,
            'tradingview': hasattr(trading_bot, 'tradingview') and trading_bot.tradingview is not None,
            'microstructure': hasattr(trading_bot, 'ml_microstructure_trainer') and trading_bot.ml_microstructure_trainer is not None
        }
        
        # Log database integration status
        available_components = [k for k, v in self.components_available.items() if v]
        logger.info(f"🎯 Integrated Signal Processor initialized:")
        logger.info(f"   Available components: {available_components}")
        logger.info(f"   Database integration: {'✅ ENABLED' if self.data_manager else '❌ DISABLED'}")
        logger.info(f"   Enhanced accuracy mode: {'✅ ACTIVE' if len(available_components) >= 2 else '⚠️ LIMITED'}")
    
    async def process_integrated_signal(self, coin: str, current_price: float, psc_ratio: float, psc_confidence: float, psc_direction: str) -> Optional[IntegratedSignal]:
        """
        Process a complete integrated signal using all available components
        Falls back gracefully if components are unavailable
        """
        try:
            if not self.integration_enabled:
                return None  # Integration disabled
            
            components = {}
            
            # COMPONENT 1: PSC Analysis (Always available)
            psc_signal = ComponentSignal(
                component="psc",
                confidence=psc_confidence,
                direction=psc_direction,
                strength=min(psc_ratio / 10.0, 1.0),  # Normalize ratio to 0-1
                timestamp=datetime.now(),
                metadata={"ratio": psc_ratio, "price": current_price}
            )
            components["psc"] = psc_signal
            
            # COMPONENT 2: ML Engine Enhancement
            if self.components_available['ml']:
                ml_signal = await self._get_ml_signal(coin, current_price, psc_ratio)
                if ml_signal:
                    components["ml"] = ml_signal
            
            # COMPONENT 3: TradingView Technical Analysis
            if self.components_available['tradingview']:
                tv_signal = await self._get_tradingview_signal(coin, psc_direction, psc_confidence)
                if tv_signal:
                    components["tradingview"] = tv_signal
            
            # COMPONENT 4: Microstructure Analysis
            if self.components_available['microstructure']:
                micro_signal = await self._get_microstructure_signal(coin, current_price)
                if micro_signal:
                    components["microstructure"] = micro_signal
            
            # Validate consensus and calculate integrated confidence
            consensus_result = self._validate_consensus(components)
            if not consensus_result:
                logger.info(f"❌ {coin}: Signal rejected due to poor consensus")
                return None
            
            direction, consensus_strength, integrated_confidence = consensus_result
            
            # Quality gate: Only proceed with high-confidence signals
            if integrated_confidence < self.min_confidence_threshold:
                logger.info(f"❌ {coin}: Signal rejected due to low confidence ({integrated_confidence:.1%})")
                return None
            
            # Calculate target price based on integrated analysis
            target_price = self._calculate_target_price(current_price, direction, integrated_confidence, components)
            
            # Create integrated signal
            integrated_signal = IntegratedSignal(
                coin=coin,
                direction=direction,
                confidence=integrated_confidence,
                consensus_strength=consensus_strength,
                entry_price=current_price,
                target_price=target_price,
                components=components,
                integration_timestamp=datetime.now(),
                prediction_id=f"INTEGRATED_{coin}_{int(datetime.now().timestamp())}"
            )
            
            # Record prediction for validation (Enhanced with Database Integration)
            if hasattr(self.trading_bot, 'prediction_validator') and self.trading_bot.prediction_validator:
                try:
                    self.trading_bot.prediction_validator.record_prediction(
                        coin=coin,
                        direction=direction,
                        confidence=integrated_confidence,
                        entry_price=current_price,
                        target_price=target_price,
                        signal_type="INTEGRATED",
                        prediction_id=integrated_signal.prediction_id,
                        prediction_data={
                            "consensus_strength": consensus_strength,
                            "components": {k: {"confidence": v.confidence, "direction": v.direction} 
                                         for k, v in components.items()},
                            "integration_timestamp": datetime.now().isoformat()
                        }
                    )
                except Exception as e:
                    logger.error(f"Error recording integrated prediction: {e}")
            
            # Save integrated signal to database
            self.save_integrated_signal(integrated_signal)
            
            logger.info(f"✅ HIGH-QUALITY INTEGRATED Signal: {coin} {direction} "
                       f"Confidence: {integrated_confidence:.1%} "
                       f"Consensus: {consensus_strength:.1%} "
                       f"Components: {list(components.keys())}")
            
            return integrated_signal
            
        except Exception as e:
            logger.error(f"Error processing integrated signal for {coin}: {e}")
            return None
    
    async def _get_ml_signal(self, coin: str, current_price: float, psc_ratio: float) -> Optional[ComponentSignal]:
        """Get ML engine signal"""
        try:
            if not self.trading_bot.ml_engine:
                return None
            
            # Use ML engine to predict trade outcome
            ml_result = self.trading_bot.ml_engine.predict_trade_outcome(
                psc_price=current_price,
                ton_price=1.0,  # Normalized
                ratio=psc_ratio
            )
            
            if ml_result:
                direction = "LONG" if ml_result.get('expected_return', 0) > 0 else "SHORT"
                confidence = ml_result.get('confidence', 0.5)
                
                return ComponentSignal(
                    component="ml",
                    confidence=confidence,
                    direction=direction,
                    strength=abs(ml_result.get('expected_return', 0)) * 10,  # Amplify for strength
                    timestamp=datetime.now(),
                    metadata=ml_result
                )
        except Exception as e:
            logger.error(f"Error getting ML signal: {e}")
        return None
    
    async def _get_tradingview_signal(self, coin: str, psc_direction: str, psc_confidence: float) -> Optional[ComponentSignal]:
        """Get TradingView technical analysis signal"""
        try:
            if not self.trading_bot.tradingview:
                return None
            
            # Get single coin analysis (optimized for speed)
            tv_analysis = await self.trading_bot.tradingview.get_single_coin_analysis(coin)
            
            if tv_analysis and 'consensus' in tv_analysis:
                consensus = tv_analysis['consensus']
                tv_direction = consensus.get('direction', 'NEUTRAL')
                tv_confidence = consensus.get('confidence', 0.5)
                tv_strength = consensus.get('strength', 0.5)
                
                # Convert TradingView direction to our format
                if tv_direction in ['BUY', 'STRONG_BUY']:
                    direction = "LONG"
                elif tv_direction in ['SELL', 'STRONG_SELL']:
                    direction = "SHORT"
                else:
                    direction = "NEUTRAL"
                
                return ComponentSignal(
                    component="tradingview",
                    confidence=tv_confidence,
                    direction=direction,
                    strength=tv_strength,
                    timestamp=datetime.now(),
                    metadata=tv_analysis
                )
        except Exception as e:
            logger.error(f"Error getting TradingView signal: {e}")
        return None
    
    async def _get_microstructure_signal(self, coin: str, current_price: float) -> Optional[ComponentSignal]:
        """Get microstructure analysis signal"""
        try:
            if not self.trading_bot.ml_microstructure_trainer:
                return None
            
            # Placeholder for microstructure analysis
            # This would integrate with your ML microstructure system
            # For now, return a basic signal based on price momentum
            
            # Simple momentum-based signal (replace with actual microstructure analysis)
            momentum = 0.5  # Placeholder
            direction = "LONG" if momentum > 0 else "SHORT"
            confidence = min(abs(momentum) + 0.3, 0.8)  # Basic confidence
            
            return ComponentSignal(
                component="microstructure",
                confidence=confidence,
                direction=direction,
                strength=abs(momentum),
                timestamp=datetime.now(),
                metadata={"momentum": momentum, "price": current_price}
            )
        except Exception as e:
            logger.error(f"Error getting microstructure signal: {e}")
        return None
    
    def _validate_consensus(self, components: Dict[str, ComponentSignal]) -> Optional[Tuple[str, float, float]]:
        """
        Validate consensus across components and calculate integrated confidence
        Returns: (direction, consensus_strength, integrated_confidence) or None
        """
        try:
            if len(components) < 2:
                return None  # Need at least 2 components
            
            # Count direction votes
            direction_votes = {}
            total_weight = 0
            weights = self.accuracy_optimizer.get_component_weights()
            
            for comp_name, signal in components.items():
                weight = weights.get(comp_name, 0.25)  # Default weight
                
                if signal.direction not in direction_votes:
                    direction_votes[signal.direction] = 0
                direction_votes[signal.direction] += weight
                total_weight += weight
            
            # Find dominant direction
            if not direction_votes:
                return None
            
            dominant_direction = max(direction_votes, key=direction_votes.get)
            consensus_strength = direction_votes[dominant_direction] / total_weight
            
            # Require minimum consensus
            if consensus_strength < self.min_consensus_threshold:
                return None
            
            # Calculate weighted integrated confidence
            integrated_confidence = 0
            confidence_weight_sum = 0
            
            for comp_name, signal in components.items():
                if signal.direction == dominant_direction:  # Only count aligned components
                    weight = weights.get(comp_name, 0.25)
                    integrated_confidence += signal.confidence * weight
                    confidence_weight_sum += weight
            
            if confidence_weight_sum > 0:
                integrated_confidence /= confidence_weight_sum
            
            # Apply consensus bonus
            if consensus_strength > 0.9:  # Very strong consensus
                integrated_confidence *= 1.1
            elif consensus_strength > 0.8:  # Strong consensus
                integrated_confidence *= 1.05
            
            # Cap confidence
            integrated_confidence = min(0.95, integrated_confidence)
            
            return dominant_direction, consensus_strength, integrated_confidence
            
        except Exception as e:
            logger.error(f"Error validating consensus: {e}")
            return None
    
    def _calculate_target_price(self, entry_price: float, direction: str, confidence: float, components: Dict[str, ComponentSignal]) -> float:
        """Calculate target price based on integrated analysis"""
        try:
            # Base target percentage (conservative)
            base_target_pct = 0.002  # 0.2% base target
            
            # Adjust based on confidence
            confidence_multiplier = 1 + (confidence - 0.5) * 2  # 0.5 conf = 1x, 0.75 conf = 1.5x, etc.
            
            # Adjust based on component strength
            avg_strength = np.mean([signal.strength for signal in components.values()])
            strength_multiplier = 1 + avg_strength * 0.5  # Up to 1.5x based on strength
            
            # Calculate final target percentage
            target_pct = base_target_pct * confidence_multiplier * strength_multiplier
            target_pct = min(target_pct, 0.005)  # Cap at 0.5% for safety
            
            # Calculate target price
            if direction == "LONG":
                target_price = entry_price * (1 + target_pct)
            else:  # SHORT
                target_price = entry_price * (1 - target_pct)
            
            return target_price
            
        except Exception as e:
            logger.error(f"Error calculating target price: {e}")
            return entry_price * 1.002 if direction == "LONG" else entry_price * 0.998
    
    def update_prediction_outcome(self, prediction_id: str, success: bool):
        """Update prediction outcome for accuracy optimization"""
        try:
            # This would be called when a prediction is validated
            # For now, we'll track basic success/failure
            if hasattr(self, '_active_predictions'):
                if prediction_id in self._active_predictions:
                    components = self._active_predictions[prediction_id]
                    self.accuracy_optimizer.update_component_accuracy(
                        prediction_id, components, success
                    )
                    del self._active_predictions[prediction_id]
        except Exception as e:
            logger.error(f"Error updating prediction outcome: {e}")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get integrated system statistics"""
        return {
            "integration_enabled": self.integration_enabled,
            "available_components": {k: v for k, v in self.components_available.items() if v},
            "component_weights": self.accuracy_optimizer.get_component_weights(),
            "min_confidence_threshold": self.min_confidence_threshold,
            "min_consensus_threshold": self.min_consensus_threshold,
            "prediction_count": self.accuracy_optimizer.prediction_count,
            "accuracy_history_size": {k: len(v) for k, v in self.accuracy_optimizer.accuracy_history.items()}
        }
    
    def save_integrated_signal(self, integrated_signal: IntegratedSignal):
        """Save integrated signal to database with enhanced accuracy metadata"""
        if not self.data_manager:
            # Fallback to JSON file storage
            self._save_to_file(integrated_signal)
            return
        
        try:
            # Convert integrated signal to enhanced PSC signal with accuracy metadata
            signal_id = self.data_manager.log_psc_signal(
                coin=integrated_signal.symbol,
                price=integrated_signal.entry_price,
                ratio=0.0,  # Will be enriched with integrated data
                confidence=integrated_signal.confidence,
                direction=integrated_signal.direction,
                exit_estimate=integrated_signal.target_price,
                ml_prediction=integrated_signal.confidence,
                market_conditions="INTEGRATED_ANALYSIS",
                ml_features={
                    "signal_type": "INTEGRATED",
                    "prediction_id": integrated_signal.prediction_id,
                    "consensus_strength": integrated_signal.consensus_strength,
                    "component_count": len(integrated_signal.components),
                    "components": {
                        name: {
                            "confidence": comp.confidence,
                            "direction": comp.direction,
                            "component": comp.component if hasattr(comp, 'component') else name
                        }
                        for name, comp in integrated_signal.components.items()
                    },
                    "integration_metadata": integrated_signal.integration_metadata,
                    "component_weights": self.accuracy_optimizer.component_weights.copy()
                }
            )
            
            # Log integration event to database
            self.data_manager.db.log_system_event(
                event_type='INTEGRATED_SIGNAL',
                component='INTEGRATED_PROCESSOR',
                message=f'High-quality integrated signal generated for {integrated_signal.symbol}',
                details={
                    'signal_id': signal_id,
                    'prediction_id': integrated_signal.prediction_id,
                    'consensus_strength': integrated_signal.consensus_strength,
                    'confidence': integrated_signal.confidence,
                    'component_count': len(integrated_signal.components),
                    'components': list(integrated_signal.components.keys())
                },
                severity='INFO'
            )
            
            logger.info(f"💾 Integrated signal saved to database: {signal_id}")
            
        except Exception as e:
            logger.error(f"Failed to save integrated signal to database: {e}")
            # Fallback to file storage
            self._save_to_file(integrated_signal)
    
    def _save_to_file(self, integrated_signal: IntegratedSignal):
        """Fallback: Save to JSON file if database unavailable"""
        try:
            # Ensure data/ml directory exists
            os.makedirs('data/ml', exist_ok=True)
            predictions_file = 'data/ml/integrated_predictions.json'
            
            # Convert to JSON-serializable format
            signal_data = {
                'prediction_id': integrated_signal.prediction_id,
                'symbol': integrated_signal.symbol,
                'timestamp': integrated_signal.timestamp.isoformat(),
                'direction': integrated_signal.direction,
                'confidence': integrated_signal.confidence,
                'consensus_strength': integrated_signal.consensus_strength,
                'components': {
                    name: {
                        'name': comp.name if hasattr(comp, 'name') else name,
                        'confidence': comp.confidence,
                        'direction': comp.direction,
                        'raw_data': comp.raw_data if hasattr(comp, 'raw_data') else {}
                    }
                    for name, comp in integrated_signal.components.items()
                },
                'integration_metadata': integrated_signal.integration_metadata
            }
            
            # Load existing predictions
            predictions = []
            if os.path.exists(predictions_file):
                try:
                    with open(predictions_file, 'r') as f:
                        predictions = json.load(f)
                except:
                    predictions = []
            
            # Add new prediction
            predictions.append(signal_data)
            
            # Keep only last 1000 predictions
            predictions = predictions[-1000:]
            
            # Save back to file
            with open(predictions_file, 'w') as f:
                json.dump(predictions, f, indent=2)
                
            logger.info(f"💾 Integrated signal saved to file: {predictions_file}")
            
        except Exception as e:
            logger.error(f"Failed to save integrated signal to file: {e}")

    def update_accuracy_weights(self, prediction_id: str, actual_outcome: str, price_change: float):
        """Update component weights based on prediction accuracy (DATABASE INTEGRATED)"""
        try:
            # Load the original prediction (try database first, fallback to file)
            integrated_signal = self.load_prediction(prediction_id)
            if not integrated_signal:
                return
            
            # Determine accuracy for each component
            was_correct = (
                (actual_outcome.upper() == "UP" and price_change > 0) or
                (actual_outcome.upper() == "DOWN" and price_change < 0)
            )
            
            accuracy_score = abs(price_change) if was_correct else -abs(price_change)
            
            # Update each component's performance
            for component_name, component_signal in integrated_signal.components.items():
                component_was_correct = (
                    (component_signal.direction.upper() == "UP" and price_change > 0) or
                    (component_signal.direction.upper() == "DOWN" and price_change < 0)
                )
                
                # Calculate performance score (weighted by confidence)
                component_score = (accuracy_score if component_was_correct else -accuracy_score) * component_signal.confidence
                
                # Update optimizer weights
                self.accuracy_optimizer.update_component_performance(component_name, component_score)
            
            # Save updated weights (uses database if available)
            self.accuracy_optimizer.save_weights()
            
            # Log outcome to database
            if self.data_manager:
                try:
                    self.data_manager.db.log_system_event(
                        event_type='ACCURACY_UPDATE',
                        component='INTEGRATED_PROCESSOR',
                        message=f'Accuracy tracking updated for prediction {prediction_id[-8:]}',
                        details={
                            'prediction_id': prediction_id,
                            'actual_outcome': actual_outcome,
                            'price_change': price_change,
                            'was_correct': was_correct,
                            'accuracy_score': accuracy_score,
                            'updated_weights': self.accuracy_optimizer.component_weights.copy()
                        },
                        severity='INFO'
                    )
                except Exception as db_error:
                    logger.warning(f"Could not log accuracy update to database: {db_error}")
            
            # Fallback: Save outcome to file
            self._save_outcome_to_file(prediction_id, actual_outcome, price_change, was_correct, accuracy_score)
                
            print(f"📊 Accuracy tracking updated for {prediction_id[-8:]}: {actual_outcome} ({'✅' if was_correct else '❌'})")
            
        except Exception as e:
            print(f"❌ Error updating accuracy weights: {e}")
    
    def _save_outcome_to_file(self, prediction_id: str, actual_outcome: str, price_change: float, was_correct: bool, accuracy_score: float):
        """Fallback: Save outcome to JSON file"""
        try:
            outcome_log = {
                'prediction_id': prediction_id,
                'timestamp': datetime.now().isoformat(),
                'actual_outcome': actual_outcome,
                'price_change': price_change,
                'was_correct': was_correct,
                'accuracy_score': accuracy_score,
                'updated_weights': self.accuracy_optimizer.component_weights.copy()
            }
            
            # Save outcome to data/ml/accuracy_outcomes.json
            os.makedirs('data/ml', exist_ok=True)
            outcomes_file = 'data/ml/accuracy_outcomes.json'
            
            outcomes = []
            if os.path.exists(outcomes_file):
                try:
                    with open(outcomes_file, 'r') as f:
                        outcomes = json.load(f)
                except:
                    outcomes = []
            
            outcomes.append(outcome_log)
            
            # Keep only last 1000 outcomes
            outcomes = outcomes[-1000:]
            
            with open(outcomes_file, 'w') as f:
                json.dump(outcomes, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save outcome to file: {e}")
    
    def load_prediction(self, prediction_id: str) -> Optional[IntegratedSignal]:
        """Load a saved prediction by ID"""
        try:
            predictions_file = 'data/ml/integrated_predictions.json'
            if not os.path.exists(predictions_file):
                return None
                
            with open(predictions_file, 'r') as f:
                predictions = json.load(f)
            
            for pred_data in predictions:
                if pred_data.get('prediction_id') == prediction_id:
                    # Reconstruct IntegratedSignal
                    components = {}
                    for comp_name, comp_data in pred_data.get('components', {}).items():
                        components[comp_name] = ComponentSignal(
                            name=comp_data['name'],
                            confidence=comp_data['confidence'],
                            direction=comp_data['direction'],
                            raw_data=comp_data.get('raw_data', {})
                        )
                    
                    return IntegratedSignal(
                        symbol=pred_data['symbol'],
                        prediction_id=pred_data['prediction_id'],
                        timestamp=datetime.fromisoformat(pred_data['timestamp']),
                        components=components,
                        consensus_strength=pred_data['consensus_strength'],
                        confidence=pred_data['confidence'],
                        direction=pred_data['direction'],
                        integration_metadata=pred_data.get('integration_metadata', {})
                    )
            
            return None
            
        except Exception as e:
            print(f"❌ Error loading prediction {prediction_id}: {e}")
            return None
    
    def get_accuracy_stats(self) -> dict:
        """Get current accuracy statistics"""
        try:
            outcomes_file = 'data/ml/accuracy_outcomes.json'
            if not os.path.exists(outcomes_file):
                return {'total_predictions': 0, 'accuracy_rate': 0.0, 'component_performance': {}}
            
            with open(outcomes_file, 'r') as f:
                outcomes = json.load(f)
            
            if not outcomes:
                return {'total_predictions': 0, 'accuracy_rate': 0.0, 'component_performance': {}}
            
            total = len(outcomes)
            correct = sum(1 for outcome in outcomes if outcome.get('was_correct', False))
            accuracy_rate = correct / total if total > 0 else 0.0
            
            # Calculate component performance
            component_stats = {}
            for component_name in self.accuracy_optimizer.component_weights.keys():
                component_stats[component_name] = {
                    'weight': self.accuracy_optimizer.component_weights.get(component_name, 0.0),
                    'recent_performance': self.accuracy_optimizer.performance_history.get(component_name, [])[-10:]  # Last 10 scores
                }
            
            return {
                'total_predictions': total,
                'accuracy_rate': accuracy_rate,
                'component_performance': component_stats,
                'current_weights': self.accuracy_optimizer.component_weights.copy()
            }
            
        except Exception as e:
            print(f"❌ Error getting accuracy stats: {e}")
            return {'error': str(e)}
