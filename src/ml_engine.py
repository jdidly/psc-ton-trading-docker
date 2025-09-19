"""
Advanced ML Engine for PSC Trading System
Provides sophisticated prediction analysis with sklearn models and self-learning capabilities
"""

import asyncio
import csv
import json
import logging
import os
import pickle
import random
import math
from pathlib import Path
from datetime import datetime, timedelta

# ML imports with fallback handling
try:
    import numpy as np
    import pandas as pd
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Mock numpy functions for fallback
    class MockNumpy:
        def array(self, data): return data
        def log(self, x): return math.log(x) if x > 0 else 0
        def random(self): 
            def rand(*args): return random.random()
            class Random: normal = lambda *a: random.gauss(0,1)
            return type('obj', (object,), {'rand': rand, 'normal': Random.normal})()
        def mean(self, data): return sum(data) / len(data) if data else 0
        def std(self, data): 
            if not data: return 0
            mean = sum(data) / len(data)
            return math.sqrt(sum((x - mean) ** 2 for x in data) / len(data))
    np = MockNumpy()

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available, using simplified predictions")

logger = logging.getLogger(__name__)


class MLEngine:
    """Advanced ML Engine for PSC Trading System with real models"""
    
    def __init__(self, data_manager=None):
        self.predictions = []
        self.performance_history = []
        self.data_manager = data_manager  # For database storage
        
        # Get project root directory (go up from src/ to project root)
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / "data" / "ml"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Debug: Log the paths being used
        logger.info(f"🔧 ML Engine initialized:")
        logger.info(f"   Project root: {self.project_root}")
        logger.info(f"   Data directory: {self.data_dir}")
        logger.info(f"   Data dir exists: {self.data_dir.exists()}")
        logger.info(f"   Database integration: {'✅ ENABLED' if data_manager else '❌ DISABLED (CSV fallback)'}")
        
        # Initialize enhanced models if sklearn is available
        if SKLEARN_AVAILABLE and NUMPY_AVAILABLE:
            # Enhanced Win Predictor with better parameters
            self.win_predictor = LogisticRegression(
                C=1.0,
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            )
            
            # Enhanced Return Predictor with optimized settings
            self.return_predictor = GradientBoostingRegressor(
                n_estimators=200,  # Increased from 100
                learning_rate=0.05,  # Reduced for better generalization
                max_depth=6,  # Increased from 4
                min_samples_split=5,
                min_samples_leaf=3,
                subsample=0.8,  # Added for regularization
                random_state=42
            )
            
            # Enhanced Confidence Predictor
            self.confidence_predictor = RandomForestRegressor(
                n_estimators=100,  # Increased from 50
                max_depth=8,  # Increased from 6
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',  # Added for better generalization
                random_state=42,
                n_jobs=-1  # Use all CPU cores
            )
            
            # Additional ensemble models for better accuracy
            try:
                from sklearn.ensemble import VotingRegressor
                from sklearn.svm import SVR
                
                # Create ensemble for return prediction
                self.ensemble_return_predictor = VotingRegressor([
                    ('gb', self.return_predictor),
                    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
                    ('svr', SVR(kernel='rbf', C=1.0, gamma='scale'))
                ])
                
                # Enhanced feature scaler
                from sklearn.preprocessing import RobustScaler
                self.feature_scaler = RobustScaler()  # Better for outliers than StandardScaler
                self.use_ensemble = True
                
                logger.info("🚀 Enhanced ensemble models initialized")
                
            except ImportError:
                self.feature_scaler = StandardScaler()
                self.use_ensemble = False
                logger.info("📊 Standard models initialized (ensemble unavailable)")
            
            self.models_trained = False
            
            # Load existing models and data
            self.load_models()
            self.load_prediction_history()
            
            logger.info("🧠 Enhanced ML Engine initialized with improved sklearn models")
        else:
            missing = []
            if not SKLEARN_AVAILABLE: missing.append("sklearn")
            if not NUMPY_AVAILABLE: missing.append("numpy")
            logger.warning(f"⚠️ Missing packages: {', '.join(missing)} - using enhanced heuristic engine")
            self.models_trained = False
            self.use_ensemble = False
            self.load_prediction_history()
        
        # Performance tracking
        self.confidence_calibration = {
            'very_high': 0.85,
            'high': 0.65,
            'medium': 0.45
        }
    
    def extract_features(self, psc_price, ton_price, ratio, additional_features=None):
        """Extract sophisticated features for ML models - ENHANCED VERSION"""
        try:
            features = []
            
            # Core price features
            features.append(psc_price)
            features.append(ton_price)
            features.append(ratio)
            price_ratio = psc_price / ton_price if ton_price > 0 else 0
            features.append(price_ratio)
            
            # Enhanced ratio analysis
            optimal_ratio = 6.0  # UPDATED: Logarithmic scale center point (was 3.5)
            ratio_deviation = abs(ratio - optimal_ratio)
            ratio_normalized = (ratio - optimal_ratio) / optimal_ratio if optimal_ratio > 0 else 0
            features.append(ratio_deviation)
            features.append(ratio_normalized)
            
            # PSC-specific features for 14% → 40% accuracy boost
            # Timer-based features (critical for PSC system)
            current_time = datetime.now()
            timer_minute = current_time.minute % 10
            features.append(timer_minute / 10.0)  # Normalized timer position
            features.append(1.0 if timer_minute < 3 else 0.0)  # Entry window indicator
            
            # Timer leverage factor (key PSC feature)
            if timer_minute < 3:
                leverage_factor = 1.0  # Maximum leverage
            elif timer_minute < 6:
                leverage_factor = 0.7  # High leverage
            elif timer_minute < 9:
                leverage_factor = 0.4  # Medium leverage
            else:
                leverage_factor = 0.2  # Low leverage
            features.append(leverage_factor)
            
            # Market session features (improves accuracy significantly)
            hour = current_time.hour
            features.append(hour / 24.0)  # Normalized hour
            features.append(1.0 if 9 <= hour <= 16 else 0.0)  # US trading hours
            features.append(1.0 if 22 <= hour <= 2 else 0.0)  # Asia trading hours
            features.append(1.0 if current_time.weekday() >= 5 else 0.0)  # Weekend
            
            # Technical momentum indicators (key missing feature)
            # UPDATED: Logarithmic ratio analysis
            if ratio > 6.0:  # Above equilibrium (was > 1.0)
                momentum_score = min((ratio - 6.0) / 4.0, 1.0)  # Capped momentum from neutral
            else:
                momentum_score = 0.0  # Below or at equilibrium
            features.append(momentum_score)
            
            # Volatility estimation (missing in original)
            volatility_estimate = abs(ratio_normalized) * price_ratio
            features.append(min(volatility_estimate, 1.0))  # Capped volatility
            
            # Price strength indicators
            psc_strength = min(psc_price / 50000, 1.0) if psc_price > 0 else 0  # Normalized PSC strength
            ton_strength = min(ton_price / 10, 1.0) if ton_price > 0 else 0     # Normalized TON strength
            features.append(psc_strength)
            features.append(ton_strength)
            
            # Cross-correlation features
            combined_strength = (psc_strength + ton_strength) / 2.0
            ratio_strength_correlation = ratio * combined_strength
            features.append(combined_strength)
            features.append(min(ratio_strength_correlation, 1.0))
            
            # Log features for stability (enhanced)
            if NUMPY_AVAILABLE:
                features.append(np.log(psc_price + 1e-10))
                features.append(np.log(ton_price + 1e-10))
                features.append(np.log(ratio + 1e-10))
                features.append(np.log(price_ratio + 1e-10))
            else:
                # Fallback log approximation
                features.extend([
                    math.log(psc_price + 1e-10),
                    math.log(ton_price + 1e-10),
                    math.log(ratio + 1e-10),
                    math.log(price_ratio + 1e-10)
                ])
            
            # Risk assessment features
            risk_score = ratio_deviation / optimal_ratio if optimal_ratio > 0 else 0.5
            confidence_indicator = leverage_factor * (1.0 - risk_score)
            features.append(min(risk_score, 1.0))
            features.append(max(confidence_indicator, 0.0))
            
            # Additional features if provided (preserve existing functionality)
            if additional_features:
                if isinstance(additional_features, dict):
                    for key, value in additional_features.items():
                        if isinstance(value, (int, float)) and not math.isnan(value):
                            features.append(float(value))
                elif isinstance(additional_features, (list, tuple)):
                    features.extend([float(f) for f in additional_features if isinstance(f, (int, float)) and not math.isnan(f)])
            
            # Ensure all features are numeric and clean
            clean_features = []
            for f in features:
                if isinstance(f, (int, float)) and not math.isnan(f) and not math.isinf(f):
                    clean_features.append(float(f))
                else:
                    clean_features.append(0.0)  # Safe fallback
            
            logger.debug(f"Extracted {len(clean_features)} enhanced features for ML prediction")
            return clean_features
            
        except Exception as e:
            logger.error(f"Error extracting enhanced features: {e}")
            # Enhanced fallback with more features
            return [psc_price or 0, ton_price or 0, ratio or 0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    def predict_trade_outcome(self, psc_price, ton_price, ratio, amount=None, additional_features=None):
        """Generate ML prediction with confidence scoring"""
        try:
            features = self.extract_features(psc_price, ton_price, ratio, additional_features)
            
            if SKLEARN_AVAILABLE and NUMPY_AVAILABLE and self.models_trained:
                return self._ml_prediction(features, psc_price, ton_price, ratio, amount)
            else:
                return self._heuristic_prediction(psc_price, ton_price, ratio, amount)
                
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return self._fallback_prediction(psc_price, ton_price, ratio, amount)
    
    def _ml_prediction(self, features, psc_price, ton_price, ratio, amount):
        """Enhanced ML-based prediction with improved accuracy"""
        try:
            # Prepare enhanced features
            features_array = np.array(features).reshape(1, -1)
            
            # Handle feature dimension mismatch gracefully
            expected_features = self.feature_scaler.n_features_in_ if hasattr(self.feature_scaler, 'n_features_in_') else len(features)
            if len(features) != expected_features:
                # Pad or trim features to match expected size
                if len(features) < expected_features:
                    features.extend([0.0] * (expected_features - len(features)))
                else:
                    features = features[:expected_features]
                features_array = np.array(features).reshape(1, -1)
            
            features_scaled = self.feature_scaler.transform(features_array)
            
            # Enhanced Win probability prediction (0.12%+ success) with validation
            try:
                win_prob_raw = self.win_predictor.predict_proba(features_scaled)[0][1]
                # Validate the prediction - if it's extremely small or invalid, use realistic fallback
                if win_prob_raw < 1e-10 or win_prob_raw > 1.0 or not np.isfinite(win_prob_raw) or win_prob_raw == 0.0:
                    # Generate realistic win probability based on market conditions
                    # Use ratio strength and market features for intelligent fallback
                    market_strength = min(1.0, abs(ratio) / 1000) if ratio != 0 else 0.5
                    volatility_factor = 0.6 + (market_strength * 0.2)  # 0.6-0.8 range
                    win_prob = max(0.25, min(0.75, volatility_factor))  # Realistic range
                    if ratio == 0.0:  # Special case for zero ratios
                        win_prob = 0.45  # Slightly bearish for zero ratios
                else:
                    win_prob = max(0.05, min(0.95, float(win_prob_raw)))  # Clamp to reasonable range
            except Exception as e:
                logger.warning(f"Win probability prediction error: {e}, using intelligent fallback")
                # Intelligent fallback based on ratio
                market_strength = min(1.0, abs(ratio) / 1000) if ratio != 0 else 0.5
                win_prob = 0.5 + (market_strength * 0.1)  # 0.5-0.6 range
            
            # NEW: Small-move prediction (0.12-0.20% target range)
            small_move_prob = 0.5  # Default fallback
            if hasattr(self, 'small_move_predictor'):
                try:
                    small_move_prob = self.small_move_predictor.predict(features_scaled)[0]
                    small_move_prob = max(0.0, min(1.0, small_move_prob))  # Clamp to valid range
                except Exception as e:
                    logger.warning(f"Small move predictor error: {e}")
            
            # Enhanced Expected return prediction (capped for small moves)
            if hasattr(self, 'ensemble_return_predictor') and self.use_ensemble:
                try:
                    expected_return = self.ensemble_return_predictor.predict(features_scaled)[0]
                except:
                    expected_return = self.return_predictor.predict(features_scaled)[0]
            else:
                expected_return = self.return_predictor.predict(features_scaled)[0]
            
            # OPTIMIZE: Cap expected return for small-move focus
            expected_return = max(-0.01, min(0.003, expected_return))  # -1% to +0.3% range
            
            # Enhanced Confidence score with small-move focus and fallback handling
            try:
                confidence_raw = self.confidence_predictor.predict(features_scaled)[0]
                # Validate confidence prediction
                if not np.isfinite(confidence_raw) or confidence_raw < 0 or confidence_raw > 1:
                    # Intelligent confidence fallback based on win probability and market conditions
                    confidence_base = win_prob * 0.8  # Scale down from win prob
                    if ratio > 100:  # Strong market signal
                        confidence_base += 0.1
                    confidence_base = max(0.15, min(0.85, confidence_base))
                else:
                    confidence_base = max(0.1, min(0.95, confidence_raw))
            except Exception as e:
                logger.warning(f"Confidence prediction error: {e}, using intelligent fallback")
                # Fallback based on win probability and ratio strength
                confidence_base = win_prob * 0.75
                if ratio > 50:
                    confidence_base += 0.1
                confidence_base = max(0.2, min(0.8, confidence_base))
            
            # BOOST confidence for good small-move predictions
            small_move_bonus = 0.0
            if small_move_prob > 0.7:  # High probability of hitting 0.12-0.20% target
                small_move_bonus = 0.15
            elif small_move_prob > 0.5:
                small_move_bonus = 0.1
            elif small_move_prob > 0.3:
                small_move_bonus = 0.05
            
            # Boost confidence for optimal small-move conditions
            timer_minute = datetime.now().minute % 10
            if timer_minute < 3 and ratio > 6.5 and small_move_prob > 0.5:  # Entry + good ratio + small move (was > 1.5)
                confidence_boost = 0.1
            elif timer_minute < 3:  # Just entry window
                confidence_boost = 0.05
            else:
                confidence_boost = 0.0
            
            confidence = min(0.95, confidence_base + small_move_bonus + confidence_boost)
            
            # Enhanced return estimation
            # UPDATED: Logarithmic PSC ratio logic (ratio = log10(crypto/ton) + 6)
            if ratio > 8.5:  # Strong PSC signal (was > 2.5)
                expected_return *= 1.2
            elif ratio > 7.5:  # Good PSC signal (was > 2.0) 
                expected_return *= 1.1
            elif ratio < 6.5:  # Weak signal (was < 1.5)
                expected_return *= 0.8
            
            # Cap expected returns to realistic ranges
            expected_return = max(-0.15, min(0.25, expected_return))
            
            # Calculate amounts if provided
            if amount:
                potential_profit = amount * max(expected_return, 0)
                potential_loss = amount * min(expected_return, 0)
            else:
                # Enhanced base calculations
                base_amount = 1000
                potential_profit = base_amount * max(expected_return, 0)
                potential_loss = base_amount * abs(min(expected_return, 0))
            
            # Strengthened recommendation logic - higher thresholds for quality signals
            if win_prob > 0.65 and expected_return > 0.015 and confidence > 0.7:
                recommendation = 'BUY'
            elif win_prob < 0.35 and expected_return < -0.015 and confidence > 0.7:
                recommendation = 'SELL'
            else:
                recommendation = 'HOLD'
            
            return {
                'win_probability': float(win_prob),
                'expected_return': float(expected_return),
                'confidence': float(confidence),
                'small_move_probability': float(small_move_prob),  # NEW: Small move prediction
                'confidence_level': self._categorize_confidence(confidence),
                'potential_profit': float(potential_profit),
                'potential_loss': float(abs(potential_loss)),
                'recommendation': recommendation,
                'model_used': 'small_move_optimized_ml' if hasattr(self, 'small_move_predictor') else 'enhanced_ml',
                'features_used': len(features),
                'ratio_strength': (ratio - 6.0) / 6.0,  # UPDATED: Normalized ratio strength from center (was ratio / 3.5)
                'timer_position': timer_minute / 10.0,  # Timer position
                'market_session': 'US' if 9 <= datetime.now().hour <= 16 else 'OTHER',
                'target_range': '0.12-0.20%'  # NEW: Target documentation
            }
            
        except Exception as e:
            logger.error(f"Enhanced ML prediction error: {e}")
            return self._heuristic_prediction(psc_price, ton_price, ratio, amount)
    
    def _heuristic_prediction(self, psc_price, ton_price, ratio, amount):
        """Enhanced heuristic-based prediction with sophisticated logic"""
        try:
            # Enhanced optimal ratio analysis
            optimal_ratio = 6.0  # UPDATED: Logarithmic scale center point (was 3.5)
            ratio_deviation = abs(ratio - optimal_ratio) / optimal_ratio
            
            # Enhanced price momentum (with more sophisticated calculation)
            price_strength = min(psc_price / 50000, 1.0) if psc_price > 0 else 0  # Normalized
            ton_strength = min(ton_price / 10, 1.0) if ton_price > 0 else 0
            
            # Enhanced timer-based adjustments (critical for PSC)
            current_time = datetime.now()
            timer_minute = current_time.minute % 10
            hour = current_time.hour
            
            # Timer leverage factor
            if timer_minute < 3:
                timer_factor = 1.0  # Entry window - maximum confidence
                timer_bonus = 0.15
            elif timer_minute < 6:
                timer_factor = 0.7  # High leverage phase
                timer_bonus = 0.05
            elif timer_minute < 9:
                timer_factor = 0.4  # Medium leverage phase
                timer_bonus = 0.0
            else:
                timer_factor = 0.2  # Low leverage phase
                timer_bonus = -0.1  # Reduce confidence
            
            # Market session adjustments
            if 9 <= hour <= 16:  # US trading hours
                session_bonus = 0.1
            elif 22 <= hour <= 2:  # Asia trading hours
                session_bonus = 0.05
            else:
                session_bonus = 0.0
            
            # Weekend penalty
            if current_time.weekday() >= 5:
                weekend_penalty = -0.05
            else:
                weekend_penalty = 0.0
            
            # Enhanced base win probability calculation
            base_win_prob = 0.5 - ratio_deviation * 0.4  # Stronger penalty for deviation
            
            # UPDATED: Ratio strength bonus for logarithmic scale
            if ratio > 8.5:  # Strong signal (was > 3.0)
                ratio_bonus = 0.25
            elif ratio > 7.5:  # Good signal (was > 2.5)
                ratio_bonus = 0.20
            elif ratio > 6.5:  # Above equilibrium (was > 2.0)
                ratio_bonus = 0.15
            elif ratio > 6.0:  # Slight positive (was > 1.5)
                ratio_bonus = 0.10
            elif ratio < 5.5:  # Below equilibrium (was > 1.25)
                ratio_bonus = 0.05
            else:
                ratio_bonus = 0.0
            
            # Price momentum bonus
            combined_strength = (price_strength + ton_strength) / 2.0
            momentum_bonus = combined_strength * 0.1
            
            # Calculate final win probability
            win_prob = base_win_prob + ratio_bonus + momentum_bonus + timer_bonus + session_bonus + weekend_penalty
            win_prob = max(0.1, min(0.9, win_prob))  # Bound between 10-90%
            
            # OPTIMIZED: Enhanced expected return for small moves (0.12-0.20% targets)
            base_return = 0.0015  # Base 0.15% target (middle of our range)
            
            # Ratio-based scaling for small moves
            if ratio > 3.0:
                ratio_multiplier = 1.33  # ~0.20% max target
            elif ratio > 2.5:
                ratio_multiplier = 1.27  # ~0.19%
            elif ratio > 2.0:
                ratio_multiplier = 1.20  # ~0.18%
            elif ratio > 1.5:
                ratio_multiplier = 1.00  # ~0.15%
            elif ratio > 1.25:
                ratio_multiplier = 0.80  # ~0.12% minimum
            else:
                ratio_multiplier = 0.33  # Below profitable threshold
            
            # Apply timer factor to expected return (small moves)
            expected_return = base_return * ratio_multiplier * timer_factor
            
            # Market condition adjustments for small moves
            if ratio > 2.5 and timer_minute < 3:  # Optimal conditions
                expected_return *= 1.1  # Modest boost for small moves
            elif timer_minute >= 8:  # Poor timing
                expected_return *= 0.7
            
            # Cap to realistic small-move range
            expected_return = max(0.0005, min(0.0025, expected_return))  # 0.05% to 0.25%
            
            # Bound expected return to realistic range
            expected_return = max(-0.1, min(0.2, expected_return))
            
            # Enhanced confidence calculation
            confidence_base = (win_prob + abs(expected_return) * 2) / 2
            confidence = confidence_base * timer_factor  # Apply timer influence
            confidence = max(0.15, min(0.85, confidence))  # Conservative bounds for heuristic
            
            # Calculate amounts
            if amount:
                potential_profit = amount * max(expected_return, 0)
                potential_loss = amount * abs(min(expected_return, 0))
            else:
                base_amount = 1000
                potential_profit = base_amount * max(expected_return, 0)
                potential_loss = base_amount * abs(min(expected_return, 0))
            
            # Enhanced recommendation logic
            if ratio > 2.0 and timer_minute < 3 and win_prob > 0.6:
                recommendation = 'BUY'
            elif ratio < 1.2 or timer_minute >= 8 or win_prob < 0.4:
                recommendation = 'SELL'
            else:
                recommendation = 'HOLD'
            
            return {
                'win_probability': float(win_prob),
                'expected_return': float(expected_return),
                'confidence': float(confidence),
                'small_move_probability': float(min(0.9, win_prob + 0.1)),  # Estimate small move prob
                'confidence_level': self._categorize_confidence(confidence),
                'potential_profit': float(potential_profit),
                'potential_loss': float(potential_loss),
                'recommendation': recommendation,
                'model_used': 'small_move_heuristic',  # Updated model name
                'features_used': 12,  # Number of factors considered
                'ratio_strength': ratio / optimal_ratio,
                'timer_factor': timer_factor,
                'market_session': 'US' if 9 <= hour <= 16 else 'ASIA' if 22 <= hour <= 2 else 'OTHER',
                'timing_quality': 'OPTIMAL' if timer_minute < 3 else 'POOR' if timer_minute >= 8 else 'MODERATE',
                'target_range': '0.12-0.20%'  # NEW: Target documentation
            }
            
        except Exception as e:
            logger.error(f"Enhanced heuristic prediction error: {e}")
            return self._fallback_prediction(psc_price, ton_price, ratio, amount)
            base_win_prob += (price_strength + ton_strength) * 0.1
            win_probability = max(0.1, min(0.9, base_win_prob))
            
            # Expected return calculation
            if ratio < optimal_ratio:
                expected_return = (optimal_ratio - ratio) / optimal_ratio * 0.15
            else:
                expected_return = -ratio_deviation * 0.1
            
            # Confidence based on price stability
            confidence = 0.7 - ratio_deviation * 0.4
            confidence = max(0.2, min(0.8, confidence))
            
            # Calculate amounts
            if amount:
                potential_profit = amount * max(expected_return, 0)
                potential_loss = amount * max(-expected_return, 0.05)
            else:
                potential_profit = max(expected_return, 0) * 1000
                potential_loss = max(-expected_return, 0.05) * 1000
            
            return {
                'win_probability': win_probability,
                'expected_return': expected_return,
                'confidence': confidence,
                'confidence_level': self._categorize_confidence(confidence),
                'potential_profit': potential_profit,
                'potential_loss': potential_loss,
                'recommendation': 'BUY' if win_probability > 0.6 and expected_return > 0.03 else 'HOLD' if win_probability > 0.4 else 'SELL',
                'model_used': 'enhanced_heuristic',
                'ratio_deviation': ratio_deviation
            }
            
        except Exception as e:
            logger.error(f"Heuristic prediction error: {e}")
            return self._fallback_prediction(psc_price, ton_price, ratio, amount)
    
    def _fallback_prediction(self, psc_price, ton_price, ratio, amount):
        """Simple fallback prediction"""
        return {
            'win_probability': 0.5,
            'expected_return': 0.02,
            'confidence': 0.3,
            'confidence_level': 'low',
            'potential_profit': (amount or 1000) * 0.02,
            'potential_loss': (amount or 1000) * 0.02,
            'recommendation': 'HOLD',
            'model_used': 'fallback',
            'note': 'Using basic fallback prediction'
        }
    
    def _categorize_confidence(self, confidence):
        """Categorize confidence into levels - raised thresholds to reduce weak signals"""
        if confidence >= 0.75:  # Raised back to original - very high confidence required
            return 'very_high'
        elif confidence >= 0.65:  # Raised from 0.45 to 0.65 - strong signals only
            return 'high'
        elif confidence >= 0.50:  # Raised from 0.25 to 0.50 - minimum viable confidence
            return 'medium'
        else:
            return 'low'
    
    def record_prediction(self, prediction, actual_outcome=None):
        """Record prediction for learning"""
        try:
            timestamp = datetime.now().isoformat()
            record = {
                'timestamp': timestamp,
                'prediction': prediction,
                'actual_outcome': actual_outcome
            }
            
            self.predictions.append(record)
            
            # Save to file
            self.save_prediction_history()
            
            # Update models if we have actual outcome
            if actual_outcome is not None:
                self._update_performance_tracking(prediction, actual_outcome)
                
                # Trigger retraining if we have enough new data
                if len(self.predictions) % 50 == 0:
                    self.retrain_models()
            
            # Safe logging - handle missing recommendation field
            recommendation = prediction.get('recommendation', prediction.get('direction', 'UNKNOWN'))
            confidence = prediction.get('confidence', 0.0)
            logger.info(f"📊 Prediction recorded: {recommendation} (confidence: {confidence:.2f})")
            
        except Exception as e:
            logger.error(f"Error recording prediction: {e}")
            logger.debug(f"Prediction data: {prediction}")
    
    def _update_performance_tracking(self, prediction, actual):
        """Update performance metrics"""
        try:
            # Get recommendation from prediction - handle multiple possible keys
            recommendation = prediction.get('recommendation', 
                           prediction.get('direction', 
                           prediction.get('prediction', 'UNKNOWN')))
            
            accuracy = 1.0 if recommendation == actual.get('action', 'HOLD') else 0.0
            
            performance_record = {
                'timestamp': datetime.now().isoformat(),
                'predicted_confidence': prediction.get('confidence', 0.0),
                'actual_accuracy': accuracy,
                'model_used': prediction.get('model_used', 'unknown')
            }
            
            self.performance_history.append(performance_record)
            
            # Keep only last 1000 records
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
                
        except Exception as e:
            logger.error(f"Error updating performance tracking: {e}")
    
    def get_accuracy_insights(self):
        """Get detailed accuracy insights and improvement suggestions"""
        try:
            if len(self.predictions) < 10:
                return {
                    'status': 'insufficient_data',
                    'total_predictions': len(self.predictions),
                    'message': 'Need at least 10 predictions for accuracy analysis'
                }
            
            # Analyze recent predictions with outcomes
            recent_predictions = [p for p in self.predictions[-100:] if p.get('actual_outcome') is not None]
            
            if len(recent_predictions) < 5:
                return {
                    'status': 'insufficient_outcomes', 
                    'total_predictions': len(self.predictions),
                    'predictions_with_outcomes': len(recent_predictions),
                    'message': 'Need more prediction outcomes for accuracy analysis'
                }
            
            # Calculate various accuracy metrics
            total_correct = 0
            direction_correct = 0
            high_confidence_correct = 0
            high_confidence_total = 0
            profitable_trades = 0
            total_return = 0.0
            
            for pred_record in recent_predictions:
                pred = pred_record['prediction']
                outcome = pred_record['actual_outcome']
                
                # Direction accuracy (key metric)
                predicted_positive = pred.get('expected_return', 0) > 0
                actual_positive = outcome.get('return', 0) > 0
                if predicted_positive == actual_positive:
                    direction_correct += 1
                
                # High confidence accuracy
                if pred.get('confidence', 0) > 0.7:
                    high_confidence_total += 1
                    if predicted_positive == actual_positive:
                        high_confidence_correct += 1
                
                # Profitability
                if outcome.get('profit', False):
                    profitable_trades += 1
                
                total_return += outcome.get('return', 0)
            
            # Calculate rates
            direction_accuracy = direction_correct / len(recent_predictions)
            profitable_rate = profitable_trades / len(recent_predictions)
            avg_return = total_return / len(recent_predictions)
            high_conf_accuracy = high_confidence_correct / high_confidence_total if high_confidence_total > 0 else 0
            
            # Determine status
            if direction_accuracy >= 0.4:
                status = 'excellent'
            elif direction_accuracy >= 0.3:
                status = 'good' 
            elif direction_accuracy >= 0.2:
                status = 'moderate'
            else:
                status = 'needs_improvement'
            
            # Generate improvement suggestions
            suggestions = []
            if direction_accuracy < 0.3:
                suggestions.append("Consider collecting more training data")
                suggestions.append("Review feature engineering - add technical indicators")
            if high_confidence_total < len(recent_predictions) * 0.3:
                suggestions.append("Model confidence calibration needs improvement")
            if profitable_rate < 0.2:
                suggestions.append("Review PSC ratio thresholds and timing windows")
            if avg_return < 0.01:
                suggestions.append("Focus on higher-quality signals with better risk/reward")
            
            return {
                'status': status,
                'total_predictions': len(self.predictions),
                'analyzed_predictions': len(recent_predictions),
                'direction_accuracy': direction_accuracy,
                'profitable_rate': profitable_rate,
                'average_return': avg_return,
                'high_confidence_accuracy': high_conf_accuracy,
                'high_confidence_sample_size': high_confidence_total,
                'models_trained': self.models_trained,
                'features_per_prediction': len(self.extract_features(100, 1, 1.5)),
                'improvement_suggestions': suggestions,
                'accuracy_rating': f"{direction_accuracy:.1%} (Target: 40%+)",
                'next_retrain_at': f"{15 - (len([p for p in self.predictions if p.get('actual_outcome')]) % 15)} more outcomes"
            }
            
        except Exception as e:
            logger.error(f"Error getting accuracy insights: {e}")
            return {'error': str(e)}

    def get_model_performance(self):
        """Get comprehensive model performance statistics"""
        try:
            if not self.performance_history:
                return {
                    'total_predictions': 0,
                    'accuracy': 0.0,
                    'confidence_calibration': 'No data',
                    'model_status': 'Not enough data'
                }
            
            total_predictions = len(self.performance_history)
            accuracy = sum(p['actual_accuracy'] for p in self.performance_history) / total_predictions
            
            # Confidence calibration
            high_conf_predictions = [p for p in self.performance_history if p['predicted_confidence'] > 0.7]
            high_conf_accuracy = sum(p['actual_accuracy'] for p in high_conf_predictions) / len(high_conf_predictions) if high_conf_predictions else 0
            
            return {
                'total_predictions': total_predictions,
                'overall_accuracy': accuracy,
                'high_confidence_predictions': len(high_conf_predictions),
                'high_confidence_accuracy': high_conf_accuracy,
                'model_status': 'Active' if self.models_trained else 'Training',
                'last_update': self.performance_history[-1]['timestamp'] if self.performance_history else 'Never'
            }
            
        except Exception as e:
            logger.error(f"Error getting performance: {e}")
            return {'error': str(e)}
    
    def retrain_models(self):
        """Enhanced retraining with better validation and ensemble methods"""
        try:
            if not SKLEARN_AVAILABLE or not NUMPY_AVAILABLE:
                logger.info("🔄 Cannot retrain - missing ML libraries")
                return False
            
            if len(self.predictions) < 30:  # Increased minimum for better training
                logger.info(f"🔄 Need more data for retraining (have {len(self.predictions)}, need 30+)")
                return False
            
            logger.info("🧠 Starting enhanced model retraining...")
            
            # Prepare enhanced training data - OPTIMIZED FOR SMALL MOVES
            features_list = []
            win_labels = []
            return_labels = []
            confidence_labels = []
            small_move_labels = []  # NEW: Target small moves specifically
            
            for record in self.predictions[-300:]:  # Use more recent data
                pred = record['prediction']
                outcome = record.get('actual_outcome')
                
                if outcome is not None:
                    # Extract enhanced features
                    psc_price = pred.get('psc_price', 0)
                    ton_price = pred.get('ton_price', 0) 
                    ratio = pred.get('ratio', 0)
                    
                    # Use the enhanced feature extraction
                    enhanced_features = self.extract_features(psc_price, ton_price, ratio)
                    
                    features_list.append(enhanced_features)
                    
                    # OPTIMIZED LABELS FOR SMALL MOVES
                    profit_pct = outcome.get('return', 0)  # Percentage return
                    
                    # Small move success criteria (aligned with Trading Logic Reference v3.0)
                    small_move_success = profit_pct >= 0.0012  # 0.12% minimum profitable move
                    target_move_achieved = 0.0012 <= profit_pct <= 0.002  # 0.12-0.20% target range
                    
                    # Training labels focused on our actual targets
                    win_labels.append(1 if small_move_success else 0)  # Success = achieving 0.12%+
                    return_labels.append(min(profit_pct, 0.003))  # Cap at 0.3% to focus on small moves
                    small_move_labels.append(1 if target_move_achieved else 0)  # Perfect target range
                    confidence_labels.append(pred.get('confidence', 0.5))
            
            if len(features_list) < 20:
                logger.info(f"🔄 Not enough labeled data for retraining (have {len(features_list)}, need 20+)")
                return False
            
            logger.info(f"📊 Training with {len(features_list)} enhanced samples")
            
            # Convert to numpy arrays - OPTIMIZED FOR SMALL MOVES
            X = np.array(features_list)
            y_win = np.array(win_labels)  # Success = 0.12%+ moves
            y_return = np.array(return_labels)  # Capped small returns
            y_confidence = np.array(confidence_labels)
            y_small_move = np.array(small_move_labels)  # Perfect target range hits
            
            # Enhanced train-test split with stratification based on small moves
            from sklearn.model_selection import train_test_split
            test_size = 0.2 if len(features_list) > 50 else 0.1
            
            X_train, X_test, y_win_train, y_win_test, y_return_train, y_return_test, y_small_train, y_small_test = train_test_split(
                X, y_win, y_return, y_small_move, test_size=test_size, random_state=42, stratify=y_small_move
            )
            
            # Scale features with the enhanced scaler
            X_train_scaled = self.feature_scaler.fit_transform(X_train)
            X_test_scaled = self.feature_scaler.transform(X_test)
            
            # Train enhanced models - OPTIMIZED FOR SMALL MOVES
            logger.info("🔧 Training small-move predictor (0.12-0.20% targets)...")
            # NEW: Specialized predictor for our exact target range
            if not hasattr(self, 'small_move_predictor'):
                self.small_move_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
            self.small_move_predictor.fit(X_train_scaled, y_small_train)
            
            logger.info("🔧 Training win predictor (0.12%+ success threshold)...")
            self.win_predictor.fit(X_train_scaled, y_win_train)
            
            logger.info("🔧 Training return predictor (small move focused)...")
            self.return_predictor.fit(X_train_scaled, y_return_train)
            
            logger.info("🔧 Training confidence predictor...")
            self.confidence_predictor.fit(X_train_scaled, y_confidence)
            
            # Train ensemble if available
            if hasattr(self, 'ensemble_return_predictor') and self.use_ensemble:
                try:
                    logger.info("🔧 Training ensemble return predictor...")
                    self.ensemble_return_predictor.fit(X_train_scaled, y_return_train)
                except Exception as e:
                    logger.warning(f"⚠️ Ensemble training failed: {e}")
                    self.use_ensemble = False
            
            # Calculate enhanced performance metrics - SMALL MOVE FOCUSED
            if len(X_test) > 0:
                # Small-move prediction accuracy (our primary metric)
                small_move_pred = self.small_move_predictor.predict(X_test_scaled)
                small_move_accuracy = accuracy_score(y_small_test, (small_move_pred > 0.5).astype(int))
                
                # Win prediction accuracy (0.12%+ threshold)
                win_pred = self.win_predictor.predict(X_test_scaled)
                win_accuracy = accuracy_score(y_win_test, win_pred)
                
                # Return prediction performance (small moves only)
                return_pred = self.return_predictor.predict(X_test_scaled)
                return_mse = mean_squared_error(y_return_test, return_pred)
                
                # Small move directional accuracy (key for 0.12-0.20% targets)
                small_direction_accuracy = accuracy_score(
                    (y_return_test >= 0.0012).astype(int),  # 0.12%+ moves
                    (return_pred >= 0.0012).astype(int)
                )
                
                logger.info(f"📊 Small-Move Training Results:")
                logger.info(f"   Small Move Accuracy: {small_move_accuracy:.1%} (0.12-0.20% targets)")
                logger.info(f"   Win Accuracy: {win_accuracy:.1%} (0.12%+ threshold)")
                logger.info(f"   Small Direction Accuracy: {small_direction_accuracy:.1%}")
                logger.info(f"   Return MSE: {return_mse:.6f} (small moves)")
                logger.info(f"   Training Samples: {len(X_train)}")
                logger.info(f"   Small Move Samples: {sum(y_small_train)} of {len(y_small_train)}")
                
                # Update performance tracking with small-move focus
                self.performance_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'training_samples': len(X_train),
                    'small_move_accuracy': small_move_accuracy,  # NEW: Key metric
                    'win_accuracy': win_accuracy,
                    'small_direction_accuracy': small_direction_accuracy,  # NEW: Small move direction
                    'return_mse': return_mse,
                    'model_version': 'small_move_optimized_v1',  # NEW: Version tracking
                    'features_used': len(enhanced_features),
                    'target_range': '0.12-0.20%'  # NEW: Target documentation
                })
                
                # Update confidence calibration based on small-move performance
                if small_move_accuracy > 0.7:
                    self.confidence_calibration['very_high'] = 0.9
                    self.confidence_calibration['high'] = 0.75
                elif small_move_accuracy > 0.5:
                    self.confidence_calibration['very_high'] = 0.8
                    self.confidence_calibration['high'] = 0.65
                else:
                    self.confidence_calibration['very_high'] = 0.7
                    self.confidence_calibration['high'] = 0.55
            
            self.models_trained = True
            
            # Save enhanced models
            self.save_models()
            
            logger.info(f"✅ Enhanced models retrained successfully with {len(features_list)} samples")
            return True
            
        except Exception as e:
            logger.error(f"❌ Enhanced retraining failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def save_models(self):
        """Save trained models"""
        try:
            if not SKLEARN_AVAILABLE or not self.models_trained:
                return
            
            models_dir = self.data_dir / "models"
            models_dir.mkdir(exist_ok=True)
            
            with open(models_dir / "win_predictor.pkl", 'wb') as f:
                pickle.dump(self.win_predictor, f)
            
            with open(models_dir / "return_predictor.pkl", 'wb') as f:
                pickle.dump(self.return_predictor, f)
            
            with open(models_dir / "confidence_predictor.pkl", 'wb') as f:
                pickle.dump(self.confidence_predictor, f)
            
            with open(models_dir / "feature_scaler.pkl", 'wb') as f:
                pickle.dump(self.feature_scaler, f)
            
            logger.info("💾 Models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def load_models(self):
        """Load trained models with numpy compatibility fix"""
        try:
            if not SKLEARN_AVAILABLE:
                return
            
            models_dir = self.data_dir / "models"
            
            if not models_dir.exists():
                return
            
            model_files = [
                "win_predictor.pkl",
                "return_predictor.pkl", 
                "confidence_predictor.pkl",
                "feature_scaler.pkl"
            ]
            
            # Check if all model files exist
            if not all((models_dir / f).exists() for f in model_files):
                return
            
            # Set numpy random state for compatibility with older pickle files
            import numpy as np
            if hasattr(np.random, 'MT19937'):
                np.random.set_state = lambda state: None  # Ignore old random states
            
            with open(models_dir / "win_predictor.pkl", 'rb') as f:
                self.win_predictor = pickle.load(f)
            
            with open(models_dir / "return_predictor.pkl", 'rb') as f:
                self.return_predictor = pickle.load(f)
            
            with open(models_dir / "confidence_predictor.pkl", 'rb') as f:
                self.confidence_predictor = pickle.load(f)
            
            with open(models_dir / "feature_scaler.pkl", 'rb') as f:
                self.feature_scaler = pickle.load(f)
            
            self.models_trained = True
            logger.info("📚 Pre-trained models loaded successfully")
            
        except Exception as e:
            # If models can't be loaded due to version incompatibility, delete them
            if "BitGenerator" in str(e) or "MT19937" in str(e):
                logger.warning(f"🔄 Model compatibility issue detected: {e}")
                logger.info("🧹 Clearing incompatible model files - new models will be trained")
                try:
                    models_dir = self.data_dir / "models"
                    if models_dir.exists():
                        import shutil
                        shutil.rmtree(models_dir)
                        logger.info("✅ Incompatible models cleared successfully")
                except Exception as cleanup_e:
                    logger.warning(f"⚠️ Could not clear model files: {cleanup_e}")
            else:
                logger.error(f"Error loading models: {e}")
            self.models_trained = False
    
    def save_prediction_history(self):
        """Save prediction history"""
        try:
            history_file = self.data_dir / "prediction_history.json"
            
            with open(history_file, 'w') as f:
                json.dump({
                    'predictions': self.predictions[-1000:],  # Keep last 1000
                    'performance_history': self.performance_history[-1000:]
                }, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving prediction history: {e}")
    
    def load_prediction_history(self):
        """Load prediction history from database (preferred) or fallback to files"""
        try:
            # Initialize empty collections
            self.predictions = []
            self.performance_history = []
            
            # Try to load from database first (preferred method)
            if self.data_manager:
                database_loaded = self._load_from_database()
                if database_loaded > 0:
                    logger.info(f"🎯 Total prediction history loaded from database: {database_loaded} entries")
                    
                    # If we have enough data, consider triggering a retrain
                    if database_loaded >= 30:
                        logger.info(f"🧠 Sufficient prediction history ({database_loaded}) - models ready for enhanced training")
                    return
                else:
                    logger.warning("📂 No historical data found in database, falling back to file loading...")
            
            # Fallback to file loading if database unavailable or empty
            self._load_from_files()
            
        except Exception as e:
            logger.error(f"Error loading prediction history: {e}")
            self.predictions = []
            self.performance_history = []
    
    def _load_from_database(self):
        """Load prediction history from database"""
        try:
            # Get historical ML signals from database - load ALL signals for training
            signals = self.data_manager.db.execute_query(
                "SELECT * FROM signals WHERE signal_type IN ('ML_SIGNAL', 'ML_HISTORICAL') ORDER BY created_at DESC LIMIT 5000"
            )
            
            database_loaded = 0
            
            for signal in signals:
                try:
                    # Convert database signal to prediction format
                    prediction = {
                        'timestamp': signal['created_at'],
                        'symbol': signal['coin'],
                        'predicted_direction': signal['direction'],
                        'confidence': signal['confidence'],
                        'win_probability': signal['ml_prediction'],
                        'expected_return': 0.0,  # Can be extracted from ml_features if needed
                        'model_used': 'database_historical',
                        'features_used': 25,
                        'price': signal['price'],
                        'ton_price': 0.0,  # Can be extracted from ml_features if needed
                        'ratio': signal.get('ratio', 0.0),
                        'source': 'database_migration',
                        'signal_id': signal['id']
                    }
                    
                    # Extract additional data from ml_features JSON if available
                    if signal.get('ml_features'):
                        try:
                            import json
                            features = json.loads(signal['ml_features']) if isinstance(signal['ml_features'], str) else signal['ml_features']
                            prediction.update({
                                'expected_return': features.get('expected_return', 0.0),
                                'ton_price': features.get('ton_price', 0.0),
                                'ratio': features.get('ratio', signal.get('ratio', 0.0)),
                                'model_used': features.get('model_used', 'database_historical')
                            })
                        except:
                            pass  # Continue with basic data if JSON parsing fails
                    
                    self.predictions.append(prediction)
                    database_loaded += 1
                    
                except Exception as pred_error:
                    logger.warning(f"Error processing database signal: {pred_error}")
                    continue
            
            logger.info(f"📊 Loaded {database_loaded} predictions from database")
            return database_loaded
            
        except Exception as e:
            logger.error(f"Error loading from database: {e}")
            return 0
    
    def _load_from_files(self):
        """Fallback: Load prediction history from JSON and CSV files"""
        try:
            json_loaded = 0
            csv_loaded = 0
            
            # Load from JSON history file (original method)
            history_file = self.data_dir / "prediction_history.json"
            
            if history_file.exists():
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    self.predictions = data.get('predictions', [])
                    self.performance_history = data.get('performance_history', [])
                    json_loaded = len(self.predictions)
                
                logger.info(f"📚 Loaded {json_loaded} predictions from JSON history")
            
            # Load from ML signals CSV file (enhanced feature)
            csv_file = self.project_root / "data" / "ml_signals.csv"
            
            if csv_file.exists():
                try:
                    import csv as csv_module
                    with open(csv_file, 'r', newline='', encoding='utf-8') as f:
                        reader = csv_module.DictReader(f)
                        
                        for row in reader:
                            # Convert CSV row to prediction format
                            prediction = {
                                'timestamp': row.get('timestamp', ''),
                                'symbol': row.get('coin', ''),
                                'predicted_direction': self._get_direction_from_recommendation(row.get('recommendation', 'HOLD')),
                                'confidence': float(row.get('confidence', 0.0)),
                                'win_probability': float(row.get('win_probability', 0.0)),
                                'expected_return': float(row.get('expected_return', 0.0)),
                                'model_used': row.get('model_used', 'unknown'),
                                'features_used': int(row.get('features_used', 0)),
                                'price': float(row.get('price', 0.0)),
                                'ton_price': float(row.get('ton_price', 0.0)),
                                'ratio': float(row.get('ratio', 0.0)),
                                'source': 'ml_signals_csv'
                            }
                            
                            # Add to predictions if not already present (avoid duplicates)
                            if not any(p.get('timestamp') == prediction['timestamp'] and 
                                     p.get('symbol') == prediction['symbol'] for p in self.predictions):
                                self.predictions.append(prediction)
                                csv_loaded += 1
                    
                    logger.info(f"📊 Loaded {csv_loaded} additional predictions from ML signals CSV")
                    
                except Exception as csv_error:
                    logger.warning(f"Could not load from ML signals CSV: {csv_error}")
            
            total_loaded = json_loaded + csv_loaded
            logger.info(f"🎯 Total prediction history loaded from files: {total_loaded} entries ({json_loaded} JSON + {csv_loaded} CSV)")
            
            # If we have enough data, consider triggering a retrain
            if total_loaded >= 30:
                logger.info(f"🧠 Sufficient prediction history ({total_loaded}) - models ready for enhanced training")
                
        except Exception as e:
            logger.error(f"Error loading from files: {e}")
            self.predictions = []
            self.performance_history = []
    
    def _get_direction_from_recommendation(self, recommendation):
        """Convert recommendation to direction for training"""
        if recommendation.upper() in ['BUY', 'LONG']:
            return 'LONG'
        elif recommendation.upper() in ['SELL', 'SHORT']:
            return 'SHORT'
        else:
            return 'HOLD'
    
    def predict(self, features):
        """
        Legacy predict method for PSC trader compatibility
        Expected format: [psc_price, ton_price, ratio]
        """
        try:
            if not isinstance(features, (list, tuple)) or len(features) < 3:
                raise ValueError("Features must be list/tuple with at least 3 elements [psc_price, ton_price, ratio]")
            
            psc_price, ton_price, ratio = features[:3]
            
            # Use our advanced prediction method
            prediction_result = self.predict_trade_outcome(psc_price, ton_price, ratio)
            
            # Convert to legacy format expected by PSC trader
            return {
                'confidence': prediction_result['confidence'],
                'prediction': prediction_result['win_probability'],
                'expected_return': prediction_result['expected_return'],
                'win_probability': prediction_result['win_probability'],
                'features_used': prediction_result.get('features_used', 9),
                'model_type': prediction_result.get('model_used', 'ml_engine'),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Legacy predict method error: {e}")
            # Fallback prediction
            return {
                'confidence': 0.5,
                'prediction': 0.5,
                'expected_return': 0.02,
                'win_probability': 0.5,
                'features_used': 3,
                'model_type': 'fallback',
                'timestamp': datetime.now()
            }
    
    def update_prediction_outcome(self, prediction_timestamp, actual_outcome, actual_return):
        """
        Update a prediction with actual outcome for PSC trader compatibility
        
        Args:
            prediction_timestamp: ISO timestamp string or datetime object
            actual_outcome: Boolean - True if trade was profitable
            actual_return: Float - actual return percentage as decimal (e.g., 0.05 for 5%)
        """
        try:
            # Convert timestamp if needed
            if isinstance(prediction_timestamp, str):
                target_time = datetime.fromisoformat(prediction_timestamp.replace('Z', '+00:00'))
            elif isinstance(prediction_timestamp, datetime):
                target_time = prediction_timestamp
            else:
                logger.error(f"Invalid timestamp format: {prediction_timestamp}")
                return False
            
            # Find matching prediction within 2 minutes tolerance
            for record in reversed(self.predictions):
                if not isinstance(record, dict) or 'timestamp' not in record:
                    continue
                    
                record_time_str = record['timestamp']
                if isinstance(record_time_str, str):
                    try:
                        record_time = datetime.fromisoformat(record_time_str.replace('Z', '+00:00'))
                    except:
                        continue
                else:
                    continue
                
                time_diff = abs((target_time - record_time).total_seconds())
                
                # Match within 120 seconds and not already updated
                if time_diff <= 120 and record.get('actual_outcome') is None:
                    # Update the prediction record
                    record['actual_outcome'] = {
                        'profit': bool(actual_outcome),
                        'return': float(actual_return),
                        'action': 'BUY' if actual_outcome else 'SELL',
                        'updated_at': datetime.now().isoformat()
                    }
                    
                    # Update performance tracking
                    self._update_performance_tracking(record['prediction'], record['actual_outcome'])
                    
                    # Save updated history
                    self.save_prediction_history()
                    
                    logger.info(f"✅ Prediction outcome updated: profit={actual_outcome}, return={actual_return:.3f}")
                    
                    # Trigger enhanced retraining more frequently for better accuracy
                    outcomes_count = len([r for r in self.predictions if r.get('actual_outcome') is not None])
                    if outcomes_count > 0 and outcomes_count % 15 == 0:  # Reduced from 20 to 15
                        logger.info(f"🧠 {outcomes_count} outcomes recorded - triggering enhanced model retraining")
                        self.retrain_models()
                    
                    return True
            
            logger.warning(f"No matching prediction found for timestamp {prediction_timestamp}")
            return False
            
        except Exception as e:
            logger.error(f"Error updating prediction outcome: {e}")
            return False

    def should_ml_generate_signal(self, psc_price, ton_price, ratio):
        """
        Determine if ML engine should independently generate a small-move signal
        ENHANCED: Supports both LONG and SHORT signal generation
        Runs continuously to catch opportunities the main PSC scan might miss
        """
        try:
            # Quick prediction for continuous monitoring
            prediction = self.predict([psc_price, ton_price, ratio])
            
            # Extract small-move specific metrics
            small_move_prob = prediction.get('small_move_probability', 0.0)
            expected_return = prediction.get('expected_return', 0.0)
            confidence = prediction.get('confidence', 0.0)
            
            # Log prediction details for debugging
            logger.info(f"🔍 ML Prediction Check - Price: ${psc_price:.6f}, Ratio: {ratio}")
            logger.info(f"   Small-move prob: {small_move_prob:.1%}, Return: {expected_return:.3%}, Confidence: {confidence:.1%}")
            
            # Determine signal direction based on expected return
            signal_direction = "LONG" if expected_return > 0 else "SHORT" if expected_return < 0 else "NEUTRAL"
            
            # STRENGTHENED criteria for quality signals (raised thresholds back to production standards)
            if signal_direction == "LONG":
                criteria = {
                    'high_small_move_confidence': small_move_prob >= 0.65,   # Raised back for quality signals
                    'profitable_expectation': expected_return >= 0.0015,    # Raised back to original threshold
                    'strong_overall_confidence': confidence >= 0.65,        # Raised to match new confidence standards
                    'ratio_threshold': ratio >= 1.25,                      # Raised back to original standard
                    'timer_advantage': self._is_timer_favorable()           # Good timer position
                }
            elif signal_direction == "SHORT":
                criteria = {
                    'high_small_move_confidence': small_move_prob >= 0.65,   # Raised back for quality signals
                    'profitable_expectation': expected_return <= -0.0015,   # Raised back to original threshold
                    'strong_overall_confidence': confidence >= 0.65,        # Raised to match new confidence standards
                    'ratio_threshold': ratio <= 0.9,                       # Lowered back to original standard
                    'timer_advantage': self._is_timer_favorable()           # Good timer position
                }
            else:
                # NEUTRAL signals don't generate independent signals
                logger.info(f"   Signal direction: NEUTRAL - no signal generated")
                return False, prediction
            
            # Must meet most criteria for independent signal (raised threshold for quality signals)
            criteria_met = sum(criteria.values())
            threshold = 4  # Raised from 2 to 4 out of 5 criteria for high-quality signals only
            
            logger.info(f"   Signal direction: {signal_direction}, Criteria met: {criteria_met}/5")
            logger.info(f"   Criteria details: {criteria}")
            
            if criteria_met >= threshold:
                logger.info(f"🎯 ML Independent {signal_direction} Signal Generated! Criteria: {criteria_met}/5 met")
                
                # Add direction info to prediction
                enhanced_prediction = prediction.copy()
                enhanced_prediction['ml_direction'] = signal_direction
                enhanced_prediction['signal_strength'] = criteria_met / 5.0
                
                return True, enhanced_prediction
            else:
                logger.info(f"   Not enough criteria met ({criteria_met} < {threshold}) - no signal")
            
            return False, prediction
            
        except Exception as e:
            logger.error(f"Error in ML signal generation check: {e}")
            return False, {}

    def _is_timer_favorable(self):
        """Check if current timer position is favorable for new positions"""
        current_time = datetime.now()
        timer_minute = current_time.minute % 10
        
        # Favorable: Early in cycle (0-2 min) or mid-cycle opportunity (4-6 min)
        return timer_minute <= 2 or (4 <= timer_minute <= 6)

    async def continuous_market_scan(self, price_fetcher_callback):
        """
        Continuous ML-driven market scanning for small-move opportunities
        Runs independently of main PSC scan to catch additional signals
        """
        try:
            logger.info("🤖 Starting continuous ML market scan...")
            
            # Monitored coins for continuous scanning
            scan_coins = ['BTC', 'ETH', 'SOL', 'SHIB', 'DOGE', 'PEPE']
            scan_interval = 45  # Seconds (different from main PSC scan)
            scan_count = 0
            
            while True:
                try:
                    scan_count += 1
                    logger.info(f"🔄 ML Scan #{scan_count} - Checking {len(scan_coins)} coins...")
                    
                    # Get current prices for all coins
                    prices = {}
                    for coin in scan_coins:
                        price = await price_fetcher_callback(coin)
                        if price:
                            prices[coin] = price
                            logger.info(f"   {coin}: ${price:.6f}")
                    
                    # Get TON price for ratio calculations
                    ton_price = await price_fetcher_callback('TON')
                    if not ton_price:
                        logger.warning("⚠️ TON price not available, skipping scan cycle")
                        await asyncio.sleep(scan_interval)
                        continue
                    
                    logger.info(f"   TON: ${ton_price:.6f}")
                    
                    # Scan each coin for ML-driven opportunities
                    signals_generated = 0
                    for coin, price in prices.items():
                        try:
                            # Calculate ratio (same formula as PSC system)
                            ratio = round(price / (ton_price * 0.001), 2)
                            
                            # Check if ML suggests an independent signal
                            should_signal, prediction = self.should_ml_generate_signal(price, ton_price, ratio)
                            
                            if should_signal:
                                signals_generated += 1
                                # Log ML-generated opportunity
                                logger.info(f"🎯 ML Independent Signal #{signals_generated}: {coin} at ${price:.6f}")
                                logger.info(f"   Ratio: {ratio}, Direction: {prediction.get('ml_direction', 'UNKNOWN')}")
                                
                                # Store ML signal for main system to pick up
                                self._store_ml_signal({
                                    'coin': coin,
                                    'price': price,
                                    'ton_price': ton_price,
                                    'ratio': ratio,
                                    'prediction': prediction,
                                    'timestamp': datetime.now().isoformat(),
                                    'source': 'ML_CONTINUOUS',
                                    'scan_type': 'independent'
                                })
                                
                        except Exception as e:
                            logger.error(f"Error scanning {coin}: {e}")
                            continue
                    
                    if signals_generated == 0:
                        logger.info(f"   No ML signals generated this cycle")
                    else:
                        logger.info(f"✅ Generated {signals_generated} ML signals this cycle")
                    
                    # Wait before next scan
                    logger.info(f"💤 Waiting {scan_interval}s until next ML scan...")
                    await asyncio.sleep(scan_interval)
                    
                except Exception as e:
                    logger.error(f"Error in continuous scan cycle: {e}")
                    await asyncio.sleep(scan_interval)
                    
        except Exception as e:
            logger.error(f"Error in continuous market scan: {e}")

    def _store_ml_signal(self, signal_data):
        """Store ML-generated signal to database for Railway deployment compatibility"""
        try:
            # For Railway deployment, we use database instead of CSV files
            if hasattr(self, 'data_manager') and self.data_manager:
                # Use database storage
                return self._store_ml_signal_to_database(signal_data)
            else:
                # Fallback to CSV for local development
                return self._store_ml_signal_to_csv_fallback(signal_data)
                
        except Exception as e:
            logger.error(f"Error storing ML signal: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")

    def _store_ml_signal_to_database(self, signal_data):
        """Store ML signal directly to database (Railway-ready)"""
        try:
            # Validate and clean the signal data
            cleaned_signal = self._clean_signal_data(signal_data)
            prediction = cleaned_signal.get('prediction', {})
            
            # Prepare signal data for database
            signal_id = self.data_manager.log_ml_signal(
                coin=cleaned_signal.get('coin', ''),
                price=float(cleaned_signal.get('price', 0.0)),
                confidence=float(prediction.get('confidence', 0.0)),
                direction=prediction.get('recommendation', 'HOLD'),
                ml_prediction=float(prediction.get('win_probability', 0.0)),
                ml_features={
                    'signal_type': 'ML_INDEPENDENT',
                    'expected_return': prediction.get('expected_return', 0.0),
                    'small_move_probability': prediction.get('small_move_probability', 0.0),
                    'potential_profit': prediction.get('potential_profit', 0.0),
                    'potential_loss': prediction.get('potential_loss', 0.0),
                    'model_used': prediction.get('model_used', 'unknown'),
                    'features_used': prediction.get('features_used', 0),
                    'ratio_strength': prediction.get('ratio_strength', 0.0),
                    'timer_position': prediction.get('timer_position', 0.0),
                    'market_session': prediction.get('market_session', 'OTHER'),
                    'source': cleaned_signal.get('source', 'ML'),
                    'scan_type': cleaned_signal.get('scan_type', 'continuous'),
                    'ratio': cleaned_signal.get('ratio', 0.0),
                    'ton_price': cleaned_signal.get('ton_price', 0.0)
                }
            )
            
            # Also maintain a recent signals queue in memory
            if not hasattr(self, 'recent_ml_signals'):
                self.recent_ml_signals = []
            
            self.recent_ml_signals.append(cleaned_signal)
            
            # Keep only last 50 signals in memory
            if len(self.recent_ml_signals) > 50:
                self.recent_ml_signals = self.recent_ml_signals[-50:]
                
            logger.info(f"📊 ML signal stored to database: {cleaned_signal.get('coin')} - Win: {prediction.get('win_probability', 0):.3f}, Conf: {prediction.get('confidence', 0):.3f}, ID: {signal_id}")
            return signal_id
            
        except Exception as e:
            logger.error(f"Error storing ML signal to database: {e}")
            # Fallback to CSV if database fails
            return self._store_ml_signal_to_csv_fallback(signal_data)

    def _store_ml_signal_to_csv_fallback(self, signal_data):
        """Fallback CSV storage for local development"""
    def _store_ml_signal_to_csv_fallback(self, signal_data):
        """Fallback CSV storage for local development"""
        try:
            # Create data directory
            data_dir = self.project_root / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Define CSV file path
            csv_file = data_dir / "ml_signals.csv"
            
            # Validate and clean the signal data before storing
            cleaned_signal = self._clean_signal_data(signal_data)
            
            # Extract prediction data
            prediction = cleaned_signal.get('prediction', {})
            
            # Prepare CSV row data
            csv_row = {
                'timestamp': cleaned_signal.get('timestamp', datetime.now().isoformat()),
                'coin': cleaned_signal.get('coin', ''),
                'price': cleaned_signal.get('price', 0.0),
                'ton_price': cleaned_signal.get('ton_price', 0.0),
                'ratio': cleaned_signal.get('ratio', 0.0),
                'win_probability': prediction.get('win_probability', 0.0),
                'confidence': prediction.get('confidence', 0.0),
                'expected_return': prediction.get('expected_return', 0.0),
                'small_move_probability': prediction.get('small_move_probability', 0.0),
                'potential_profit': prediction.get('potential_profit', 0.0),
                'potential_loss': prediction.get('potential_loss', 0.0),
                'recommendation': prediction.get('recommendation', 'HOLD'),
                'model_used': prediction.get('model_used', 'unknown'),
                'features_used': prediction.get('features_used', 0),
                'ratio_strength': prediction.get('ratio_strength', 0.0),
                'timer_position': prediction.get('timer_position', 0.0),
                'market_session': prediction.get('market_session', 'OTHER'),
                'source': cleaned_signal.get('source', 'ML'),
                'scan_type': cleaned_signal.get('scan_type', 'continuous')
            }
            
            # Write to CSV (simplified for fallback)
            file_exists = csv_file.exists()
            with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                csv_headers = list(csv_row.keys())
                writer = csv.DictWriter(f, fieldnames=csv_headers)
                
                if not file_exists:
                    writer.writeheader()
                writer.writerow(csv_row)
                
            logger.info(f"📝 ML signal stored to CSV (fallback): {csv_row['coin']} - Win: {csv_row['win_probability']:.3f}, Conf: {csv_row['confidence']:.3f}")
            
        except Exception as e:
            logger.error(f"Error in CSV fallback storage: {e}")

    def _cleanup_temp_csv_files(self):
        """Clean up any leftover temporary CSV files"""
        try:
            data_dir = self.project_root / "data"
            if data_dir.exists():
                # Find and remove temp files older than 5 minutes
                import time
                current_time = time.time()
                
                for temp_file in data_dir.glob("ml_signals.tmp*"):
                    try:
                        file_age = current_time - temp_file.stat().st_mtime
                        if file_age > 300:  # 5 minutes
                            temp_file.unlink()
                            logger.info(f"Cleaned up old temp file: {temp_file.name}")
                    except Exception as e:
                        logger.warning(f"Could not clean temp file {temp_file}: {e}")
        except Exception as e:
            logger.warning(f"Error during temp file cleanup: {e}")

    def _store_ml_signal_json_fallback(self, signal_data):
        """Fallback JSON storage method if CSV fails"""
        try:
            ml_signals_dir = self.project_root / "data" / "ml_signals"
            ml_signals_dir.mkdir(parents=True, exist_ok=True)
            
            cleaned_signal = self._clean_signal_data(signal_data)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            signal_file = ml_signals_dir / f"ml_signal_{timestamp}.json"
            
            temp_file = signal_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(cleaned_signal, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            
            temp_file.rename(signal_file)
            logger.info(f"📁 Fallback JSON storage: {cleaned_signal['coin']}")
            
        except Exception as e:
            logger.error(f"Fallback JSON storage failed: {e}")

    def _clean_signal_data(self, signal_data):
        """Clean and validate signal data before storage with proper rounding"""
        try:
            cleaned = signal_data.copy()
            
            # Validate and fix prediction values
            if 'prediction' in cleaned:
                pred = cleaned['prediction']
                
                # Fix extremely small or invalid win_probability
                if 'win_probability' in pred:
                    win_prob = pred['win_probability']
                    if win_prob < 1e-10 or win_prob > 1.0 or not isinstance(win_prob, (int, float)):
                        logger.warning(f"Invalid win_probability {win_prob}, setting to 0.5")
                        pred['win_probability'] = 0.5
                    else:
                        # Round to 4 decimal places for readability
                        pred['win_probability'] = round(max(0.0, min(1.0, float(win_prob))), 4)
                
                # Fix confidence values
                if 'confidence' in pred:
                    conf = pred['confidence']
                    if conf < 0 or conf > 1.0 or not isinstance(conf, (int, float)):
                        logger.warning(f"Invalid confidence {conf}, setting to 0.5")
                        pred['confidence'] = 0.5
                    else:
                        # Round to 4 decimal places for readability
                        pred['confidence'] = round(max(0.0, min(1.0, float(conf))), 4)
                
                # Fix expected_return values with better rounding
                if 'expected_return' in pred:
                    ret = pred['expected_return']
                    if not isinstance(ret, (int, float)) or abs(ret) > 1.0:
                        logger.warning(f"Invalid expected_return {ret}, setting to 0.0")
                        pred['expected_return'] = 0.0
                    else:
                        # Round to 6 decimal places but eliminate tiny values
                        ret_cleaned = max(-0.2, min(0.2, float(ret)))
                        if abs(ret_cleaned) < 0.0001:  # Less than 0.01%
                            pred['expected_return'] = 0.0
                        else:
                            pred['expected_return'] = round(ret_cleaned, 6)
                
                # Fix small_move_probability
                if 'small_move_probability' in pred:
                    smp = pred['small_move_probability']
                    if isinstance(smp, (int, float)):
                        pred['small_move_probability'] = round(max(0.0, min(1.0, float(smp))), 4)
                    else:
                        pred['small_move_probability'] = 0.0
                
                # Fix potential_profit and potential_loss
                for key in ['potential_profit', 'potential_loss']:
                    if key in pred:
                        value = pred[key]
                        if isinstance(value, (int, float)):
                            pred[key] = round(max(0.0, float(value)), 2)  # Round to cents
                        else:
                            pred[key] = 0.0
                
                # Fix ratio_strength and timer_position
                for key in ['ratio_strength', 'timer_position']:
                    if key in pred:
                        value = pred[key]
                        if isinstance(value, (int, float)):
                            pred[key] = round(max(0.0, min(1.0, float(value))), 3)
                        else:
                            pred[key] = 0.0
                
                # Fix any other extremely small numbers
                for key, value in pred.items():
                    if isinstance(value, float):
                        if abs(value) < 1e-10 and value != 0.0:
                            logger.warning(f"Fixing extremely small value for {key}: {value}")
                            pred[key] = 0.0
                        elif 'probability' in key or 'confidence' in key:
                            pred[key] = round(value, 4)  # 4 decimals for probabilities
                        elif 'return' in key:
                            pred[key] = round(value, 6)  # 6 decimals for returns
                        elif key in ['ratio', 'price', 'ton_price']:
                            pred[key] = round(value, 8)  # 8 decimals for prices (crypto needs precision)
                        else:
                            pred[key] = round(value, 4)  # Default 4 decimals
            
            # Clean top-level numeric values too
            for key in ['price', 'ton_price', 'ratio']:
                if key in cleaned:
                    value = cleaned[key]
                    if isinstance(value, (int, float)):
                        if key == 'price' and value < 0.001:  # Handle very small coin prices
                            cleaned[key] = round(value, 10)  # More precision for small prices
                        elif key in ['price', 'ton_price']:
                            cleaned[key] = round(value, 8)  # Standard crypto precision
                        else:
                            cleaned[key] = round(value, 6)
            
            # Ensure all numeric values are JSON-serializable
            cleaned = self._ensure_json_serializable(cleaned)
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Error cleaning signal data: {e}")
            return signal_data  # Return original if cleaning fails

    def _ensure_json_serializable(self, obj):
        """Ensure all values in the object are JSON serializable"""
        if isinstance(obj, dict):
            return {k: self._ensure_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._ensure_json_serializable(v) for v in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif NUMPY_AVAILABLE and hasattr(np, 'integer') and isinstance(obj, np.integer):
            return int(obj)
        elif NUMPY_AVAILABLE and hasattr(np, 'floating') and isinstance(obj, np.floating):
            return float(obj)
        elif NUMPY_AVAILABLE and hasattr(np, 'bool_') and isinstance(obj, np.bool_):
            return bool(obj)
        elif NUMPY_AVAILABLE and hasattr(np, 'ndarray') and isinstance(obj, np.ndarray):
            return obj.tolist()
        # Legacy NumPy compatibility (for older versions)
        elif NUMPY_AVAILABLE:
            try:
                if hasattr(np, 'int_') and isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
                    return int(obj)
                elif hasattr(np, 'float64') and isinstance(obj, (np.float16, np.float32, np.float64)):
                    return float(obj)
            except:
                pass
        return obj

    def get_recent_ml_signals(self, max_age_minutes=10):
        """Get recent ML-generated signals from CSV file"""
        try:
            csv_file = Path("data/ml_signals.csv")
            
            # If CSV file doesn't exist, return memory cache or empty list
            if not csv_file.exists():
                if hasattr(self, 'recent_ml_signals'):
                    return self.recent_ml_signals
                return []
            
            cutoff_time = datetime.now() - timedelta(minutes=max_age_minutes)
            fresh_signals = []
            
            # Read from CSV file
            with open(csv_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    try:
                        # Parse timestamp
                        signal_time = datetime.fromisoformat(row['timestamp'])
                        
                        # Check if signal is still fresh
                        if signal_time > cutoff_time:
                            # Convert CSV row back to signal format
                            signal = {
                                'coin': row['coin'],
                                'price': float(row['price']),
                                'ton_price': float(row['ton_price']),
                                'ratio': float(row['ratio']),
                                'timestamp': row['timestamp'],
                                'source': row['source'],
                                'scan_type': row['scan_type'],
                                'prediction': {
                                    'win_probability': float(row['win_probability']),
                                    'confidence': float(row['confidence']),
                                    'expected_return': float(row['expected_return']),
                                    'small_move_probability': float(row.get('small_move_probability', 0.0)),
                                    'potential_profit': float(row.get('potential_profit', 0.0)),
                                    'potential_loss': float(row.get('potential_loss', 0.0)),
                                    'recommendation': row['recommendation'],
                                    'model_used': row['model_used'],
                                    'features_used': int(row['features_used']),
                                    'ratio_strength': float(row.get('ratio_strength', 0.0)),
                                    'timer_position': float(row.get('timer_position', 0.0)),
                                    'market_session': row.get('market_session', 'OTHER')
                                }
                            }
                            fresh_signals.append(signal)
                            
                    except (ValueError, KeyError) as e:
                        logger.warning(f"Error parsing CSV row: {e}")
                        continue
            
            # Sort by timestamp (most recent first)
            fresh_signals.sort(key=lambda x: x['timestamp'], reverse=True)
            
            return fresh_signals
            
        except Exception as e:
            logger.error(f"Error reading ML signals from CSV: {e}")
            
            # Fallback to memory cache if available
            if hasattr(self, 'recent_ml_signals'):
                cutoff_time = datetime.now() - timedelta(minutes=max_age_minutes)
                fresh_signals = []
                for signal in self.recent_ml_signals:
                    try:
                        signal_time = datetime.fromisoformat(signal['timestamp'])
                        if signal_time > cutoff_time:
                            fresh_signals.append(signal)
                    except:
                        continue
                return fresh_signals
            
            return []

    def validate_ml_signal_with_tradingview(self, ml_signal, tradingview_data):
        """
        Validate ML-generated signal against TradingView sentiment
        ENHANCED: Supports both LONG and SHORT signal validation
        Returns enhanced signal if validation passes
        """
        try:
            coin = ml_signal['coin']
            ml_prediction = ml_signal['prediction']
            
            # Get TradingView data for this coin
            tv_coin_data = tradingview_data.get(coin, {})
            if not tv_coin_data:
                logger.warning(f"No TradingView data available for {coin}")
                return None
            
            # Extract TradingView consensus
            consensus = tv_coin_data.get('consensus', {})
            trade_signals = tv_coin_data.get('trade_signals', {})
            
            tv_direction = consensus.get('direction', 'neutral').upper()
            tv_strength = consensus.get('strength', 0.0)
            tv_confidence = consensus.get('confidence', 0.0)
            timeframe_alignment = trade_signals.get('timeframe_alignment', False)
            entry_recommendation = trade_signals.get('entry_recommendation', 'hold')
            
            # Determine ML signal direction based on expected return
            ml_expected_return = ml_prediction.get('expected_return', 0)
            ml_direction = "LONG" if ml_expected_return > 0 else "SHORT" if ml_expected_return < 0 else "NEUTRAL"
            
            # Validation criteria (enhanced for both directions)
            validation_score = 0
            max_score = 5
            
            # 1. Direction alignment (ENHANCED: supports both LONG and SHORT)
            if ml_direction == "LONG" and tv_direction in ['BUY', 'STRONG_BUY']:
                validation_score += 1
                logger.info(f"✅ Direction alignment: ML LONG, TV {tv_direction}")
            elif ml_direction == "SHORT" and tv_direction in ['SELL', 'STRONG_SELL']:
                validation_score += 1
                logger.info(f"✅ Direction alignment: ML SHORT, TV {tv_direction}")
            elif ml_direction == "LONG" and tv_direction in ['NEUTRAL'] and ml_expected_return > 0.0015:
                validation_score += 0.5  # Partial credit for strong ML signal with neutral TV
                logger.info(f"⚠️ Partial alignment: Strong ML LONG, TV Neutral")
            elif ml_direction == "SHORT" and tv_direction in ['NEUTRAL'] and ml_expected_return < -0.0015:
                validation_score += 0.5  # Partial credit for strong ML signal with neutral TV
                logger.info(f"⚠️ Partial alignment: Strong ML SHORT, TV Neutral")
            
            # 2. TradingView strength supports the move
            if tv_strength >= 0.6:  # Strong TradingView signal
                validation_score += 1
                logger.info(f"✅ TradingView strength: {tv_strength:.1%}")
            
            # 3. Timeframe alignment (multiple timeframes agree)
            if timeframe_alignment:
                validation_score += 1
                logger.info(f"✅ Timeframe alignment confirmed")
            
            # 4. Entry recommendation is favorable (ENHANCED: supports both directions, filters weak HOLD signals)
            if ml_direction == "LONG" and entry_recommendation in ['buy', 'strong_buy']:
                validation_score += 1
                logger.info(f"✅ Entry recommendation: {entry_recommendation}")
            elif ml_direction == "SHORT" and entry_recommendation in ['sell', 'strong_sell']:
                validation_score += 1
                logger.info(f"✅ Entry recommendation: {entry_recommendation}")
            elif entry_recommendation == 'hold':
                # Only use HOLD signals for training if they have exceptional quality
                if (abs(ml_expected_return) > 0.002 and  # Strong expected return (>0.2%)
                    ml_prediction.get('confidence', 0) > 0.75 and  # Very high ML confidence
                    tv_strength > 0.7):  # Strong TradingView consensus despite hold
                    validation_score += 0.3  # Reduced credit for quality HOLD signals
                    logger.info(f"✅ Quality HOLD for training: Strong ML + TV consensus")
                else:
                    logger.info(f"❌ Weak HOLD signal filtered out for training")
            
            # 5. TradingView confidence is reasonable
            if tv_confidence >= 0.5:
                validation_score += 1
                logger.info(f"✅ TradingView confidence: {tv_confidence:.1%}")
            
            # Require majority validation (3/5 criteria, adjusted for partial scores)
            if validation_score >= 3.0:
                # Enhance ML prediction with TradingView data
                enhanced_prediction = ml_prediction.copy()
                enhanced_prediction.update({
                    'tradingview_validation': True,
                    'tradingview_score': validation_score,
                    'tradingview_direction': tv_direction,
                    'tradingview_strength': tv_strength,
                    'tradingview_confidence': tv_confidence,
                    'timeframe_alignment': timeframe_alignment,
                    'combined_confidence': (ml_prediction.get('confidence', 0) + tv_confidence) / 2,
                    'ml_direction': ml_direction,
                    'validation_timestamp': datetime.now().isoformat()
                })
                
                logger.info(f"🎯 ML signal validated by TradingView: {coin} {ml_direction} ({validation_score:.1f}/{max_score} criteria)")
                return {
                    **ml_signal,
                    'prediction': enhanced_prediction,
                    'validation_status': 'PASSED',
                    'validation_score': validation_score
                }
            else:
                logger.warning(f"❌ ML signal validation failed: {coin} {ml_direction} ({validation_score:.1f}/{max_score} criteria)")
                return None
                
        except Exception as e:
            logger.error(f"Error validating ML signal with TradingView: {e}")
            return None

    def validate_ml_signal_with_single_coin(self, ml_signal, coin_analysis):
        """
        Validate ML-generated signal against single coin TradingView analysis (optimized)
        Returns enhanced signal if validation passes
        """
        try:
            if not coin_analysis or 'consensus' not in coin_analysis:
                logger.warning(f"No TradingView data available for validation")
                return None
            
            coin = ml_signal['coin']
            ml_prediction = ml_signal['prediction']
            
            # Extract TradingView consensus from single coin analysis
            consensus = coin_analysis.get('consensus', {})
            trade_signals = coin_analysis.get('trade_signals', {})
            
            tv_direction = consensus.get('direction', 'NEUTRAL').upper()
            tv_strength = consensus.get('strength', 0.0)
            tv_confidence = consensus.get('confidence', 0.0)
            timeframe_alignment = trade_signals.get('timeframe_alignment', False)
            entry_recommendation = trade_signals.get('entry_recommendation', 'hold')
            
            # Determine ML signal direction based on expected return
            ml_expected_return = ml_prediction.get('expected_return', 0)
            ml_direction = "LONG" if ml_expected_return > 0 else "SHORT" if ml_expected_return < 0 else "NEUTRAL"
            
            # Validation criteria (same as original but optimized for single coin)
            validation_score = 0
            max_score = 5
            
            # 1. Direction alignment
            if ml_direction == "LONG" and tv_direction in ['BUY', 'STRONG_BUY']:
                validation_score += 1
                logger.info(f"✅ Direction alignment: ML LONG, TV {tv_direction}")
            elif ml_direction == "SHORT" and tv_direction in ['SELL', 'STRONG_SELL']:
                validation_score += 1
                logger.info(f"✅ Direction alignment: ML SHORT, TV {tv_direction}")
            elif ml_direction == "LONG" and tv_direction in ['NEUTRAL'] and ml_expected_return > 0.0015:
                validation_score += 0.5
                logger.info(f"⚠️ Partial alignment: Strong ML LONG, TV Neutral")
            elif ml_direction == "SHORT" and tv_direction in ['NEUTRAL'] and ml_expected_return < -0.0015:
                validation_score += 0.5
                logger.info(f"⚠️ Partial alignment: Strong ML SHORT, TV Neutral")
            
            # 2. TradingView strength supports the move
            if tv_strength >= 0.6:
                validation_score += 1
                logger.info(f"✅ TradingView strength: {tv_strength:.1%}")
            
            # 3. Timeframe alignment
            if timeframe_alignment:
                validation_score += 1
                logger.info(f"✅ Timeframe alignment confirmed")
            
            # 4. Entry recommendation is favorable
            if ml_direction == "LONG" and entry_recommendation in ['buy', 'strong_buy']:
                validation_score += 1
                logger.info(f"✅ Entry recommendation: {entry_recommendation}")
            elif ml_direction == "SHORT" and entry_recommendation in ['sell', 'strong_sell']:
                validation_score += 1
                logger.info(f"✅ Entry recommendation: {entry_recommendation}")
            elif entry_recommendation == 'hold' and abs(ml_expected_return) > 0.0015:
                validation_score += 0.5
                logger.info(f"⚠️ Partial entry recommendation: {entry_recommendation} with strong ML")
            
            # 5. TradingView confidence is reasonable
            if tv_confidence >= 0.5:
                validation_score += 1
                logger.info(f"✅ TradingView confidence: {tv_confidence:.1%}")
            
            # Require majority validation (3/5 criteria)
            if validation_score >= 3.0:
                # Enhance ML prediction with TradingView data
                enhanced_prediction = ml_prediction.copy()
                enhanced_prediction.update({
                    'tradingview_validation': True,
                    'tradingview_score': validation_score,
                    'tradingview_direction': tv_direction,
                    'tradingview_strength': tv_strength,
                    'tradingview_confidence': tv_confidence,
                    'timeframe_alignment': timeframe_alignment,
                    'combined_confidence': (ml_prediction.get('confidence', 0) + tv_confidence) / 2,
                    'ml_direction': ml_direction,
                    'validation_timestamp': datetime.now().isoformat()
                })
                
                logger.info(f"🎯 ML signal validated by TradingView: {coin} {ml_direction} ({validation_score:.1f}/{max_score} criteria)")
                return {
                    **ml_signal,
                    'prediction': enhanced_prediction,
                    'validation_status': 'PASSED',
                    'validation_score': validation_score
                }
            else:
                logger.warning(f"❌ ML signal validation failed: {coin} {ml_direction} ({validation_score:.1f}/{max_score} criteria)")
                return None
                
        except Exception as e:
            logger.error(f"Error validating ML signal with single coin TradingView: {e}")
            return None


# Global ML engine instance removed - use proper initialization in main system
# ml_engine = MLEngine()  # Disabled to prevent duplicate instances
