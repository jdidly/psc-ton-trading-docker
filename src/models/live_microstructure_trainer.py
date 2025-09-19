#!/usr/bin/env python3
"""
Live Microstructure Trainer - Enhanced PSC Integration
======================================================
Enhanced microstructure system that integrates with PSC trading logic,
ML engine, and SuperP technology for optimal signal generation.

Integration Points:
- PSC ratio calculations with logarithmic scaling
- ML confidence scoring aligned with 0.1% profit targets
- Timer-based trading windows (10-minute cycles)
- Bidirectional signal generation (LONG/SHORT)
- SuperP leverage optimization
"""

import asyncio
import json
import time
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import aiohttp
import yaml
from pathlib import Path
import logging
import warnings
import sys
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

warnings.filterwarnings('ignore')

# Setup logging
project_root = Path(__file__).parent.parent.parent
logs_dir = project_root / 'logs'
logs_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_dir / 'microstructure_live_trainer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# PSC System Integration Data Classes
@dataclass
class PSCSignal:
    """PSC-aligned signal structure"""
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    psc_ratio: float
    confidence_score: float
    leverage: float
    position_size: float
    timer_window: str
    signal_strength: str
    reasons: List[str]
    microstructure_score: float
    ml_validation: Optional[float] = None

@dataclass 
class TimerStatus:
    """Timer-based trading window status"""
    current_minute: int
    window_type: str  # 'ENTRY_WINDOW', 'MID_TIMER', 'LATE_TIMER', 'COOLDOWN'
    efficiency: float
    seconds_remaining: int

# Leverage ranges aligned with PSC system
class LeverageRanges:
    CONSERVATIVE = (10, 100)      # 10x-100x for low confidence
    MODERATE = (100, 1000)        # 100x-1000x for medium confidence  
    AGGRESSIVE = (1000, 5000)     # 1000x-5000x for high confidence
    EXTREME = (5000, 10000)       # 5000x-10000x for maximum confidence

class LiveMicrostructureTrainer:
    def __init__(self, data_manager=None):
        """Initialize live microstructure trainer with PSC system integration"""
        # Debug logging for data_manager
        logger = logging.getLogger(__name__)
        logger.debug(f"LiveMicrostructureTrainer.__init__ called with data_manager: {data_manager}")
        logger.debug(f"data_manager type: {type(data_manager)}")
        if data_manager is not None:
            logger.debug(f"data_manager has connection: {hasattr(data_manager, 'connection')}")
            if hasattr(data_manager, 'connection'):
                logger.debug(f"data_manager connection is: {data_manager.connection}")
        
        self.config = self.load_config()
        self.data_manager = data_manager  # For database storage
        
        # Additional debug check
        if self.data_manager is not None:
            logger.info("✅ LiveMicrostructureTrainer initialized WITH data_manager")
        else:
            logger.warning("⚠️ LiveMicrostructureTrainer initialized WITHOUT data_manager")
        
        # Current trading coins from PSC system with enhanced metadata
        self.trading_coins = [
            {
                'symbol': 'BTC', 
                'pair': 'BTCUSDT', 
                'volatility': 'Medium',
                'psc_weight': 1.0,
                'ml_priority': 'high'
            },
            {
                'symbol': 'ETH', 
                'pair': 'ETHUSDT', 
                'volatility': 'Medium',
                'psc_weight': 1.0,
                'ml_priority': 'high'
            },
            {
                'symbol': 'SOL', 
                'pair': 'SOLUSDT', 
                'volatility': 'High',
                'psc_weight': 0.8,
                'ml_priority': 'medium'
            },
            {
                'symbol': 'SHIB', 
                'pair': 'SHIBUSDT', 
                'volatility': 'Very High',
                'psc_weight': 0.6,
                'ml_priority': 'medium'
            },
            {
                'symbol': 'DOGE', 
                'pair': 'DOGEUSDT', 
                'volatility': 'High',
                'psc_weight': 0.7,
                'ml_priority': 'medium'
            },
            {
                'symbol': 'PEPE', 
                'pair': 'PEPEUSDT', 
                'volatility': 'Extreme',
                'psc_weight': 0.5,
                'ml_priority': 'low'
            }
        ]
        
        # PSC system alignment
        self.min_signal_ratio = self.config.get('trading', {}).get('min_signal_ratio', 7.0)
        self.min_confidence_threshold = self.config.get('trading', {}).get('min_confidence_threshold', 0.5)
        
        # Confidence scoring methodology
        # Confidence represents probability of achieving >0.1% profit (system break-even threshold)
        # Aligned with main PSC system where 0.1% = break-even, >0.12% = profitable
        # Factors: microstructure quality + PSC ratio strength + market conditions + timer efficiency
        
        # SuperP configuration
        self.superp_enabled = self.config.get('superp', {}).get('enabled', True)
        self.max_leverage = self.config.get('superp', {}).get('max_leverage', 10000)
        self.time_limit_minutes = self.config.get('superp', {}).get('time_limit_minutes', 10)
        
        # Enhanced training data storage
        self.order_book_history = []
        self.feature_history = []
        self.signal_history = []
        self.psc_signals = []  # PSC-aligned signals
        
        # Performance tracking
        self.model_accuracy = 0.0
        self.training_samples = 0
        self.signals_generated = 0
        self.psc_alignment_score = 0.0
        
        # TON price for PSC ratio calculations
        self.ton_price = 0.0
        
        logger.info(f"Enhanced Live Microstructure Trainer initialized")
        logger.info(f"Trading coins: {[coin['symbol'] for coin in self.trading_coins]}")
        logger.info(f"PSC integration: min_ratio={self.min_signal_ratio}, min_confidence={self.min_confidence_threshold}")
        logger.info(f"SuperP enabled: {self.superp_enabled}")
        if self.superp_enabled:
            logger.info(f"Max leverage: {self.max_leverage}x, Timer limit: {self.time_limit_minutes} min")

    def load_config(self):
        """Load configuration from settings.yaml"""
        try:
            # Go up to project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / 'config' / 'settings.yaml'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            return {}
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
            return {}

    async def get_ton_price(self):
        """Get current TON price for PSC ratio calculations"""
        try:
            url = "https://api.binance.com/api/v3/ticker/price"
            params = {'symbol': 'TONUSDT'}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.ton_price = float(data['price'])
                        return self.ton_price
        except Exception as e:
            logger.error(f"Error fetching TON price: {e}")
            # Fallback to last known price or default
            if self.ton_price == 0.0:
                self.ton_price = 6.5  # Approximate fallback
        return self.ton_price

    def calculate_psc_ratio(self, current_price: float) -> float:
        """Calculate PSC ratio using logarithmic scaling (aligned with PSC system)"""
        try:
            if self.ton_price <= 0:
                return 0.0
            
            # UPDATED: Logarithmic PSC ratio for meaningful comparisons
            log_ratio = math.log10(current_price) - math.log10(self.ton_price)
            psc_ratio = log_ratio + 6  # Shift to positive scale (range ~1-11)
            
            return round(psc_ratio, 3)
        except Exception as e:
            logger.error(f"Error calculating PSC ratio: {e}")
            return 0.0

    def get_timer_status(self) -> TimerStatus:
        """Get current timer window status (10-minute cycles)"""
        current_time = datetime.now()
        current_minute = current_time.minute % 10
        current_second = current_time.second
        
        if current_minute <= 3:
            window_type = "ENTRY_WINDOW"
            efficiency = 1.0
        elif current_minute <= 6:
            window_type = "MID_TIMER"
            efficiency = 0.8
        elif current_minute <= 8:
            window_type = "LATE_TIMER"
            efficiency = 0.6
        else:
            window_type = "COOLDOWN"
            efficiency = 0.3
        
        # Calculate seconds remaining in current window
        if current_minute <= 3:
            seconds_remaining = (4 - current_minute) * 60 - current_second
        elif current_minute <= 6:
            seconds_remaining = (7 - current_minute) * 60 - current_second
        elif current_minute <= 8:
            seconds_remaining = (9 - current_minute) * 60 - current_second
        else:
            seconds_remaining = (10 - current_minute) * 60 - current_second
        
        return TimerStatus(
            current_minute=current_minute,
            window_type=window_type,
            efficiency=efficiency,
            seconds_remaining=max(0, seconds_remaining)
        )

    async def get_live_order_book(self, symbol, limit=100):
        """Get live order book data from Binance API"""
        try:
            url = f"https://api.binance.com/api/v3/depth"
            params = {'symbol': symbol, 'limit': limit}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        timestamp_info = self.get_current_timestamp()
                        return {
                            'symbol': symbol,
                            **timestamp_info,
                            'bids': [[float(price), float(qty)] for price, qty in data['bids']],
                            'asks': [[float(price), float(qty)] for price, qty in data['asks']]
                        }
        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {e}")
        return None

    def get_current_timestamp(self):
        """Get properly formatted timestamp"""
        return {
            'timestamp': datetime.now().isoformat(),
            'timestamp_unix': time.time()
        }
        """Extract advanced microstructure features from order book"""
        try:
            bids = np.array(order_book['bids'])
            asks = np.array(order_book['asks'])
            
            if len(bids) == 0 or len(asks) == 0:
                return None
            
            best_bid = bids[0][0]
            best_ask = asks[0][0]
            mid_price = (best_bid + best_ask) / 2
            
            # Basic features
            spread = best_ask - best_bid
            spread_bps = (spread / mid_price) * 10000
            
            # Order book imbalance
            bid_volume_1 = bids[0][1] if len(bids) > 0 else 0
            ask_volume_1 = asks[0][1] if len(asks) > 0 else 0
            total_volume_1 = bid_volume_1 + ask_volume_1
            imbalance = (bid_volume_1 - ask_volume_1) / total_volume_1 if total_volume_1 > 0 else 0
            
            # Depth analysis (top 10 levels)
            bid_depth = np.sum(bids[:10, 1]) if len(bids) >= 10 else np.sum(bids[:, 1])
            ask_depth = np.sum(asks[:10, 1]) if len(asks) >= 10 else np.sum(asks[:, 1])
            total_depth = bid_depth + ask_depth
            depth_imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0
            
            # Price pressure features
            bid_pressure = np.sum([qty for price, qty in bids[:5]]) if len(bids) >= 5 else 0
            ask_pressure = np.sum([qty for price, qty in asks[:5]]) if len(asks) >= 5 else 0
            pressure_ratio = bid_pressure / ask_pressure if ask_pressure > 0 else 1
            
            # Advanced features
            features = {
                'symbol': order_book['symbol'],
                'timestamp': order_book['timestamp'],
                'mid_price': mid_price,
                'spread_bps': spread_bps,
                'imbalance': imbalance,
                'depth_imbalance': depth_imbalance,
                'bid_pressure': bid_pressure,
                'ask_pressure': ask_pressure,
                'pressure_ratio': pressure_ratio,
                'bid_depth': bid_depth,
                'ask_depth': ask_depth,
                'volatility_proxy': spread_bps * abs(imbalance)  # Volatility estimate
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None

    def generate_psc_aligned_signal(self, features, coin_data) -> Optional[PSCSignal]:
        """Generate PSC-aligned trading signal with bidirectional logic"""
        try:
            # Get current timer status
            timer_status = self.get_timer_status()
            
            # Skip if not in optimal trading window
            if timer_status.window_type not in ['ENTRY_WINDOW', 'MID_TIMER']:
                return None
            
            # Calculate PSC ratio
            psc_ratio = self.calculate_psc_ratio(features['mid_price'])
            if psc_ratio <= 0:
                return None
            
            # Get coin-specific parameters
            volatility = coin_data['volatility']
            psc_weight = coin_data['psc_weight']
            ml_priority = coin_data['ml_priority']
            
            # Enhanced microstructure scoring
            microstructure_score = self.calculate_microstructure_score(features, volatility)
            
            # Apply PSC weighting
            weighted_score = microstructure_score * psc_weight
            
            # PSC-aligned bidirectional signal logic
            signal = None
            
            # LONG Signal Logic (PSC ratio >= 6.5 indicates outperformance vs TON)
            if psc_ratio >= self.min_signal_ratio:
                signal = self.generate_long_signal(
                    features, coin_data, psc_ratio, weighted_score, timer_status
                )
            
            # SHORT Signal Logic (PSC ratio <= 5.5 indicates underperformance vs TON)  
            elif psc_ratio <= (self.min_signal_ratio - 2.0):
                signal = self.generate_short_signal(
                    features, coin_data, psc_ratio, weighted_score, timer_status
                )
            
            # Add ML validation if signal generated
            if signal:
                signal.ml_validation = self.get_ml_confidence_estimate(features, signal.direction)
                
                # Final confidence adjustment based on ML validation
                if signal.ml_validation:
                    signal.confidence_score = min(0.95, 
                        (signal.confidence_score + signal.ml_validation) / 2
                    )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating PSC signal: {e}")
            return None

    def generate_long_signal(self, features, coin_data, psc_ratio, microstructure_score, timer_status) -> Optional[PSCSignal]:
        """Generate LONG signal following PSC logic with 0.1% profit-aligned confidence"""
        try:
            symbol = features['symbol']
            volatility = coin_data['volatility']
            
            # Calculate probability of achieving >0.1% profit based on microstructure + PSC alignment
            # This aligns with system-wide 0.1% break-even / >0.12% profit targets
            
            # Base profit probability from microstructure quality
            microstructure_profit_prob = min(0.7, microstructure_score / 12.0)  # More conservative scaling
            
            # PSC ratio strength - higher ratio increases profit probability for LONG
            ratio_strength = min(0.25, max(0, (psc_ratio - self.min_signal_ratio) * 0.08))
            
            # Market conditions adjustment - spread and volatility impact profit probability
            market_conditions_boost = 0.0
            if features['spread_bps'] < 5.0:  # Tight spread = better execution
                market_conditions_boost += 0.05
            if 0.02 < volatility < 0.08:  # Optimal volatility range for 0.1%+ moves
                market_conditions_boost += 0.08
            elif volatility > 0.1:  # High volatility reduces profit probability
                market_conditions_boost -= 0.1
                
            # Timer efficiency - represents optimal entry timing for profit achievement
            timer_profit_multiplier = timer_status.efficiency
            
            # Final confidence = probability of achieving >0.1% profit
            confidence = (microstructure_profit_prob + ratio_strength + market_conditions_boost) * timer_profit_multiplier
            
            # Check minimum thresholds
            if confidence < self.min_confidence_threshold:
                return None
            
            # Determine signal strength and leverage
            if psc_ratio >= 8.0 and confidence > 0.7:
                strength = "STRONG"
                leverage_range = LeverageRanges.AGGRESSIVE
            elif psc_ratio >= 7.5 and confidence > 0.6:
                strength = "GOOD"
                leverage_range = LeverageRanges.MODERATE
            else:
                strength = "ENTRY"
                leverage_range = LeverageRanges.CONSERVATIVE
            
            # Calculate optimal leverage
            leverage = self.calculate_optimal_leverage(
                confidence, volatility, leverage_range, timer_status
            )
            
            # Position sizing
            position_size = self.calculate_position_size(confidence, leverage, volatility)
            
            # Generate reasons
            reasons = [
                f"PSC ratio {psc_ratio:.3f} > {self.min_signal_ratio} (outperforming TON)",
                f"Microstructure score: {microstructure_score:.1f}/10",
                f"Timer window: {timer_status.window_type} ({timer_status.efficiency:.0%} efficiency)",
                f"Spread: {features['spread_bps']:.2f} bps",
                f"Order imbalance: {features['imbalance']:.3f}"
            ]
            
            return PSCSignal(
                symbol=symbol,
                direction="LONG",
                psc_ratio=psc_ratio,
                confidence_score=confidence,
                leverage=leverage,
                position_size=position_size,
                timer_window=timer_status.window_type,
                signal_strength=strength,
                reasons=reasons,
                microstructure_score=microstructure_score
            )
            
        except Exception as e:
            logger.error(f"Error generating LONG signal: {e}")
            return None

    def generate_short_signal(self, features, coin_data, psc_ratio, microstructure_score, timer_status) -> Optional[PSCSignal]:
        """Generate SHORT signal following PSC logic with 0.1% profit-aligned confidence"""
        try:
            symbol = features['symbol']
            volatility = coin_data['volatility']
            
            # Calculate probability of achieving >0.1% profit based on microstructure + PSC alignment
            # This aligns with system-wide 0.1% break-even / >0.12% profit targets
            
            # Base profit probability from microstructure quality
            microstructure_profit_prob = min(0.7, microstructure_score / 12.0)  # More conservative scaling
            
            # PSC ratio strength - lower ratio increases profit probability for SHORT
            ratio_strength = min(0.25, max(0, (self.min_signal_ratio - 2.0 - psc_ratio) * 0.08))
            
            # Market conditions adjustment - spread and volatility impact profit probability
            market_conditions_boost = 0.0
            if features['spread_bps'] < 5.0:  # Tight spread = better execution
                market_conditions_boost += 0.05
            if 0.02 < volatility < 0.08:  # Optimal volatility range for 0.1%+ moves
                market_conditions_boost += 0.08
            elif volatility > 0.1:  # High volatility reduces profit probability
                market_conditions_boost -= 0.1
                
            # Timer efficiency - represents optimal entry timing for profit achievement
            timer_profit_multiplier = timer_status.efficiency
            
            # Final confidence = probability of achieving >0.1% profit
            confidence = (microstructure_profit_prob + ratio_strength + market_conditions_boost) * timer_profit_multiplier
            
            # Check minimum thresholds
            if confidence < self.min_confidence_threshold:
                return None
            
            # Determine signal strength and leverage
            if psc_ratio <= 4.5 and confidence > 0.7:
                strength = "STRONG"
                leverage_range = LeverageRanges.AGGRESSIVE
            elif psc_ratio <= 5.0 and confidence > 0.6:
                strength = "GOOD"
                leverage_range = LeverageRanges.MODERATE
            else:
                strength = "ENTRY"
                leverage_range = LeverageRanges.CONSERVATIVE
            
            # Calculate optimal leverage
            leverage = self.calculate_optimal_leverage(
                confidence, volatility, leverage_range, timer_status
            )
            
            # Position sizing
            position_size = self.calculate_position_size(confidence, leverage, volatility)
            
            # Generate reasons
            reasons = [
                f"PSC ratio {psc_ratio:.3f} < {self.min_signal_ratio - 2.0} (underperforming TON)",
                f"Microstructure score: {microstructure_score:.1f}/10",
                f"Timer window: {timer_status.window_type} ({timer_status.efficiency:.0%} efficiency)",
                f"Spread: {features['spread_bps']:.2f} bps",
                f"Order imbalance: {features['imbalance']:.3f}"
            ]
            
            return PSCSignal(
                symbol=symbol,
                direction="SHORT",
                psc_ratio=psc_ratio,
                confidence_score=confidence,
                leverage=leverage,
                position_size=position_size,
                timer_window=timer_status.window_type,
                signal_strength=strength,
                reasons=reasons,
                microstructure_score=microstructure_score
            )
            
        except Exception as e:
            logger.error(f"Error generating SHORT signal: {e}")
            return None

    def calculate_microstructure_score(self, features, volatility) -> float:
        """Calculate comprehensive microstructure score (0-10 scale)"""
        try:
            score = 0.0
            
            # Volatility multipliers
            vol_multipliers = {
                'Medium': 1.0,
                'High': 1.2,
                'Very High': 1.5,
                'Extreme': 2.0
            }
            vol_mult = vol_multipliers.get(volatility, 1.0)
            
            # 1. Spread quality (0-3 points)
            if features['spread_bps'] < 1.0:
                score += 3.0
            elif features['spread_bps'] < 3.0:
                score += 2.0
            elif features['spread_bps'] < 5.0:
                score += 1.0
            
            # 2. Order book imbalance (0-3 points)
            imbalance_abs = abs(features['imbalance'])
            if imbalance_abs > 0.5:
                score += 3.0
            elif imbalance_abs > 0.3:
                score += 2.0
            elif imbalance_abs > 0.1:
                score += 1.0
            
            # 3. Depth quality (0-2 points)
            if abs(features['depth_imbalance']) > 0.3:
                score += 2.0
            elif abs(features['depth_imbalance']) > 0.1:
                score += 1.0
            
            # 4. Pressure analysis (0-2 points)
            if features['pressure_ratio'] > 2.0 or features['pressure_ratio'] < 0.5:
                score += 2.0
            elif features['pressure_ratio'] > 1.5 or features['pressure_ratio'] < 0.67:
                score += 1.0
            
            # Apply volatility adjustment
            score *= vol_mult
            
            return min(10.0, score)
            
        except Exception as e:
            logger.error(f"Error calculating microstructure score: {e}")
            return 0.0

    def calculate_optimal_leverage(self, confidence: float, volatility: str, 
                                 leverage_range: Tuple[int, int], timer_status: TimerStatus) -> float:
        """Calculate optimal leverage based on multiple factors"""
        try:
            min_lev, max_lev = leverage_range
            
            # Base leverage from confidence
            base_leverage = min_lev + (max_lev - min_lev) * confidence
            
            # Volatility adjustment
            vol_adjustments = {
                'Medium': 1.0,
                'High': 0.85,
                'Very High': 0.7,
                'Extreme': 0.5
            }
            vol_adj = vol_adjustments.get(volatility, 0.8)
            
            # Timer efficiency adjustment
            timer_adj = timer_status.efficiency
            
            # Calculate final leverage
            final_leverage = base_leverage * vol_adj * timer_adj
            
            # Ensure within SuperP limits
            return min(max(final_leverage, 10), self.max_leverage)
            
        except Exception as e:
            logger.error(f"Error calculating leverage: {e}")
            return 100.0  # Safe default

    def calculate_position_size(self, confidence: float, leverage: float, volatility: str) -> float:
        """Calculate position size based on confidence and risk"""
        try:
            # Base position from config
            base_size = self.config.get('superp', {}).get('min_buy_in', 10.0)
            max_size = self.config.get('superp', {}).get('max_buy_in', 100.0)
            
            # Confidence-based sizing
            size_factor = confidence * 0.8 + 0.2  # 20%-100% of range
            base_position = base_size + (max_size - base_size) * size_factor
            
            # Volatility adjustment
            vol_adjustments = {
                'Medium': 1.0,
                'High': 0.9,
                'Very High': 0.75,
                'Extreme': 0.6
            }
            vol_adj = vol_adjustments.get(volatility, 0.8)
            
            return round(base_position * vol_adj, 2)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 10.0  # Safe default

    def get_ml_confidence_estimate(self, features, direction: str) -> Optional[float]:
        """Get ML confidence estimate (simplified integration point)"""
        try:
            # Simplified ML confidence based on microstructure features
            # This would integrate with your actual ML engine
            
            spread_factor = max(0, 1.0 - features['spread_bps'] / 10.0)
            imbalance_factor = abs(features['imbalance'])
            
            # Direction alignment
            if direction == "LONG" and features['imbalance'] > 0:
                direction_bonus = 0.2
            elif direction == "SHORT" and features['imbalance'] < 0:
                direction_bonus = 0.2
            else:
                direction_bonus = 0.0
            
            ml_confidence = (spread_factor + imbalance_factor + direction_bonus) / 3.0
            return min(0.8, ml_confidence)  # Cap at 80% for conservative estimates
            
        except Exception as e:
            logger.error(f"Error getting ML confidence: {e}")
            return None

    async def collect_training_data(self, duration_minutes=30):
        """Collect live training data with PSC integration"""
        logger.info(f"Starting {duration_minutes}-minute PSC-aligned data collection...")
        
        # Get initial TON price
        await self.get_ton_price()
        logger.info(f"TON price: ${self.ton_price:.6f}")
        
        end_time = time.time() + (duration_minutes * 60)
        collection_count = 0
        
        while time.time() < end_time:
            try:
                # Update TON price periodically
                if collection_count % 20 == 0:
                    await self.get_ton_price()
                
                # Collect data from all coins
                for coin in self.trading_coins:
                    order_book = await self.get_live_order_book(coin['pair'])
                    
                    if order_book:
                        features = self.extract_microstructure_features(order_book)
                        
                        if features:
                            # Store for training
                            self.order_book_history.append(order_book)
                            self.feature_history.append(features)
                            
                            # Generate PSC-aligned signal
                            psc_signal = self.generate_psc_aligned_signal(features, coin)
                            
                            if psc_signal:
                                self.psc_signals.append(psc_signal)
                                self.signals_generated += 1
                                
                                # Save signal immediately
                                await self.save_psc_signal(psc_signal)
                                
                                # Log signal details
                                logger.info(f"PSC Signal: {psc_signal.symbol} {psc_signal.direction}")
                                logger.info(f"  Ratio: {psc_signal.psc_ratio:.3f}, Confidence: {psc_signal.confidence_score:.1%}")
                                logger.info(f"  Leverage: {psc_signal.leverage:.0f}x, Window: {psc_signal.timer_window}")
                            
                            collection_count += 1
                
                # Progress update
                if collection_count % 50 == 0:
                    timer_status = self.get_timer_status()
                    logger.info(f"Collected {collection_count} samples, {self.signals_generated} PSC signals")
                    logger.info(f"Timer: {timer_status.window_type} ({timer_status.seconds_remaining}s remaining)")
                
                # Wait before next collection
                await asyncio.sleep(5)  # 5-second intervals
                
            except Exception as e:
                logger.error(f"Error in data collection: {e}")
                await asyncio.sleep(1)
        
        self.training_samples = len(self.feature_history)
        logger.info(f"Collection complete: {self.training_samples} samples, {self.signals_generated} PSC signals")

    async def save_psc_signal(self, psc_signal: PSCSignal):
        """Save PSC-aligned signal to database (Railway ready) or file fallback"""
        try:
            # Try database storage first (Railway deployment)
            if self.data_manager:
                signal_id = self.data_manager.log_psc_signal(
                    coin=psc_signal.symbol,
                    price=psc_signal.current_price if hasattr(psc_signal, 'current_price') else 0.0,
                    ratio=psc_signal.psc_ratio,
                    confidence=psc_signal.confidence_score,
                    direction=psc_signal.direction,
                    exit_estimate=0.0,  # Will be calculated in data_manager
                    ml_prediction=psc_signal.microstructure_score,
                    market_conditions=f"Timer: {psc_signal.timer_window}, Leverage: {psc_signal.leverage}x",
                    ml_features={
                        'signal_type': 'MICROSTRUCTURE',
                        'psc_ratio': psc_signal.psc_ratio,
                        'confidence_score': psc_signal.confidence_score,
                        'leverage': psc_signal.leverage,
                        'position_size': psc_signal.position_size,
                        'timer_window': psc_signal.timer_window,
                        'signal_strength': psc_signal.signal_strength,
                        'reasons': psc_signal.reasons,
                        'microstructure_score': psc_signal.microstructure_score,
                        'ml_validation': psc_signal.ml_validation,
                        'superp_enabled': self.superp_enabled,
                        'system_version': 'PSC_ALIGNED_v1.0'
                    }
                )
                logger.info(f"📊 Microstructure signal logged to database: {psc_signal.symbol} {psc_signal.direction} (ID: {signal_id})")
                return
            
            # Fallback to JSON file storage for local development
            timestamp_info = self.get_current_timestamp()
            signal_dict = {
                **timestamp_info,
                'symbol': psc_signal.symbol,
                'direction': psc_signal.direction,
                'psc_ratio': psc_signal.psc_ratio,
                'confidence_score': psc_signal.confidence_score,
                'leverage': psc_signal.leverage,
                'position_size': psc_signal.position_size,
                'timer_window': psc_signal.timer_window,
                'signal_strength': psc_signal.signal_strength,
                'reasons': psc_signal.reasons,
                'microstructure_score': psc_signal.microstructure_score,
                'ml_validation': psc_signal.ml_validation,
                'superp_enabled': self.superp_enabled,
                'system_version': 'PSC_ALIGNED_v1.0'
            }
            
            # Load existing signals - use project root
            project_root = Path(__file__).parent.parent.parent
            signals_file = project_root / 'data' / 'live_microstructure_signals.json'
            signals_file.parent.mkdir(exist_ok=True)
            
            if signals_file.exists():
                with open(signals_file, 'r') as f:
                    existing_signals = json.load(f)
            else:
                existing_signals = []
            
            # Add new signal
            existing_signals.append(signal_dict)
            
            # Save updated signals
            with open(signals_file, 'w') as f:
                json.dump(existing_signals, f, indent=2, default=str)
            
            logger.info(f"PSC signal saved to JSON: {psc_signal.symbol} {psc_signal.direction}")
            
        except Exception as e:
            logger.error(f"Error saving PSC signal: {e}")

    async def save_signal(self, signal):
        """Legacy method - kept for compatibility"""
        pass  # Replaced by save_psc_signal

    def calculate_training_metrics(self):
        """Calculate enhanced training performance metrics"""
        if len(self.feature_history) == 0:
            return {'error': 'No training data collected'}
        
        # Basic statistics
        features_df = pd.DataFrame(self.feature_history)
        
        # PSC-specific metrics
        psc_signals_df = pd.DataFrame([
            {
                'symbol': s.symbol,
                'direction': s.direction,
                'psc_ratio': s.psc_ratio,
                'confidence': s.confidence_score,
                'leverage': s.leverage,
                'strength': s.signal_strength,
                'timer_window': s.timer_window
            } for s in self.psc_signals
        ]) if self.psc_signals else pd.DataFrame()
        
        metrics = {
            'training_samples': self.training_samples,
            'signals_generated': self.signals_generated,
            'signal_rate': self.signals_generated / self.training_samples if self.training_samples > 0 else 0,
            'coins_analyzed': len(set(features_df['symbol'])) if not features_df.empty else 0,
            'avg_spread_bps': features_df['spread_bps'].mean() if not features_df.empty else 0,
            'avg_imbalance': features_df['imbalance'].mean() if not features_df.empty else 0,
            'ton_price': self.ton_price,
            'psc_integration': True,
            'superp_enabled': self.superp_enabled,
            'max_leverage': self.max_leverage
        }
        
        # PSC signal breakdown
        if not psc_signals_df.empty:
            metrics.update({
                'long_signals': len(psc_signals_df[psc_signals_df['direction'] == 'LONG']),
                'short_signals': len(psc_signals_df[psc_signals_df['direction'] == 'SHORT']),
                'avg_psc_ratio': psc_signals_df['psc_ratio'].mean(),
                'avg_confidence': psc_signals_df['confidence'].mean(),
                'avg_leverage': psc_signals_df['leverage'].mean(),
                'strong_signals': len(psc_signals_df[psc_signals_df['strength'] == 'STRONG']),
                'entry_window_signals': len(psc_signals_df[psc_signals_df['timer_window'] == 'ENTRY_WINDOW'])
            })
        
        return metrics

    async def run_live_training(self, duration_minutes=30):
        """Run the complete PSC-aligned live training process"""
        logger.info("Starting PSC-Aligned Live Microstructure Training Session")
        logger.info("=" * 60)
        
        try:
            # Collect training data
            await self.collect_training_data(duration_minutes)
            
            # Calculate metrics
            metrics = self.calculate_training_metrics()
            
            # Save training summary
            summary = {
                'timestamp': datetime.now().isoformat(),
                'duration_minutes': duration_minutes,
                'system_version': 'PSC_ALIGNED_v1.0',
                'integration_features': [
                    'PSC ratio calculations',
                    'Timer-based trading windows',
                    'Bidirectional signal generation',
                    'ML confidence validation',
                    'SuperP leverage optimization'
                ],
                'metrics': metrics,
                'trading_coins': [
                    {
                        'symbol': coin['symbol'],
                        'volatility': coin['volatility'],
                        'psc_weight': coin['psc_weight'],
                        'ml_priority': coin['ml_priority']
                    } for coin in self.trading_coins
                ],
                'psc_config': {
                    'min_signal_ratio': self.min_signal_ratio,
                    'min_confidence_threshold': self.min_confidence_threshold,
                    'ton_price': self.ton_price
                },
                'superp_config': {
                    'enabled': self.superp_enabled,
                    'max_leverage': self.max_leverage,
                    'time_limit_minutes': self.time_limit_minutes
                }
            }
            
            # Save summary
            project_root = Path(__file__).parent.parent.parent
            summary_file = project_root / 'data' / 'live_training_summary.json'
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            # Print results
            logger.info("PSC Training Session Complete!")
            logger.info("=" * 60)
            logger.info(f"Duration: {duration_minutes} minutes")
            logger.info(f"Samples collected: {metrics['training_samples']}")
            logger.info(f"PSC signals generated: {metrics['signals_generated']}")
            logger.info(f"Signal rate: {metrics['signal_rate']:.1%}")
            logger.info(f"TON price: ${metrics['ton_price']:.6f}")
            if 'long_signals' in metrics:
                logger.info(f"LONG signals: {metrics['long_signals']}, SHORT signals: {metrics['short_signals']}")
                logger.info(f"Avg PSC ratio: {metrics['avg_psc_ratio']:.3f}")
                logger.info(f"Avg confidence: {metrics['avg_confidence']:.1%}")
            if self.superp_enabled:
                logger.info(f"SuperP leverage: up to {self.max_leverage}x")
                logger.info(f"Timer positions: {self.time_limit_minutes} min limit")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in live training: {e}")
            return {'error': str(e)}
    
    def get_recent_psc_signals(self, max_age_minutes=5, min_confidence=0.6):
        """
        Get recent PSC-aligned signals for integration with main trading system
        
        Args:
            max_age_minutes: Maximum age of signals to return
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of recent PSCSignal objects
        """
        try:
            # Load recent signals from file
            project_root = Path(__file__).parent.parent.parent
            signals_file = project_root / 'data' / 'live_microstructure_signals.json'
            
            if not signals_file.exists():
                return []
            
            with open(signals_file, 'r') as f:
                all_signals = json.load(f)
            
            if not all_signals:
                return []
            
            # Get current time
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(minutes=max_age_minutes)
            
            # Filter recent signals
            recent_signals = []
            for signal_data in all_signals[-50:]:  # Check last 50 signals
                try:
                    # Parse timestamp - handle both old and new formats
                    timestamp_value = signal_data.get('timestamp', 0)
                    
                    if isinstance(timestamp_value, str):
                        # New ISO format
                        signal_time = datetime.fromisoformat(timestamp_value.replace('Z', '+00:00'))
                    elif isinstance(timestamp_value, (int, float)):
                        # Old timestamp format - check if it's milliseconds or seconds
                        if timestamp_value > 1e12:  # Likely milliseconds (or weird large number)
                            # If it's too large, it might be the weird timestamp - try dividing
                            if timestamp_value > 1e15:
                                timestamp_value = timestamp_value / 1000  # Convert weird format
                            signal_time = datetime.fromtimestamp(timestamp_value / 1000)
                        else:
                            # Regular Unix timestamp
                            signal_time = datetime.fromtimestamp(timestamp_value)
                    else:
                        # Use unix timestamp if available
                        unix_ts = signal_data.get('timestamp_unix', 0)
                        signal_time = datetime.fromtimestamp(unix_ts) if unix_ts else datetime.now()
                    
                    # Check if signal is recent enough
                    if signal_time < cutoff_time:
                        continue
                    
                    # Check if signal has PSC data
                    if 'psc_ratio' not in signal_data:
                        continue
                    
                    # Check confidence threshold
                    confidence = signal_data.get('confidence', 0.0)
                    if confidence < min_confidence:
                        continue
                    
                    # Convert to PSCSignal object
                    psc_signal = PSCSignal(
                        symbol=signal_data.get('symbol', ''),
                        direction=signal_data.get('action', ''),
                        psc_ratio=signal_data.get('psc_ratio', 0.0),
                        confidence_score=confidence,
                        leverage=signal_data.get('leverage', 1.0),
                        position_size=signal_data.get('position_size', 100.0),
                        timer_window=signal_data.get('timer_window', 'UNKNOWN'),
                        signal_strength=signal_data.get('signal_strength', 'medium'),
                        reasons=signal_data.get('reasons', []),
                        microstructure_score=signal_data.get('microstructure_score', 0.0),
                        ml_validation=confidence
                    )
                    
                    recent_signals.append(psc_signal)
                    
                except Exception as signal_error:
                    logger.warning(f"Error parsing signal: {signal_error}")
                    continue
            
            logger.info(f"🔍 Found {len(recent_signals)} recent PSC signals (last {max_age_minutes} min)")
            return recent_signals
            
        except Exception as e:
            logger.error(f"Error getting recent PSC signals: {e}")
            return []

async def main():
    """Main execution function with PSC integration"""
    trainer = LiveMicrostructureTrainer()
    
    # Run training session
    print("🚀 Starting PSC-Aligned Live Microstructure Training")
    print("=" * 60)
    print(f"💎 Trading Coins: {', '.join([coin['symbol'] for coin in trainer.trading_coins])}")
    print(f"📊 PSC Integration: min_ratio={trainer.min_signal_ratio}, min_confidence={trainer.min_confidence_threshold}")
    print(f"⚡ SuperP Enabled: {trainer.superp_enabled}")
    print(f"🎯 Max Leverage: {trainer.max_leverage}x")
    print(f"⏰ Timer-based trading: {trainer.time_limit_minutes} min cycles")
    print("=" * 60)
    
    # Start training
    result = await trainer.run_live_training(duration_minutes=15)  # 15-minute session
    
    print("\n🎉 PSC-aligned training session completed!")
    print(f"📄 Results saved to: data/live_training_summary.json")
    print(f"💾 PSC signals saved to: data/live_microstructure_signals.json")
    print("\n✅ Enhanced model ready for integration with PSC TON system!")

if __name__ == "__main__":
    asyncio.run(main())
