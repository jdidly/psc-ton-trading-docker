#!/usr/bin/env python3
"""
Advanced Signal Quality Filter for PSC Trading System
Implements intelligent filtering to reduce false signals and improve profitability
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sqlite3
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class SignalQualityMetrics:
    """Metrics for evaluating signal quality"""
    volume_factor: float = 0.0
    volatility_score: float = 0.0
    trend_alignment: float = 0.0
    technical_confluence: float = 0.0
    market_structure: float = 0.0
    liquidity_score: float = 0.0
    overall_quality: float = 0.0

class AdvancedSignalFilter:
    """
    Advanced signal filtering system to improve trade quality
    """
    
    def __init__(self, db_path: str = "psc_trading.db"):
        self.db_path = db_path
        
        # Quality thresholds - ADJUSTED for better balance
        self.min_volume_multiplier = 1.2  # Volume must be 1.2x average (reduced)
        self.min_market_cap = 50_000_000  # $50M minimum market cap (reduced for more coins)
        self.volatility_range = (1.0, 20.0)  # 1-20% daily volatility preferred (wider range)
        self.min_liquidity_score = 0.15  # Reduced for better signal acceptance
        self.min_overall_quality = 0.5  # Reduced for better signal acceptance
        
        # Historical performance tracking
        self.performance_weights = {
            'recent_accuracy': 0.3,
            'profit_ratio': 0.25,
            'signal_frequency': 0.15,
            'risk_adjusted_return': 0.3
        }
        
    def calculate_volume_factor(self, current_volume: float, avg_volume: float) -> float:
        """
        Calculate volume factor (higher is better)
        """
        if avg_volume <= 0:
            return 0.0
            
        volume_ratio = current_volume / avg_volume
        
        # Score based on volume increase
        if volume_ratio >= 3.0:
            return 1.0
        elif volume_ratio >= 2.0:
            return 0.8
        elif volume_ratio >= 1.5:
            return 0.6
        elif volume_ratio >= 1.0:
            return 0.4
        else:
            return 0.2
    
    def analyze_volatility_score(self, change_24h: float, historical_volatility: Optional[float] = None) -> float:
        """
        Analyze volatility for optimal trading conditions
        """
        abs_change = abs(change_24h)
        
        # Ideal volatility range for crypto trading
        if self.volatility_range[0] <= abs_change <= self.volatility_range[1]:
            return 1.0
        elif abs_change < self.volatility_range[0]:
            # Too low volatility
            return abs_change / self.volatility_range[0]
        else:
            # Too high volatility (risky)
            excess = abs_change - self.volatility_range[1]
            return max(0.1, 1.0 - (excess / 20.0))
    
    def evaluate_trend_alignment(self, price: float, sma_20: Optional[float], sma_50: Optional[float], 
                                signal_direction: str) -> float:
        """
        Evaluate trend alignment for signal confirmation
        """
        if not sma_20 or not sma_50:
            return 0.5  # Neutral if no trend data
        
        # Determine trend direction
        short_trend = "bullish" if price > sma_20 else "bearish"
        long_trend = "bullish" if sma_20 > sma_50 else "bearish"
        
        # Calculate alignment score
        alignment_score = 0.0
        
        if signal_direction.upper() == "BUY":
            if short_trend == "bullish" and long_trend == "bullish":
                alignment_score = 1.0  # Perfect bullish alignment
            elif short_trend == "bullish":
                alignment_score = 0.7  # Short-term bullish
            elif long_trend == "bullish":
                alignment_score = 0.5  # Long-term bullish only
            else:
                alignment_score = 0.2  # Counter-trend
        
        elif signal_direction.upper() == "SELL":
            if short_trend == "bearish" and long_trend == "bearish":
                alignment_score = 1.0  # Perfect bearish alignment
            elif short_trend == "bearish":
                alignment_score = 0.7  # Short-term bearish
            elif long_trend == "bearish":
                alignment_score = 0.5  # Long-term bearish only
            else:
                alignment_score = 0.2  # Counter-trend
        
        else:
            alignment_score = 0.3  # Neutral signals get lower score
        
        return alignment_score
    
    def calculate_technical_confluence(self, market_data: Dict, signal_data: Dict) -> float:
        """
        Calculate technical indicator confluence
        """
        confluence_score = 0.0
        indicator_count = 0
        
        # RSI confluence
        rsi = market_data.get('rsi')
        if rsi:
            indicator_count += 1
            signal_direction = signal_data.get('direction', '').upper()
            
            if signal_direction == "BUY" and rsi < 40:  # Oversold buy signal
                confluence_score += 0.8
            elif signal_direction == "SELL" and rsi > 60:  # Overbought sell signal
                confluence_score += 0.8
            elif 40 <= rsi <= 60:  # Neutral RSI
                confluence_score += 0.4
            else:
                confluence_score += 0.2  # Counter-signal
        
        # Bollinger Bands confluence
        price = market_data.get('price', 0)
        bb_upper = market_data.get('bb_upper')
        bb_lower = market_data.get('bb_lower')
        
        if bb_upper and bb_lower and price > 0:
            indicator_count += 1
            signal_direction = signal_data.get('direction', '').upper()
            
            if signal_direction == "BUY" and price <= bb_lower:  # Oversold bounce
                confluence_score += 0.9
            elif signal_direction == "SELL" and price >= bb_upper:  # Overbought rejection
                confluence_score += 0.9
            elif bb_lower < price < bb_upper:  # Normal range
                confluence_score += 0.5
            else:
                confluence_score += 0.3
        
        # Average the confluence scores
        return confluence_score / max(indicator_count, 1)
    
    def assess_market_structure(self, symbol: str) -> float:
        """
        Assess overall market structure and conditions
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get recent signal performance for this symbol
            cursor.execute("""
                SELECT COUNT(*) as total_signals,
                       AVG(CASE WHEN actual_direction = predicted_direction THEN 1.0 ELSE 0.0 END) as accuracy
                FROM ml_validations 
                WHERE symbol = ? AND timestamp > datetime('now', '-7 days')
            """, (symbol,))
            
            result = cursor.fetchone()
            total_signals, accuracy = result if result else (0, 0)
            
            conn.close()
            
            # Structure score based on recent performance
            if total_signals >= 10:
                if accuracy >= 0.7:
                    return 0.9
                elif accuracy >= 0.6:
                    return 0.7
                elif accuracy >= 0.5:
                    return 0.5
                else:
                    return 0.3
            elif total_signals >= 5:
                return 0.6  # Limited data
            else:
                return 0.4  # New or infrequent symbol
                
        except Exception as e:
            logger.error(f"Market structure assessment error: {e}")
            return 0.5
    
    def calculate_liquidity_score(self, volume_24h: float, market_cap: float) -> float:
        """
        Calculate liquidity score based on volume and market cap
        Enhanced to be more realistic for crypto markets
        """
        if market_cap <= 0 or volume_24h <= 0:
            return 0.0
        
        # Volume to market cap ratio (higher is more liquid)
        volume_ratio = volume_24h / market_cap
        
        # Enhanced liquidity scoring for crypto markets
        if volume_ratio >= 0.05:  # 5%+ turnover (very liquid for crypto)
            return 1.0
        elif volume_ratio >= 0.02:  # 2-5% turnover (good liquidity)
            return 0.8
        elif volume_ratio >= 0.01:  # 1-2% turnover (moderate liquidity)
            return 0.6
        elif volume_ratio >= 0.005:  # 0.5-1% turnover (acceptable liquidity)
            return 0.4
        elif volume_ratio >= 0.001:  # 0.1-0.5% turnover (low but tradeable)
            return 0.2
        else:  # <0.1% turnover (very poor liquidity)
            return 0.1
    
    def evaluate_signal_quality(self, symbol: str, signal_data: Dict, market_data: Dict) -> SignalQualityMetrics:
        """
        Comprehensive signal quality evaluation
        """
        try:
            # Extract data safely
            current_volume = market_data.get('volume_24h', 0)
            avg_volume = market_data.get('volume_sma', current_volume)
            change_24h = market_data.get('change_24h', 0)
            price = market_data.get('price', 0)
            market_cap = market_data.get('market_cap', 0)
            
            # Calculate individual quality metrics
            volume_factor = self.calculate_volume_factor(current_volume, avg_volume)
            volatility_score = self.analyze_volatility_score(change_24h)
            trend_alignment = self.evaluate_trend_alignment(
                price, 
                market_data.get('sma_20'), 
                market_data.get('sma_50'),
                signal_data.get('direction', 'NEUTRAL')
            )
            technical_confluence = self.calculate_technical_confluence(market_data, signal_data)
            market_structure = self.assess_market_structure(symbol)
            liquidity_score = self.calculate_liquidity_score(current_volume, market_cap)
            
            # Calculate overall quality (weighted average)
            overall_quality = (
                volume_factor * 0.20 +
                volatility_score * 0.15 +
                trend_alignment * 0.25 +
                technical_confluence * 0.20 +
                market_structure * 0.10 +
                liquidity_score * 0.10
            )
            
            return SignalQualityMetrics(
                volume_factor=volume_factor,
                volatility_score=volatility_score,
                trend_alignment=trend_alignment,
                technical_confluence=technical_confluence,
                market_structure=market_structure,
                liquidity_score=liquidity_score,
                overall_quality=overall_quality
            )
            
        except Exception as e:
            logger.error(f"Signal quality evaluation error for {symbol}: {e}")
            return SignalQualityMetrics()  # Return default (all zeros)
    
    def should_accept_signal(self, symbol: str, signal_data: Dict, market_data: Dict) -> Tuple[bool, SignalQualityMetrics, str]:
        """
        Determine if signal should be accepted based on quality filters
        Returns: (accept_signal, quality_metrics, rejection_reason)
        """
        # Evaluate signal quality
        quality = self.evaluate_signal_quality(symbol, signal_data, market_data)
        
        # Apply hard filters first
        market_cap = market_data.get('market_cap', 0)
        if market_cap < self.min_market_cap:
            return False, quality, f"Market cap too low: ${market_cap:,.0f}"
        
        volume_24h = market_data.get('volume_24h', 0)
        if volume_24h <= 0:
            return False, quality, "No volume data available"
        
        # Apply quality thresholds
        if quality.overall_quality < self.min_overall_quality:
            return False, quality, f"Quality score too low: {quality.overall_quality:.2f}"
        
        if quality.liquidity_score < self.min_liquidity_score:
            return False, quality, f"Liquidity too low: {quality.liquidity_score:.2f}"
        
        # Signal passed all filters
        return True, quality, "Signal accepted"
    
    def get_position_size_multiplier(self, quality_metrics: SignalQualityMetrics) -> float:
        """
        Calculate position size multiplier based on signal quality
        Higher quality signals get larger position sizes
        """
        base_multiplier = 1.0
        
        # Quality-based multiplier (0.5x to 2.0x)
        quality_multiplier = 0.5 + (quality_metrics.overall_quality * 1.5)
        
        # Confidence boosts
        if quality_metrics.trend_alignment >= 0.8 and quality_metrics.technical_confluence >= 0.7:
            quality_multiplier *= 1.2  # Strong confluence bonus
        
        if quality_metrics.volume_factor >= 0.8:
            quality_multiplier *= 1.1  # High volume bonus
        
        # Cap the multiplier
        return min(quality_multiplier, 2.0)


# Test function
def test_signal_filter():
    """Test the advanced signal filter"""
    
    print("Testing Advanced Signal Filter...")
    
    filter_system = AdvancedSignalFilter()
    
    # Mock data for testing
    test_signal = {
        'direction': 'BUY',
        'confidence': 0.75,
        'symbol': 'BTC'
    }
    
    test_market_data = {
        'price': 115000,
        'volume_24h': 25_000_000_000,
        'volume_sma': 20_000_000_000,
        'change_24h': 3.5,
        'market_cap': 2_300_000_000_000,
        'rsi': 35,
        'sma_20': 114000,
        'sma_50': 112000,
        'bb_upper': 118000,
        'bb_lower': 110000
    }
    
    # Test signal evaluation
    quality = filter_system.evaluate_signal_quality('BTC', test_signal, test_market_data)
    accept, quality_metrics, reason = filter_system.should_accept_signal('BTC', test_signal, test_market_data)
    position_multiplier = filter_system.get_position_size_multiplier(quality)
    
    print(f"Signal Quality Analysis:")
    print(f"  Volume Factor: {quality.volume_factor:.2f}")
    print(f"  Volatility Score: {quality.volatility_score:.2f}")
    print(f"  Trend Alignment: {quality.trend_alignment:.2f}")
    print(f"  Technical Confluence: {quality.technical_confluence:.2f}")
    print(f"  Market Structure: {quality.market_structure:.2f}")
    print(f"  Liquidity Score: {quality.liquidity_score:.2f}")
    print(f"  Overall Quality: {quality.overall_quality:.2f}")
    print(f"\nSignal Decision: {'ACCEPT' if accept else 'REJECT'}")
    print(f"Reason: {reason}")
    print(f"Position Size Multiplier: {position_multiplier:.2f}x")

if __name__ == "__main__":
    test_signal_filter()