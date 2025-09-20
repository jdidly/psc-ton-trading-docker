#!/usr/bin/env python3
"""
TradingView Integration for PSC TON System
Fetches technical analysis data from TradingView for bias confirmation
"""

import asyncio
import aiohttp
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List
from bs4 import BeautifulSoup
import json
import time

# Import real market data provider
from real_market_data import RealMarketDataProvider

logger = logging.getLogger(__name__)

class TechnicalAnalysis:
    """TradingView Technical Analysis Data Structure"""
    
    def __init__(self):
        self.symbol = ""
        self.timeframe = ""
        self.summary = "neutral"
        self.ma_score = 0
        self.osc_score = 0
        self.overall_score = 0
        self.ma_recommendation = "neutral"
        self.osc_recommendation = "neutral"
        self.pivot_points = {}
        self.timestamp = datetime.now()
        
        # Individual indicator scores
        self.indicators = {
            'moving_averages': {},
            'oscillators': {}
        }

class TradingViewIntegration:
    """TradingView API Integration for PSC System"""
    
    def __init__(self, use_real_data=True):
        self.base_url = "https://www.tradingview.com"
        self.session = None
        self.cache = {}
        self.cache_duration = 60  # 1 minute cache
        self.use_real_data = use_real_data
        
        # Initialize real market data provider
        if self.use_real_data:
            self.market_data_provider = RealMarketDataProvider()
            logger.info("✅ Real market data provider initialized")
        else:
            self.market_data_provider = None
            logger.info("⚠️ Using simulated market data")
        self.request_delay = 1.5  # 1.5 seconds between requests (faster for multiple calls)
        self.last_request = 0
        
        # Multiple timeframe support
        self.timeframes = ['1m', '5m', '10m']
        self.monitored_coins = ['BTC', 'ETH', 'SOL', 'SHIB', 'DOGE', 'PEPE']
        
        # Comprehensive analysis storage
        self.multi_timeframe_data = {}
        self.last_full_scan = 0
        self.full_scan_interval = 30  # Full scan every 30 seconds
        
        # TradingView symbol mapping
        self.symbol_map = {
            'BTC': 'BTCUSD',
            'ETH': 'ETHUSD', 
            'SOL': 'SOLUSD',
            'SHIB': 'SHIBUSD',
            'DOGE': 'DOGEUSD',
            'PEPE': 'PEPEUSD',
            'TON': 'TONUSD'
        }
        
    async def initialize(self):
        """Initialize HTTP session"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            )
            logger.info("✅ TradingView session initialized")
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
            
        # Close real market data provider
        if self.market_data_provider:
            await self.market_data_provider.close()
    
    def _respect_rate_limit(self):
        """Ensure we don't exceed rate limits"""
        now = time.time()
        time_since_last = now - self.last_request
        if time_since_last < self.request_delay:
            time.sleep(self.request_delay - time_since_last)
        self.last_request = time.time()
    
    def _get_cache_key(self, symbol: str, timeframe: str) -> str:
        """Generate cache key"""
        return f"{symbol}_{timeframe}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache:
            return False
        
        cached_time = self.cache[cache_key].get('timestamp', 0)
        return (time.time() - cached_time) < self.cache_duration
    
    async def get_technical_analysis(self, symbol: str, timeframe: str = '1m') -> Optional[TechnicalAnalysis]:
        """Get technical analysis for symbol"""
        
        # Check cache first
        cache_key = self._get_cache_key(symbol, timeframe)
        if self._is_cache_valid(cache_key):
            logger.info(f"📋 Using cached TradingView data for {symbol} {timeframe}")
            return self.cache[cache_key]['data']
        
        try:
            await self.initialize()
            self._respect_rate_limit()
            
            # Map symbol to TradingView format
            tv_symbol = self.symbol_map.get(symbol, f"{symbol}USD")
            
            # For now, simulate TradingView data since scraping is complex
            # In production, this would connect to TradingView's API or scrape their data
            analysis = await self._fetch_analysis_data(tv_symbol, timeframe)
            
            if analysis:
                # Cache the result
                self.cache[cache_key] = {
                    'data': analysis,
                    'timestamp': time.time()
                }
                logger.info(f"✅ TradingView analysis fetched for {symbol} {timeframe}: {analysis.summary}")
                return analysis
            else:
                logger.warning(f"⚠️ No TradingView data for {symbol} {timeframe}")
                return None
                
        except Exception as e:
            logger.error(f"❌ TradingView fetch error for {symbol}: {e}")
            return None
    
    async def _fetch_analysis_data(self, tv_symbol: str, timeframe: str) -> Optional[TechnicalAnalysis]:
        """Fetch actual analysis data (simulated for now)"""
        
        # For MVP, we'll simulate realistic TradingView technical analysis
        # In production, this would scrape or use TradingView's actual API
        
        import random
        
        analysis = TechnicalAnalysis()
        analysis.symbol = tv_symbol
        analysis.timeframe = timeframe
        
        # Generate realistic technical analysis scores
        # Moving Averages (15 indicators): -2 to +2 each
        ma_indicators = [
            'EMA10', 'EMA20', 'EMA30', 'EMA50', 'EMA100', 'EMA200',
            'SMA10', 'SMA20', 'SMA30', 'SMA50', 'SMA100', 'SMA200',
            'Ichimoku', 'VWMA', 'HullMA'
        ]
        
        ma_total = 0
        for indicator in ma_indicators:
            score = random.choice([-2, -1, 0, 1, 2])
            analysis.indicators['moving_averages'][indicator] = score
            ma_total += score
        
        # Oscillators (11 indicators): -2 to +2 each  
        osc_indicators = [
            'RSI', 'Stochastic', 'CCI', 'ADX', 'AO', 'Momentum',
            'MACD', 'StochRSI', 'WilliamsR', 'BullBearPower', 'UO'
        ]
        
        osc_total = 0
        for indicator in osc_indicators:
            score = random.choice([-2, -1, 0, 1, 2])
            analysis.indicators['oscillators'][indicator] = score
            osc_total += score
        
        analysis.ma_score = ma_total
        analysis.osc_score = osc_total
        analysis.overall_score = ma_total + osc_total
        
        # Determine recommendations
        if ma_total >= 8:
            analysis.ma_recommendation = "strong_buy"
        elif ma_total >= 4:
            analysis.ma_recommendation = "buy"
        elif ma_total <= -8:
            analysis.ma_recommendation = "strong_sell"
        elif ma_total <= -4:
            analysis.ma_recommendation = "sell"
        else:
            analysis.ma_recommendation = "neutral"
            
        if osc_total >= 6:
            analysis.osc_recommendation = "strong_buy"
        elif osc_total >= 3:
            analysis.osc_recommendation = "buy" 
        elif osc_total <= -6:
            analysis.osc_recommendation = "strong_sell"
        elif osc_total <= -3:
            analysis.osc_recommendation = "sell"
        else:
            analysis.osc_recommendation = "neutral"
        
        # Overall summary
        total_score = analysis.overall_score
        if total_score >= 14:
            analysis.summary = "strong_buy"
        elif total_score >= 7:
            analysis.summary = "buy"
        elif total_score <= -14:
            analysis.summary = "strong_sell"
        elif total_score <= -7:
            analysis.summary = "sell"
        else:
            analysis.summary = "neutral"
        
        # Add realistic pivot points
        base_price = random.uniform(40000, 50000)  # Simulated base price
        analysis.pivot_points = {
            'P': base_price,
            'R1': base_price * 1.02,
            'R2': base_price * 1.04,
            'R3': base_price * 1.06,
            'S1': base_price * 0.98,
            'S2': base_price * 0.96,
            'S3': base_price * 0.94
        }
        
        analysis.timestamp = datetime.now()
        
        logger.info(f"📊 Generated TradingView analysis for {tv_symbol}: MA={ma_total}, OSC={osc_total}, Summary={analysis.summary}")
        
        return analysis
    
    def get_bias_confirmation(self, analysis: TechnicalAnalysis) -> Dict:
        """Get bias confirmation data from TradingView analysis"""
        
        if not analysis:
            return {
                'direction': 'unknown',
                'strength': 0.0,
                'confidence': 0.0,
                'ma_signals': 'neutral',
                'osc_signals': 'neutral'
            }
        
        # Calculate bias strength (0.0 to 1.0)
        max_possible_score = 26 * 2  # 26 indicators * 2 points each = 52
        strength = abs(analysis.overall_score) / max_possible_score
        
        # Determine direction
        if analysis.overall_score > 7:
            direction = 'bullish'
        elif analysis.overall_score < -7:
            direction = 'bearish'
        else:
            direction = 'neutral'
        
        # Calculate confidence based on alignment
        ma_bullish = analysis.ma_score > 0
        osc_bullish = analysis.osc_score > 0
        
        if ma_bullish == osc_bullish:
            confidence = min(1.0, strength * 1.5)  # Boost for alignment
        else:
            confidence = strength * 0.7  # Reduce for conflict
        
        return {
            'direction': direction,
            'strength': strength,
            'confidence': confidence,
            'ma_signals': analysis.ma_recommendation,
            'osc_signals': analysis.osc_recommendation,
            'alignment': ma_bullish == osc_bullish
        }
    
    def enhance_psc_prediction(self, psc_confidence: float, psc_direction: str, 
                             tv_analysis: TechnicalAnalysis) -> Dict:
        """Enhance PSC prediction with TradingView bias"""
        
        if not tv_analysis:
            return {
                'enhanced_confidence': psc_confidence,
                'alignment_score': 0.0,
                'recommendation': 'PSC ONLY - No TradingView data'
            }
        
        bias_data = self.get_bias_confirmation(tv_analysis)
        
        # Calculate alignment score
        psc_bullish = psc_direction.upper() in ['LONG', 'BUY']
        tv_bullish = bias_data['direction'] == 'bullish'
        tv_neutral = bias_data['direction'] == 'neutral'
        
        if tv_neutral:
            alignment_score = 0.5  # Neutral TV doesn't help or hurt
        elif psc_bullish == tv_bullish:
            alignment_score = 0.8 + (bias_data['strength'] * 0.2)  # Strong alignment
        else:
            alignment_score = 0.2 - (bias_data['strength'] * 0.2)  # Conflict
        
        # Enhance confidence based on alignment
        if alignment_score > 0.7:
            # Strong alignment - boost confidence
            enhanced_confidence = min(0.95, psc_confidence * 1.3)
            recommendation = f"STRONG ALIGNMENT - {tv_analysis.summary.upper()} confirms PSC {psc_direction}"
        elif alignment_score > 0.4:
            # Moderate alignment - slight boost
            enhanced_confidence = min(0.9, psc_confidence * 1.1)
            recommendation = f"MODERATE ALIGNMENT - {tv_analysis.summary.upper()} supports PSC {psc_direction}"
        else:
            # Weak/conflicting - reduce confidence
            enhanced_confidence = psc_confidence * 0.8
            recommendation = f"CONFLICT - {tv_analysis.summary.upper()} disagrees with PSC {psc_direction}"
        
        return {
            'enhanced_confidence': enhanced_confidence,
            'alignment_score': alignment_score,
            'recommendation': recommendation,
            'tv_strength': bias_data['strength'],
            'tv_direction': bias_data['direction']
        }
    
    async def get_comprehensive_market_analysis(self) -> Dict:
        """
        Get comprehensive TradingView analysis for all monitored coins across multiple timeframes
        This runs every 30 seconds to assist with trade logic decisions
        """
        current_time = time.time()
        
        # Check if we need to do a full scan
        if current_time - self.last_full_scan < self.full_scan_interval:
            return self.multi_timeframe_data
            
        logger.info("📊 Starting comprehensive TradingView market scan (1m, 5m, 10m timeframes)")
        
        comprehensive_data = {}
        total_analyses = 0
        successful_analyses = 0
        
        try:
            for coin in self.monitored_coins:
                coin_data = {
                    'symbol': coin,
                    'timeframes': {},
                    'consensus': {},
                    'trade_signals': {},
                    'timestamp': datetime.now().isoformat()
                }
                
                # Analyze each timeframe for this coin
                timeframe_scores = []
                timeframe_summaries = []
                
                for timeframe in self.timeframes:
                    total_analyses += 1
                    logger.info(f"📊 Fetching {coin} {timeframe} analysis...")
                    
                    analysis = await self.get_technical_analysis(coin, timeframe)
                    
                    if analysis:
                        successful_analyses += 1
                        
                        coin_data['timeframes'][timeframe] = {
                            'summary': analysis.summary,
                            'ma_score': analysis.ma_score,
                            'osc_score': analysis.osc_score,
                            'overall_score': analysis.overall_score,
                            'ma_recommendation': analysis.ma_recommendation,
                            'osc_recommendation': analysis.osc_recommendation
                        }
                        
                        timeframe_scores.append(analysis.overall_score)
                        timeframe_summaries.append(analysis.summary)
                        
                        # Get bias for this timeframe
                        bias = self.get_bias_confirmation(analysis)
                        coin_data['timeframes'][timeframe]['bias_direction'] = bias['direction']
                        coin_data['timeframes'][timeframe]['bias_strength'] = bias['strength']
                    else:
                        logger.warning(f"⚠️ No data for {coin} {timeframe} - using fallback")
                        # Provide fallback data instead of None to prevent N/A values
                        coin_data['timeframes'][timeframe] = {
                            'summary': 'neutral',
                            'ma_score': 0,
                            'osc_score': 0,
                            'overall_score': 0,
                            'ma_recommendation': 'neutral',
                            'osc_recommendation': 'neutral',
                            'bias_direction': 'neutral',
                            'bias_strength': 0.0
                        }
                        timeframe_scores.append(0)
                        timeframe_summaries.append('neutral')
                
                # Calculate consensus across timeframes
                if timeframe_scores:
                    avg_score = sum(timeframe_scores) / len(timeframe_scores)
                    
                    # Determine consensus direction
                    bullish_count = sum(1 for summary in timeframe_summaries if summary == 'buy')
                    bearish_count = sum(1 for summary in timeframe_summaries if summary == 'sell')
                    neutral_count = len(timeframe_summaries) - bullish_count - bearish_count
                    
                    # Calculate consensus strength
                    total_timeframes = len(self.timeframes)
                    consensus_strength = max(bullish_count, bearish_count, neutral_count) / total_timeframes
                    
                    if bullish_count > bearish_count and bullish_count > neutral_count:
                        consensus_direction = 'bullish'
                    elif bearish_count > bullish_count and bearish_count > neutral_count:
                        consensus_direction = 'bearish'
                    else:
                        consensus_direction = 'neutral'
                    
                    coin_data['consensus'] = {
                        'direction': consensus_direction,
                        'strength': consensus_strength,
                        'avg_score': avg_score,
                        'bullish_timeframes': bullish_count,
                        'bearish_timeframes': bearish_count,
                        'neutral_timeframes': neutral_count,
                        'confidence': consensus_strength * (abs(avg_score) / 52)  # Normalized confidence
                    }
                    
                    # Generate trade signals based on consensus
                    coin_data['trade_signals'] = self._generate_trade_signals(coin_data)
                    
                else:
                    coin_data['consensus'] = {
                        'direction': 'neutral',
                        'strength': 0.0,
                        'avg_score': 0,
                        'confidence': 0.0
                    }
                    coin_data['trade_signals'] = {}
                
                comprehensive_data[coin] = coin_data
                
                # Small delay between coins to avoid rate limiting
                await asyncio.sleep(0.5)
            
            # Update cached data
            self.multi_timeframe_data = comprehensive_data
            self.last_full_scan = current_time
            
            success_rate = (successful_analyses / total_analyses * 100) if total_analyses > 0 else 0
            logger.info(f"✅ Comprehensive scan complete: {successful_analyses}/{total_analyses} analyses successful ({success_rate:.1f}%)")
            
            return comprehensive_data
            
        except Exception as e:
            logger.error(f"❌ Comprehensive analysis error: {e}")
            return self.multi_timeframe_data or {}
    
    def _generate_trade_signals(self, coin_data: Dict) -> Dict:
        """Generate trade signals based on multi-timeframe analysis"""
        
        consensus = coin_data.get('consensus', {})
        timeframes = coin_data.get('timeframes', {})
        
        if not consensus or not timeframes:
            return {}
        
        signals = {
            'entry_recommendation': 'hold',
            'confidence_multiplier': 1.0,
            'risk_level': 'medium',
            'timeframe_alignment': False,
            'momentum_direction': 'neutral'
        }
        
        # Analyze timeframe alignment
        directions = []
        for tf_data in timeframes.values():
            if tf_data and tf_data.get('summary'):
                directions.append(tf_data['summary'])
        
        # Check for alignment (all timeframes agreeing)
        if len(set(directions)) == 1 and len(directions) == 3:
            signals['timeframe_alignment'] = True
            signals['confidence_multiplier'] = 1.5  # Boost confidence for alignment
            
            if directions[0] == 'buy':
                signals['entry_recommendation'] = 'strong_buy'
                signals['momentum_direction'] = 'bullish'
                signals['risk_level'] = 'low'
            elif directions[0] == 'sell':
                signals['entry_recommendation'] = 'strong_sell'
                signals['momentum_direction'] = 'bearish'
                signals['risk_level'] = 'low'
        
        # Check for divergence (timeframes disagreeing)
        elif len(set(directions)) == 3:
            signals['entry_recommendation'] = 'avoid'
            signals['confidence_multiplier'] = 0.7  # Reduce confidence for divergence
            signals['risk_level'] = 'high'
        
        # Partial alignment
        else:
            direction_counts = {d: directions.count(d) for d in set(directions)}
            dominant_direction = max(direction_counts, key=direction_counts.get)
            
            if direction_counts[dominant_direction] >= 2:
                if dominant_direction == 'buy':
                    signals['entry_recommendation'] = 'buy'
                    signals['momentum_direction'] = 'bullish'
                elif dominant_direction == 'sell':
                    signals['entry_recommendation'] = 'sell'
                    signals['momentum_direction'] = 'bearish'
                
                signals['confidence_multiplier'] = 1.2
        
        # Factor in consensus strength
        consensus_strength = consensus.get('strength', 0)
        if consensus_strength >= 0.8:  # 80%+ agreement
            signals['confidence_multiplier'] *= 1.3
        elif consensus_strength >= 0.6:  # 60%+ agreement
            signals['confidence_multiplier'] *= 1.1
        
        return signals
    
    def enhance_psc_with_comprehensive_analysis(self, symbol: str, psc_confidence: float, psc_direction: str) -> Dict:
        """
        Enhance PSC predictions using comprehensive multi-timeframe TradingView analysis
        This is called for every PSC prediction to improve accuracy
        """
        
        # Get the comprehensive data for this symbol
        symbol_data = self.multi_timeframe_data.get(symbol, {})
        
        if not symbol_data:
            logger.warning(f"⚠️ No comprehensive TradingView data for {symbol}")
            return {
                'enhanced_confidence': psc_confidence,
                'confidence_multiplier': 1.0,
                'recommendation': 'No TradingView data available',
                'alignment_score': 0.0,
                'trade_signals': {}
            }
        
        consensus = symbol_data.get('consensus', {})
        trade_signals = symbol_data.get('trade_signals', {})
        timeframes = symbol_data.get('timeframes', {})
        
        # Calculate alignment between PSC and TradingView
        tv_direction = consensus.get('direction', 'neutral')
        tv_confidence = consensus.get('confidence', 0.0)
        
        # Map PSC direction to TradingView terminology
        psc_tv_direction = 'bullish' if psc_direction in ['LONG', 'BUY'] else 'bearish' if psc_direction in ['SHORT', 'SELL'] else 'neutral'
        
        # Calculate alignment score
        if tv_direction == psc_tv_direction:
            alignment_score = tv_confidence * consensus.get('strength', 0.5)
        elif tv_direction == 'neutral':
            alignment_score = 0.3  # Neutral TradingView doesn't contradict PSC
        else:
            alignment_score = -tv_confidence * consensus.get('strength', 0.5)  # Contradiction
        
        # Get confidence multiplier from trade signals
        confidence_multiplier = trade_signals.get('confidence_multiplier', 1.0)
        
        # Apply enhancements
        enhanced_confidence = psc_confidence * confidence_multiplier
        
        # Cap at reasonable limits
        enhanced_confidence = max(0.1, min(0.95, enhanced_confidence))
        
        # Generate recommendation
        if alignment_score > 0.5:
            if trade_signals.get('timeframe_alignment', False):
                recommendation = f"Strong TradingView support - all timeframes align {tv_direction}"
            else:
                recommendation = f"TradingView supports PSC signal - {tv_direction} consensus"
        elif alignment_score > 0:
            recommendation = f"Weak TradingView support - {tv_direction} bias"
        elif alignment_score == 0:
            recommendation = "Neutral TradingView stance - no clear bias"
        else:
            recommendation = f"TradingView contradiction - shows {tv_direction} vs PSC {psc_tv_direction}"
        
        return {
            'enhanced_confidence': enhanced_confidence,
            'confidence_multiplier': confidence_multiplier,
            'recommendation': recommendation,
            'alignment_score': max(0, alignment_score),  # Only positive alignment
            'trade_signals': trade_signals,
            'consensus_data': consensus,
            'timeframe_summary': {
                '1m': timeframes.get('1m', {}).get('summary', 'N/A'),
                '5m': timeframes.get('5m', {}).get('summary', 'N/A'),
                '10m': timeframes.get('10m', {}).get('summary', 'N/A')
            }
        }

    async def log_tradingview_data(self, symbol: str, timeframe: str, analysis: TechnicalAnalysis):
        """Log TradingView data for tracking"""
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'timeframe': timeframe,
            'summary': analysis.summary if analysis else 'N/A',
            'ma_score': analysis.ma_score if analysis else 0,
            'osc_score': analysis.osc_score if analysis else 0,
            'overall_score': analysis.overall_score if analysis else 0
        }
        
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Log to CSV file
        log_file = log_dir / "tradingview_data.csv"
        
        # Write header if file doesn't exist
        if not log_file.exists():
            with open(log_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=log_entry.keys())
                writer.writeheader()
        
        # Append data
        with open(log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=log_entry.keys())
            writer.writerow(log_entry)

    async def get_single_coin_analysis(self, coin: str) -> Dict:
        """
        Get TradingView analysis for a single coin across all timeframes
        Optimized for signal validation - only fetches data for the specific coin needed
        Enhanced with real market data integration
        """
        logger.info(f"📊 Getting optimized TradingView analysis for {coin}")
        
        coin_data = {
            'symbol': coin,
            'timeframes': {},
            'consensus': {},
            'trade_signals': {},
            'real_market_data': {},
            'market_quality_score': 0.0,
            'timestamp': datetime.now().isoformat()
        }
        
        # Fetch real market data first
        if self.use_real_data and self.market_data_provider:
            try:
                real_data = await self.market_data_provider.get_market_data(coin)
                if real_data:
                    coin_data['real_market_data'] = {
                        'price': real_data.price,
                        'volume_24h': real_data.volume_24h,
                        'change_24h': real_data.change_24h,
                        'market_cap': real_data.market_cap,
                        'rsi': real_data.rsi,
                        'sma_20': real_data.sma_20,
                        'sma_50': real_data.sma_50,
                        'bb_upper': real_data.bb_upper,
                        'bb_lower': real_data.bb_lower,
                        'volume_sma': real_data.volume_sma
                    }
                    coin_data['market_quality_score'] = self.market_data_provider.get_market_quality_score(real_data)
                    logger.info(f"✅ Real market data loaded for {coin}: ${real_data.price:.6f} (Quality: {coin_data['market_quality_score']:.2f})")
                else:
                    logger.warning(f"⚠️ No real market data available for {coin}")
            except Exception as market_error:
                logger.error(f"❌ Real market data error for {coin}: {market_error}")
                coin_data['market_quality_score'] = 0.0
        
        # Add error recovery mechanism for parsing issues
        try:
            # Analyze each timeframe for this specific coin
            timeframe_scores = []
            timeframe_summaries = []
            
            for timeframe in self.timeframes:
                logger.info(f"📊 Fetching {coin} {timeframe} analysis...")
                
                analysis = await self.get_technical_analysis(coin, timeframe)
                
                if analysis:
                    coin_data['timeframes'][timeframe] = {
                        'summary': analysis.summary,
                        'ma_score': analysis.ma_score,
                        'osc_score': analysis.osc_score,
                        'overall_score': analysis.overall_score,
                        'ma_recommendation': analysis.ma_recommendation,
                        'osc_recommendation': analysis.osc_recommendation
                    }
                    
                    timeframe_scores.append(analysis.overall_score)
                    timeframe_summaries.append(analysis.summary)
                    
                    # Get bias for this timeframe
                    bias = self.get_bias_confirmation(analysis)
                    coin_data['timeframes'][timeframe]['bias_direction'] = bias['direction']
                    coin_data['timeframes'][timeframe]['bias_strength'] = bias['strength']
                else:
                    logger.warning(f"⚠️ No data for {coin} {timeframe}")
                    coin_data['timeframes'][timeframe] = None
            
            # Calculate consensus across timeframes
            if timeframe_scores:
                avg_score = sum(timeframe_scores) / len(timeframe_scores)
                
                # Determine consensus direction
                bullish_count = sum(1 for summary in timeframe_summaries if summary == 'buy')
                bearish_count = sum(1 for summary in timeframe_summaries if summary == 'sell')
                neutral_count = len(timeframe_summaries) - bullish_count - bearish_count
                
                # Calculate consensus
                total_timeframes = len(self.timeframes)
                if bullish_count > bearish_count and bullish_count > neutral_count:
                    direction = 'BUY'
                    strength = bullish_count / total_timeframes
                elif bearish_count > bullish_count and bearish_count > neutral_count:
                    direction = 'SELL' 
                    strength = bearish_count / total_timeframes
                else:
                    direction = 'NEUTRAL'
                    strength = max(bullish_count, bearish_count, neutral_count) / total_timeframes
                
                coin_data['consensus'] = {
                    'direction': direction,
                    'strength': strength,
                    'confidence': strength * 0.8 + (abs(avg_score) / 1.0) * 0.2,
                    'average_score': avg_score,
                    'timeframe_agreement': strength
                }
                
                # Trade signals enhanced with real market data
                timeframe_alignment = strength >= 0.67  # 2/3 timeframes agree
                market_quality_factor = coin_data['market_quality_score']
                
                # Adjust entry recommendations based on market quality
                base_confidence = strength
                enhanced_confidence = base_confidence * (0.7 + 0.3 * market_quality_factor)
                
                entry_recommendation = 'buy' if direction == 'BUY' and enhanced_confidence >= 0.6 and market_quality_factor >= 0.3 else \
                                     'sell' if direction == 'SELL' and enhanced_confidence >= 0.6 and market_quality_factor >= 0.3 else 'hold'
                
                coin_data['trade_signals'] = {
                    'timeframe_alignment': timeframe_alignment,
                    'entry_recommendation': entry_recommendation,
                    'confidence_level': 'high' if enhanced_confidence >= 0.8 and market_quality_factor >= 0.5 else \
                                      'medium' if enhanced_confidence >= 0.6 and market_quality_factor >= 0.3 else 'low',
                    'market_quality_factor': market_quality_factor,
                    'enhanced_confidence': enhanced_confidence,
                    'real_data_influence': 'active' if self.use_real_data else 'inactive'
                }
            
            logger.info(f"✅ Single coin analysis complete for {coin}: {coin_data['consensus'].get('direction', 'N/A')}")
            return coin_data
        
        except Exception as analysis_error:
            logger.error(f"❌ TradingView analysis error for {coin}: {analysis_error}")
            # Return minimal data structure to prevent crashes
            return {
                'symbol': coin,
                'timeframes': {'1m': None, '5m': None, '10m': None},
                'consensus': {'direction': 'NEUTRAL', 'strength': 0, 'confidence': 0},
                'trade_signals': {'timeframe_alignment': False, 'entry_recommendation': 'hold', 'real_data_influence': 'error'},
                'real_market_data': {},
                'market_quality_score': 0.0,
                'timestamp': datetime.now().isoformat(),
                'error': str(analysis_error)
            }

    def enhance_psc_with_single_coin_analysis(self, coin_analysis: Dict, psc_confidence: float, psc_direction: str) -> Dict:
        """
        Enhance PSC prediction using single coin TradingView analysis (optimized version)
        """
        if not coin_analysis or 'consensus' not in coin_analysis:
            return None
        
        consensus = coin_analysis['consensus']
        trade_signals = coin_analysis.get('trade_signals', {})
        
        tv_direction = consensus.get('direction', 'NEUTRAL')
        tv_strength = consensus.get('strength', 0)
        tv_confidence = consensus.get('confidence', 0)
        
        # Calculate alignment between PSC and TradingView
        alignment_score = 0
        
        # Direction alignment
        if psc_direction == 'LONG' and tv_direction in ['BUY']:
            alignment_score += 0.4
        elif psc_direction == 'SHORT' and tv_direction in ['SELL']:
            alignment_score += 0.4
        elif tv_direction == 'NEUTRAL':
            alignment_score += 0.2  # Neutral doesn't hurt
        
        # Strength alignment
        alignment_score += tv_strength * 0.3
        
        # Confidence alignment
        alignment_score += tv_confidence * 0.3
        
        # Calculate confidence multiplier
        if alignment_score >= 0.8:
            confidence_multiplier = 1.4
        elif alignment_score >= 0.6:
            confidence_multiplier = 1.2
        elif alignment_score >= 0.4:
            confidence_multiplier = 1.1
        else:
            confidence_multiplier = 0.9
        
        enhanced_confidence = min(psc_confidence * confidence_multiplier, 0.95)
        
        # Extract timeframe summaries from coin analysis
        timeframe_summary = {}
        if 'timeframes' in coin_analysis:
            for tf, data in coin_analysis['timeframes'].items():
                if data and 'summary' in data:
                    timeframe_summary[tf] = data['summary'].upper()
        
        return {
            'enhanced_confidence': enhanced_confidence,
            'confidence_multiplier': confidence_multiplier,
            'alignment_score': alignment_score,
            'tv_direction': tv_direction,
            'tv_strength': tv_strength,
            'tv_confidence': tv_confidence,
            'trade_signals': trade_signals,
            'timeframe_summary': timeframe_summary,
            'consensus_data': consensus,
            'recommendation': 'STRONG' if alignment_score >= 0.8 else 'MODERATE' if alignment_score >= 0.6 else 'WEAK'
        }
        
        logger.info(f"📋 TradingView data logged for {symbol} {timeframe}")


# Test function
async def test_tradingview_integration():
    """Test TradingView integration"""
    
    print("🧪 Testing TradingView Integration...")
    
    tv = TradingViewIntegration()
    
    try:
        # Test analysis for BTC
        analysis = await tv.get_technical_analysis('BTC', '1m')
        
        if analysis:
            print(f"✅ BTC Analysis:")
            print(f"   Summary: {analysis.summary}")
            print(f"   MA Score: {analysis.ma_score}")
            print(f"   OSC Score: {analysis.osc_score}")
            print(f"   Overall: {analysis.overall_score}")
            
            # Test bias confirmation
            bias = tv.get_bias_confirmation(analysis)
            print(f"   Bias Direction: {bias['direction']}")
            print(f"   Bias Strength: {bias['strength']:.2f}")
            
            # Test PSC enhancement
            enhancement = tv.enhance_psc_prediction(0.7, 'LONG', analysis)
            print(f"   Enhanced Confidence: {enhancement['enhanced_confidence']:.1%}")
            print(f"   Recommendation: {enhancement['recommendation']}")
            
        else:
            print("❌ No analysis data received")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
    finally:
        await tv.close()

if __name__ == "__main__":
    import csv
    asyncio.run(test_tradingview_integration())
