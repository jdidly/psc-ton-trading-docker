#!/usr/bin/env python3
"""
Real Market Data Integration for PSC Trading System
Replaces simulated data with actual market feeds from multiple sources
"""

import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import json
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Real market data structure"""
    symbol: str
    price: float
    volume_24h: float
    change_24h: float
    market_cap: float
    timestamp: datetime
    
    # Technical indicators
    rsi: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_lower: Optional[float] = None
    volume_sma: Optional[float] = None

class RealMarketDataProvider:
    """
    Real market data provider using multiple sources for reliability
    """
    
    def __init__(self):
        self.session = None
        self.cache = {}
        self.cache_duration = 30  # 30 seconds cache
        self.last_request = {}
        self.rate_limit_delay = 0.1  # 100ms between requests
        
        # Data sources (primary and fallback)
        self.sources = {
            'coinbase': 'https://api.coinbase.com/v2/exchange-rates',
            'coingecko': 'https://api.coingecko.com/api/v3',
            'binance': 'https://api.binance.com/api/v3'
        }
        
        self.symbol_mappings = {
            'BTC': {'coingecko': 'bitcoin', 'binance': 'BTCUSDT', 'coinbase': 'BTC'},
            'ETH': {'coingecko': 'ethereum', 'binance': 'ETHUSDT', 'coinbase': 'ETH'},
            'SOL': {'coingecko': 'solana', 'binance': 'SOLUSDT', 'coinbase': 'SOL'},
            'DOGE': {'coingecko': 'dogecoin', 'binance': 'DOGEUSDT', 'coinbase': 'DOGE'},
            'SHIB': {'coingecko': 'shiba-inu', 'binance': 'SHIBUSDT', 'coinbase': 'SHIB'},
            'PEPE': {'coingecko': 'pepe', 'binance': 'PEPEUSDT', 'coinbase': 'PEPE'}
        }
        
    async def initialize(self):
        """Initialize HTTP session"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=10)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    'User-Agent': 'PSC-Trading-System/1.0',
                    'Accept': 'application/json'
                }
            )
            logger.info("✅ Real market data provider initialized")
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    def _respect_rate_limit(self, source: str):
        """Ensure we don't exceed rate limits"""
        now = time.time()
        if source in self.last_request:
            time_since_last = now - self.last_request[source]
            if time_since_last < self.rate_limit_delay:
                time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request[source] = time.time()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache:
            return False
        
        cached_time = self.cache[cache_key]['timestamp']
        return (time.time() - cached_time) < self.cache_duration
    
    async def get_coingecko_data(self, symbol: str) -> Optional[MarketData]:
        """Fetch data from CoinGecko API (reliable, good for fundamentals)"""
        try:
            self._respect_rate_limit('coingecko')
            
            coin_id = self.symbol_mappings[symbol]['coingecko']
            url = f"{self.sources['coingecko']}/simple/price"
            params = {
                'ids': coin_id,
                'vs_currencies': 'usd',
                'include_market_cap': 'true',
                'include_24hr_vol': 'true',
                'include_24hr_change': 'true'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    coin_data = data.get(coin_id, {})
                    
                    if coin_data:
                        return MarketData(
                            symbol=symbol,
                            price=coin_data.get('usd', 0),
                            volume_24h=coin_data.get('usd_24h_vol', 0),
                            change_24h=coin_data.get('usd_24h_change', 0),
                            market_cap=coin_data.get('usd_market_cap', 0),
                            timestamp=datetime.now()
                        )
        except Exception as e:
            logger.warning(f"CoinGecko error for {symbol}: {e}")
        
        return None
    
    async def get_binance_data(self, symbol: str) -> Optional[MarketData]:
        """Fetch data from Binance API (good for volume and price action)"""
        try:
            self._respect_rate_limit('binance')
            
            binance_symbol = self.symbol_mappings[symbol]['binance']
            
            # Get 24hr ticker statistics
            ticker_url = f"{self.sources['binance']}/ticker/24hr"
            params = {'symbol': binance_symbol}
            
            async with self.session.get(ticker_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    return MarketData(
                        symbol=symbol,
                        price=float(data.get('lastPrice', 0)),
                        volume_24h=float(data.get('volume', 0)),
                        change_24h=float(data.get('priceChangePercent', 0)),
                        market_cap=0,  # Binance doesn't provide market cap
                        timestamp=datetime.now()
                    )
        except Exception as e:
            logger.warning(f"Binance error for {symbol}: {e}")
        
        return None
    
    async def calculate_technical_indicators(self, symbol: str, market_data: MarketData) -> MarketData:
        """Calculate technical indicators using historical data"""
        try:
            # Get historical prices for technical analysis
            binance_symbol = self.symbol_mappings[symbol]['binance']
            klines_url = f"{self.sources['binance']}/klines"
            params = {
                'symbol': binance_symbol,
                'interval': '1h',
                'limit': 50
            }
            
            async with self.session.get(klines_url, params=params) as response:
                if response.status == 200:
                    klines = await response.json()
                    closes = [float(kline[4]) for kline in klines]  # Close prices
                    volumes = [float(kline[5]) for kline in klines]  # Volumes
                    
                    if len(closes) >= 20:
                        # Simple Moving Averages
                        market_data.sma_20 = sum(closes[-20:]) / 20
                        if len(closes) >= 50:
                            market_data.sma_50 = sum(closes[-50:]) / 50
                        
                        # RSI calculation (simplified)
                        if len(closes) >= 14:
                            gains = []
                            losses = []
                            for i in range(1, 15):
                                change = closes[-i] - closes[-i-1]
                                if change > 0:
                                    gains.append(change)
                                    losses.append(0)
                                else:
                                    gains.append(0)
                                    losses.append(abs(change))
                            
                            avg_gain = sum(gains) / 14
                            avg_loss = sum(losses) / 14
                            
                            if avg_loss != 0:
                                rs = avg_gain / avg_loss
                                market_data.rsi = 100 - (100 / (1 + rs))
                        
                        # Volume SMA
                        market_data.volume_sma = sum(volumes[-20:]) / 20
                        
                        # Bollinger Bands (simplified)
                        sma = market_data.sma_20
                        variance = sum([(close - sma) ** 2 for close in closes[-20:]]) / 20
                        std_dev = variance ** 0.5
                        market_data.bb_upper = sma + (2 * std_dev)
                        market_data.bb_lower = sma - (2 * std_dev)
                        
        except Exception as e:
            logger.warning(f"Technical indicators error for {symbol}: {e}")
        
        return market_data
    
    async def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """
        Get comprehensive market data with fallback sources
        """
        cache_key = f"{symbol}_market_data"
        
        # Check cache first
        if self._is_cache_valid(cache_key):
            logger.debug(f"📋 Using cached market data for {symbol}")
            return self.cache[cache_key]['data']
        
        await self.initialize()
        
        # Try primary source (CoinGecko) first
        market_data = await self.get_coingecko_data(symbol)
        
        # If primary fails, try Binance
        if not market_data:
            market_data = await self.get_binance_data(symbol)
            logger.info(f"📊 Using Binance data for {symbol}")
        else:
            logger.info(f"📊 Using CoinGecko data for {symbol}")
        
        if market_data:
            # Add technical indicators
            market_data = await self.calculate_technical_indicators(symbol, market_data)
            
            # Cache the result
            self.cache[cache_key] = {
                'data': market_data,
                'timestamp': time.time()
            }
            
            logger.info(f"✅ Real market data fetched for {symbol}: ${market_data.price:.6f}")
            return market_data
        else:
            logger.error(f"❌ Failed to fetch market data for {symbol}")
            return None
    
    async def get_multiple_coins_data(self, symbols: List[str]) -> Dict[str, MarketData]:
        """Get market data for multiple coins efficiently"""
        tasks = [self.get_market_data(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        market_data = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, MarketData):
                market_data[symbol] = result
            else:
                logger.error(f"❌ Failed to get data for {symbol}: {result}")
        
        return market_data
    
    def get_market_quality_score(self, market_data: MarketData) -> float:
        """
        Calculate market quality score for signal filtering
        Higher score = better trading conditions
        """
        score = 0.0
        
        # Volume factor (higher volume = better)
        if market_data.volume_sma and market_data.volume_24h > market_data.volume_sma:
            score += 0.3
        
        # Volatility factor (moderate volatility preferred)
        if abs(market_data.change_24h) > 1 and abs(market_data.change_24h) < 10:
            score += 0.2
        
        # RSI factor (not overbought/oversold)
        if market_data.rsi and 30 < market_data.rsi < 70:
            score += 0.2
        
        # Trend factor (price above SMA20)
        if market_data.sma_20 and market_data.price > market_data.sma_20:
            score += 0.15
        
        # Market cap factor (avoid very small caps)
        if market_data.market_cap > 1_000_000_000:  # > $1B
            score += 0.15
        
        return min(score, 1.0)


# Test function
async def test_real_market_data():
    """Test real market data provider"""
    
    print("Testing Real Market Data Provider...")
    
    provider = RealMarketDataProvider()
    
    try:
        # Test single coin
        btc_data = await provider.get_market_data('BTC')
        
        if btc_data:
            print(f"BTC Real Data:")
            print(f"   Price: ${btc_data.price:,.2f}")
            print(f"   24h Change: {btc_data.change_24h:.2f}%")
            print(f"   24h Volume: ${btc_data.volume_24h:,.0f}")
            print(f"   Market Cap: ${btc_data.market_cap:,.0f}")
            print(f"   RSI: {btc_data.rsi:.1f}" if btc_data.rsi else "   RSI: N/A")
            print(f"   SMA20: ${btc_data.sma_20:.2f}" if btc_data.sma_20 else "   SMA20: N/A")
            print(f"   Quality Score: {provider.get_market_quality_score(btc_data):.2f}")
        
        # Test multiple coins
        print(f"\nTesting Multiple Coins...")
        coins = ['BTC', 'ETH', 'SOL']
        multi_data = await provider.get_multiple_coins_data(coins)
        
        for symbol, data in multi_data.items():
            quality = provider.get_market_quality_score(data)
            print(f"   {symbol}: ${data.price:.6f} (Quality: {quality:.2f})")
            
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await provider.close()

if __name__ == "__main__":
    asyncio.run(test_real_market_data())