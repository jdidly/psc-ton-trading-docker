# ⚡ PSC Trading Logic - Complete Algorithm Guide

**Purpose**: Detailed explanation of core trading algorithms and decision-making processes

---

## 🎯 **CORE TRADING ALGORITHM**

### **PSC (Price-Signal-Confidence) Arbitrage Strategy**

The PSC Trading System identifies arbitrage opportunities by analyzing price relationships between cryptocurrencies and TON, validated through machine learning predictions and technical analysis.

**Core Formula**:
```python
# UPDATED: Logarithmic PSC ratio for meaningful comparisons
import math
log_ratio = math.log10(current_price) - math.log10(ton_price)
psc_ratio = log_ratio + 6  # Shift to positive scale (range ~1-11)
```

**Why This Works**:
- Identifies relative value misalignments between assets
- TON serves as the base currency for ratio calculations
- Ratios outside normal ranges indicate arbitrage opportunities
- Small-move focus (0.12-0.20%) aligns with Superp's timer-based trading

---

## 📊 **BIDIRECTIONAL SIGNAL GENERATION**

### **LONG Signal Logic**

**When to Go LONG** (Buy/Bullish Position):
```python
def generate_long_signal(crypto, ratio, confidence, ta_sentiment):
    # Strong LONG signals
    if ratio >= 2.0 and confidence > 0.7 and ta_sentiment >= 0.6:
        return {"direction": "LONG", "strength": "STRONG", "leverage": "AGGRESSIVE"}
    
    # Good LONG signals  
    elif ratio >= 1.5 and confidence > 0.6 and ta_sentiment >= 0.5:
        return {"direction": "LONG", "strength": "GOOD", "leverage": "MODERATE"}
    
    # Entry-level LONG signals
    elif ratio >= 1.25 and confidence > 0.65 and ta_sentiment >= 0.4:
        return {"direction": "LONG", "strength": "ENTRY", "leverage": "CONSERVATIVE"}
    
    return None  # No signal
```

**LONG Signal Criteria**:
1. **PSC Ratio**: ≥ 6.5 (indicating crypto outperforming TON - logarithmic scale)
2. **ML Confidence**: ≥ 65% prediction confidence
3. **Technical Analysis**: Bullish indicators from TradingView
4. **Timer Window**: Must be in minutes 0-3 of 10-minute cycle
5. **Volume Confirmation**: Above-average trading volume preferred

### **SHORT Signal Logic**

**When to Go SHORT** (Sell/Bearish Position):
```python
def generate_short_signal(crypto, ratio, confidence, ta_sentiment):
    # Strong SHORT signals
    if ratio <= 0.7 and confidence > 0.7 and ta_sentiment <= 0.3:
        return {"direction": "SHORT", "strength": "STRONG", "leverage": "AGGRESSIVE"}
    
    # Good SHORT signals
    elif ratio <= 0.8 and confidence > 0.6 and ta_sentiment <= 0.4:
        return {"direction": "SHORT", "strength": "GOOD", "leverage": "MODERATE"}
    
    # Entry-level SHORT signals
    elif ratio <= 0.9 and confidence > 0.65 and ta_sentiment <= 0.5:
        return {"direction": "SHORT", "strength": "ENTRY", "leverage": "CONSERVATIVE"}
    
    return None  # No signal
```

**SHORT Signal Criteria**:
1. **PSC Ratio**: ≤ 5.5 (indicating crypto underperforming TON - logarithmic scale)
2. **ML Confidence**: ≥ 65% prediction confidence  
3. **Technical Analysis**: Bearish indicators from TradingView
4. **Timer Window**: Must be in minutes 0-3 of 10-minute cycle
5. **Volume Confirmation**: Above-average trading volume preferred

---

## ⏰ **TIMER-BASED TRADING SYSTEM**

### **10-Minute Trading Cycles**

The system operates on strict 10-minute cycles aligned with Superp's timer-based positions:

```python
def get_timer_status():
    current_minute = datetime.now().minute % 10
    
    if current_minute <= 3:
        return "ENTRY_WINDOW"     # Full leverage available
    elif current_minute <= 6:
        return "MID_TIMER"        # 80% leverage efficiency
    elif current_minute <= 8:
        return "LATE_TIMER"       # 60% leverage efficiency
    else:
        return "EXIT_WINDOW"      # 40% leverage, prepare for exit
```

### **Entry and Exit Logic**

**Entry Conditions** (Minutes 0-3):
- ✅ All signal criteria must be met
- ✅ ML confidence ≥ 65%
- ✅ TradingView technical analysis supports direction
- ✅ PSC ratio within acceptable ranges
- ✅ No conflicting signals from other timeframes

**Exit Conditions**:
```python
def determine_exit_strategy(position, current_minute, profit_pct):
    # Profit target hit
    if profit_pct >= 0.12:
        return "TAKE_PROFIT"
    
    # Stop loss hit
    elif profit_pct <= -0.1:
        return "STOP_LOSS"
    
    # Timer-based exit (minutes 9-10)
    elif current_minute >= 9:
        return "TIMER_EXIT"
    
    # Continue holding
    else:
        return "HOLD"
```

---

## 🧠 **ML INTEGRATION & VALIDATION**

### **Prediction Request Process**

```python
def validate_signal_with_ml(crypto_price, ton_price, psc_ratio):
    # Request ML prediction
    prediction = ml_engine.predict_trade_outcome(
        psc_price=crypto_price,
        ton_price=ton_price,
        ratio=psc_ratio
    )
    
    if prediction:
        confidence = prediction.get('confidence', 0)
        recommendation = prediction.get('recommendation', 'NEUTRAL')
        
        # Only proceed if confidence meets threshold
        if confidence >= 0.65:
            return {
                'approved': True,
                'direction': recommendation,
                'confidence': confidence,
                'expected_return': prediction.get('expected_return', 0)
            }
    
    return {'approved': False, 'reason': 'Low ML confidence'}
```

### **Continuous ML Monitoring**

The system runs independent ML scanning every 45 seconds:

```python
async def continuous_ml_monitoring():
    while True:
        for crypto in MONITORED_CRYPTOS:
            # Get current market data
            price_data = await get_real_time_prices(crypto)
            
            # Calculate PSC ratio
            ratio = calculate_psc_ratio(price_data)
            
            # Get ML prediction
            prediction = await ml_engine.predict_trade_outcome(
                price_data['price'], 
                price_data['ton_price'], 
                ratio
            )
            
            # Log prediction for validation
            log_ml_prediction(crypto, prediction, price_data)
            
            # Check if signal qualifies for trade consideration
            if prediction and prediction['confidence'] >= 0.65:
                await queue_potential_signal(crypto, prediction, price_data)
        
        await asyncio.sleep(45)  # Wait 45 seconds before next scan
```

---

## 📊 **TECHNICAL ANALYSIS INTEGRATION**

### **TradingView Signal Validation**

```python
def validate_with_technical_analysis(crypto, direction):
    # Get TradingView analysis for multiple timeframes
    analysis_1m = get_tradingview_analysis(crypto, '1m')
    analysis_5m = get_tradingview_analysis(crypto, '5m')
    analysis_10m = get_tradingview_analysis(crypto, '10m')
    
    # Calculate consensus score
    consensus = calculate_consensus([analysis_1m, analysis_5m, analysis_10m])
    
    # Validate direction alignment
    if direction == "LONG" and consensus >= 0.6:
        return {"approved": True, "ta_score": consensus}
    elif direction == "SHORT" and consensus <= 0.4:
        return {"approved": True, "ta_score": consensus}
    else:
        return {"approved": False, "ta_score": consensus}
```

### **26-Indicator Analysis**

**Trend Indicators**:
- Moving Averages (SMA 5, 10, 20, EMA 12, 26)
- MACD (12, 26, 9)
- Bollinger Bands (20, 2)
- Parabolic SAR

**Momentum Indicators**:
- RSI (14)
- Stochastic (14, 3, 3)
- Williams %R (14)
- CCI (20)

**Volume Indicators**:
- Volume Rate of Change
- On-Balance Volume
- Volume Profile

**Support/Resistance**:
- Pivot Points
- Fibonacci Retracements
- Support/Resistance Levels

---

## 🛡️ **RISK MANAGEMENT ALGORITHMS**

### **Position Sizing Logic**

```python
def calculate_position_size(confidence, signal_strength, available_capital):
    # Base position size
    base_size = 10  # $10 minimum
    
    # Confidence multiplier
    confidence_multiplier = min(confidence / 0.65, 2.0)  # Max 2x for high confidence
    
    # Signal strength multiplier
    strength_multipliers = {
        "ENTRY": 1.0,
        "GOOD": 1.5,
        "STRONG": 2.0
    }
    
    # Calculate final position size
    position_size = base_size * confidence_multiplier * strength_multipliers[signal_strength]
    
    # Cap at maximum allowed
    max_position = min(available_capital * 0.1, 100)  # 10% of capital or $100 max
    
    return min(position_size, max_position)
```

### **Dynamic Leverage Selection**

```python
def select_leverage_category(confidence, timer_position, signal_strength):
    # Base leverage from signal strength
    base_leverage = {
        "ENTRY": "CONSERVATIVE",
        "GOOD": "MODERATE", 
        "STRONG": "AGGRESSIVE"
    }[signal_strength]
    
    # Adjust for confidence
    if confidence >= 0.8:
        # High confidence can upgrade leverage
        if base_leverage == "MODERATE":
            base_leverage = "AGGRESSIVE"
        elif base_leverage == "AGGRESSIVE":
            base_leverage = "EXTREME"
    
    # Adjust for timer position
    if timer_position in ["MID_TIMER", "LATE_TIMER"]:
        # Reduce leverage for late entries
        downgrade_map = {
            "EXTREME": "AGGRESSIVE",
            "AGGRESSIVE": "MODERATE",
            "MODERATE": "CONSERVATIVE"
        }
        base_leverage = downgrade_map.get(base_leverage, "CONSERVATIVE")
    
    return base_leverage
```

---

## 📈 **PROFIT TARGET & STOP LOSS SYSTEM**

### **Dynamic Target Setting**

```python
def calculate_profit_targets(confidence, direction, volatility):
    # Base targets
    base_profit = 0.0012  # 0.12%
    base_stop = 0.001     # 0.1%
    
    # Adjust for confidence
    if confidence >= 0.8:
        profit_target = base_profit * 1.5  # 0.18% for high confidence
    elif confidence >= 0.7:
        profit_target = base_profit * 1.25 # 0.15% for good confidence
    else:
        profit_target = base_profit        # 0.12% for entry-level
    
    # Adjust for volatility
    if volatility > 0.02:  # High volatility
        profit_target *= 1.2
        stop_loss = base_stop * 1.2
    else:
        stop_loss = base_stop
    
    return {
        'profit_target': profit_target,
        'stop_loss': stop_loss,
        'risk_reward_ratio': profit_target / stop_loss
    }
```

### **Exit Strategy Implementation**

```python
def monitor_position_exit(position):
    current_profit = calculate_current_profit(position)
    timer_minute = get_current_timer_minute()
    
    # Check exit conditions in priority order
    
    # 1. Profit target reached
    if current_profit >= position['profit_target']:
        return execute_exit(position, "PROFIT_TARGET")
    
    # 2. Stop loss hit
    elif current_profit <= -position['stop_loss']:
        return execute_exit(position, "STOP_LOSS")
    
    # 3. Timer-based exit (minutes 9-10)
    elif timer_minute >= 9:
        return execute_exit(position, "TIMER_EXIT")
    
    # 4. ML confidence drop (continuous monitoring)
    elif get_current_ml_confidence(position['crypto']) < 0.4:
        return execute_exit(position, "CONFIDENCE_DROP")
    
    # 5. Technical analysis reversal
    elif technical_analysis_reversal(position['crypto'], position['direction']):
        return execute_exit(position, "TECHNICAL_REVERSAL")
    
    # Continue holding
    return "HOLD"
```

---

## 🔄 **CONTINUOUS LEARNING INTEGRATION**

### **Trade Outcome Analysis**

```python
def analyze_trade_outcome(completed_trade):
    # Calculate actual vs predicted returns
    actual_return = completed_trade['exit_price'] / completed_trade['entry_price'] - 1
    predicted_return = completed_trade['ml_prediction']['expected_return']
    
    # Accuracy metrics
    direction_correct = (
        (actual_return > 0 and completed_trade['direction'] == 'LONG') or
        (actual_return < 0 and completed_trade['direction'] == 'SHORT')
    )
    
    # Small-move accuracy (key metric)
    small_move_success = abs(actual_return) >= 0.0012  # Hit 0.12% target
    
    # Update ML model with outcome
    ml_engine.update_with_outcome({
        'prediction': completed_trade['ml_prediction'],
        'actual_outcome': {
            'return': actual_return,
            'direction_correct': direction_correct,
            'small_move_success': small_move_success,
            'confidence_validated': completed_trade['ml_prediction']['confidence'] > 0.65
        }
    })
    
    # Log for analysis
    log_trade_outcome(completed_trade, {
        'direction_accuracy': direction_correct,
        'small_move_accuracy': small_move_success,
        'return_accuracy': abs(actual_return - predicted_return)
    })
```

---

**🔗 Navigation**: Continue to `03_ML_ENGINE_GUIDE.md` for detailed ML system explanations.
