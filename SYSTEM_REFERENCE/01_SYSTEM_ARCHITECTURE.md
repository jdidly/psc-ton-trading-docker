# 🏗️ PSC Trading System - Complete Architecture Guide

**Purpose**: Detailed technical overview of system components and their relationships

---

## 📊 **SYSTEM ARCHITECTURE OVERVIEW**

```
PSC Trading System Architecture
├── 🎯 CORE TRADING ENGINE
│   ├── psc_ton_system.py (Main Bot)
│   ├── Timer-Based Trading (10-min cycles)
│   ├── Bidirectional Signals (LONG/SHORT)
│   └── Telegram Integration
│
├── 🧠 MACHINE LEARNING LAYER
│   ├── src/ml_engine.py (Prediction Engine)
│   ├── Continuous Monitoring (45-sec cycles)
│   ├── Real Data Training
│   └── Confidence Scoring
│
├── 📊 TECHNICAL ANALYSIS
│   ├── tradingview_integration.py
│   ├── 26 Technical Indicators
│   ├── Multi-Timeframe Analysis
│   └── Signal Validation
│
├── 🛡️ RISK MANAGEMENT
│   ├── Superp No-Liquidation Technology
│   ├── Dynamic Leverage (1x-10,000x)
│   ├── Position Limits
│   └── Time-Based Exits
│
├── �️ DATABASE LAYER
│   ├── psc_database.py (SQLite Operations)
│   ├── psc_data_manager.py (Unified Interface)
│   ├── Real-time Data Storage
│   ├── ACID Transactions
│   └── Multi-table Analytics
│
├── �📱 USER INTERFACES
│   ├── Telegram Bot (Database-Integrated)
│   ├── Simple Dashboard (Real-time Data)
│   ├── Web Dashboard (Live Analytics)
│   └── Database Viewer Tools
│
└── 📈 DATA & MONITORING
    ├── SQLite Database (Primary Storage)
    ├── Real-Time Queries
    ├── Performance Analytics
    ├── Trade History
    ├── ML Validation Tracking
    └── CSV Export (Backup/Analysis)
```

---

## 🎯 **CORE COMPONENT RELATIONSHIPS**

### **1. Main Trading Engine (`psc_ton_system.py`)**

**Role**: Central orchestrator that coordinates all system components

**Key Responsibilities**:
- Timer-based trading cycles (10-minute intervals)
- PSC ratio calculations and signal generation
- ML prediction integration and validation
- TradingView technical analysis coordination
- Trade execution through Superp platform
- Telegram bot communication and remote control

**Integration Points**:
```python
# Main execution flow
1. Timer Check (every 10 minutes)
   ↓
2. Price Data Collection (6 cryptocurrencies)
   ↓
3. PSC Ratio Calculation (vs TON base)
   ↓
4. ML Prediction Request (confidence scoring)
   ↓
5. TradingView Analysis (technical validation)
   ↓
6. Signal Generation (LONG/SHORT determination)
   ↓
7. Trade Execution (Superp platform)
   ↓
8. Result Logging (performance tracking)
```

### **2. ML Prediction Engine (`src/ml_engine.py`)**

**Role**: Intelligent prediction system that validates trading opportunities

**Core Capabilities**:
- **Real-Time Predictions**: Continuous market scanning every 45 seconds
- **Bidirectional Analysis**: Separate models for LONG and SHORT signals
- **Confidence Scoring**: Realistic confidence levels (0.15-0.35 typical range)
- **Small-Move Optimization**: Trained specifically for 0.12-0.20% movements
- **Continuous Learning**: Self-improving through trade outcome analysis

**Model Architecture**:
```python
ML Engine Components:
├── Win Predictor (Success probability)
├── Return Predictor (Expected profit %)
├── Confidence Predictor (Reliability score)
├── Direction Classifier (LONG/SHORT recommendation)
└── Feature Scaler (Data normalization)
```

**Integration with Trading Engine**:
- Called for every potential trading signal
- Provides confidence threshold filtering (≥65% for execution)
- Validates signals against historical patterns
- Updates models based on actual trade outcomes

### **3. TradingView Technical Analysis (`tradingview_integration.py`)**

**Role**: Professional technical analysis integration for signal validation

**Analysis Framework**:
- **Multi-Timeframe**: 1-minute, 5-minute, 10-minute charts
- **26 Indicators**: Comprehensive technical analysis suite
- **Real-Time Data**: Live market data from TradingView
- **Consensus Scoring**: Weighted agreement across timeframes

**Technical Indicators Include**:
```
Trend Indicators:
- Moving Averages (5, 10, 20, 50)
- MACD, RSI, Bollinger Bands
- Momentum Oscillators

Volume Analysis:
- Volume Profile
- On-Balance Volume
- Volume Rate of Change

Support/Resistance:
- Pivot Points
- Fibonacci Levels
- Price Action Patterns
```

### **4. Superp No-Liquidation Technology**

**Role**: Revolutionary trading platform that eliminates liquidation risk

**Key Features**:
- **Zero Liquidation Risk**: Positions cannot be forcefully closed
- **Extreme Leverage**: Up to 10,000x leverage available
- **Timer-Based Trading**: Fixed position durations (10 minutes)
- **TON Blockchain**: Fast, low-cost transactions
- **Telegram Integration**: Seamless trading through Telegram Mini App

**Leverage Categories**:
```python
class SuperpLeverageType(Enum):
    CONSERVATIVE = (1, 100)      # Low-risk signals
    MODERATE = (100, 1000)       # Medium confidence
    AGGRESSIVE = (1000, 5000)    # High confidence
    EXTREME = (5000, 10000)      # Maximum confidence
```

---

## 🔄 **DATA FLOW ARCHITECTURE**

### **Real-Time Data Pipeline**

```
External Data Sources
├── Cryptocurrency Exchanges (Price Data)
├── TradingView (Technical Analysis)
└── TON Network (Blockchain Data)
         ↓
    Data Collection Layer
├── Price Monitoring (6 cryptocurrencies)
├── Technical Analysis Updates (30-second intervals)
└── Market Sentiment Analysis
         ↓
    Processing Layer
├── PSC Ratio Calculations
├── ML Feature Engineering
├── Technical Indicator Computation
└── Signal Generation Logic
         ↓
    Decision Layer
├── ML Prediction Integration
├── Confidence Threshold Filtering
├── Risk Assessment
└── Trade Signal Validation
         ↓
    Execution Layer
├── Superp Platform Integration
├── Position Management
├── Risk Controls
└── Performance Tracking
         ↓
    Monitoring & Logging
├── Trade Execution Records
├── ML Prediction History
├── Performance Analytics
└── Error Handling & Alerts
```

### **Configuration Management**

**Central Configuration** (`config/settings.yaml`):
```yaml
trading:
  scan_interval: 45              # ML scanning frequency (seconds)
  confidence_threshold: 0.65     # Minimum ML confidence for trades
  min_signal_ratio: 6.5         # UPDATED: Logarithmic scale LONG signal threshold (was 1.25)
  max_short_ratio: 0.9           # SHORT signal threshold
  timer_interval: 600            # Trade cycle duration (seconds)

risk_management:
  max_position_size: 100         # Maximum position size ($)
  stop_loss_percentage: 0.1      # Stop loss threshold (%)
  target_profit_percentage: 0.12 # Profit target (%)
  max_daily_trades: 50           # Daily trade limit

superp:
  leverage_type: "MODERATE"      # Default leverage category
  buy_in_amount: 10              # Default position size ($)
  platform_url: "superp_api"    # Platform connection
```

---

## 🔌 **INTEGRATION POINTS**

### **External APIs**
- **TradingView**: Technical analysis data and indicators
- **Cryptocurrency Exchanges**: Real-time price feeds
- **Telegram Bot API**: Remote control and monitoring
- **TON Network**: Blockchain interaction for Superp platform

### **Internal Communication**
- **ML Engine ↔ Trading Bot**: Prediction requests and confidence scoring
- **TradingView ↔ Trading Bot**: Technical analysis validation
- **Telegram Bot ↔ All Components**: Status monitoring and control commands
- **Dashboard ↔ Configuration**: Parameter tuning and system monitoring

### **Data Persistence**
```
data/
├── live_trades.csv (Trade execution log)
├── psc_signals.csv (Signal generation history)
├── paper_trades.csv (Validation tracking)
├── ml/
│   ├── prediction_history.json (ML performance)
│   └── models/ (Trained ML models)
└── logs/
    └── tradingview_data.csv (Technical analysis history)
```

---

## 🔧 **SYSTEM REQUIREMENTS**

### **Hardware Requirements**
- **CPU**: Multi-core processor for ML calculations
- **RAM**: 4GB+ for data processing and model operations
- **Storage**: 2GB+ for data persistence and logs
- **Network**: Stable internet connection for real-time data

### **Software Dependencies**
```
Core Python Libraries:
├── scikit-learn (ML models)
├── pandas, numpy (Data processing)
├── requests (API communication)
├── asyncio (Asynchronous operations)
├── telegram-bot (Telegram integration)
└── pyyaml (Configuration management)

Optional Libraries:
├── streamlit (Web dashboard)
├── plotly (Data visualization)
└── sqlite3 (Local data storage)
```

### **External Services**
- **TradingView Account**: For technical analysis API access
- **Telegram Bot Token**: For remote monitoring and control
- **Cryptocurrency Exchange API**: For price data (if required)
- **TON Wallet**: For Superp platform integration

---

**🔗 Navigation**: Continue to `02_TRADING_LOGIC.md` for detailed trading algorithm explanations.
