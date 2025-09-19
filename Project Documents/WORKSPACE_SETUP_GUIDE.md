# PSC Trading System - Clean Workspace Setup Guide
## 🚀 **Complete Setup Instructions for New Agent**

---

## 📋 **PRE-SETUP CHECKLIST**

### **Requirements**
- ✅ Python 3.11+ installed
- ✅ Virtual environment capability
- ✅ Windows PowerShell (system tested on Windows)
- ✅ Internet connection (for market data APIs)

---

## 🔧 **STEP-BY-STEP SETUP**

### **Step 1: Copy Working System**
```bash
# The docker_deployment/psc_trading_minimal folder contains:
# - All fixes and enhancements implemented
# - Working database integration
# - Fixed import issues (__init__.py files)
# - Enhanced signal filtering
# - Corrected dashboard SQL queries

# Copy this entire folder to your new workspace location
```

### **Step 2: Create Virtual Environment**
```bash
# Navigate to your workspace
cd "path/to/psc_trading_minimal"

# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate (Windows CMD)
.venv\Scripts\activate.bat
```

### **Step 3: Install Dependencies**
```bash
# Install required packages
pip install streamlit plotly pandas numpy aiohttp pyyaml scikit-learn requests python-telegram-bot

# Or if requirements.txt exists:
pip install -r requirements.txt
```

### **Step 4: Verify Core Components**
```bash
# Test 1: Check imports (should work with fixed __init__.py files)
python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from src.integrated_signal_processor import IntegratedSignalProcessor
print('✅ Import fix working')
"

# Test 2: Check database
python -c "
from psc_database import PSCDatabase
db = PSCDatabase()
print('✅ Database system working')
"

# Test 3: Check ML engine
python -c "
import sys
sys.path.insert(0, 'src')
from ml_engine import MLEngine
engine = MLEngine()
print('✅ ML Engine working')
"
```

### **Step 5: Start System Components**

#### **Terminal 1: Main Trading System**
```bash
# Start production system
python start_production.py

# Should show:
# - ML engine initialization
# - Database connection
# - TradingView integration
# - Signal generation every 45 seconds
```

#### **Terminal 2: Dashboard**
```bash
# Start database dashboard
python start_database_dashboard.py

# Or use direct command:
python -m streamlit run Dashboard/database_dashboard.py --server.port=8502

# Dashboard available at: http://localhost:8502
```

#### **Terminal 3: ML Training (Optional)**
```bash
# Test ML microstructure trainer
python -c "
import asyncio
import sys
sys.path.insert(0, 'src')
from models.live_microstructure_trainer import LiveMicrostructureTrainer

async def test_training():
    trainer = LiveMicrostructureTrainer()
    await trainer.run_live_training(duration_minutes=5)

asyncio.run(test_training())
"
```

---

## 🔍 **VERIFICATION TESTS**

### **Test 1: System Integration**
```bash
python -c "
print('=== PSC SYSTEM INTEGRATION TEST ===')
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

try:
    from psc_ton_system import PSCTONTradingBot
    bot = PSCTONTradingBot()
    print('✅ Main trading bot initialized')
    print(f'✅ ML Engine: {bot.ml_engine is not None}')
    print(f'✅ Database: {bot.data_manager is not None}')
    print(f'✅ TradingView: {hasattr(bot, \"tradingview\")}')
    print('✅ All systems operational')
except Exception as e:
    print(f'❌ Integration test failed: {e}')
"
```

### **Test 2: Database Functionality**
```bash
python -c "
print('=== DATABASE FUNCTIONALITY TEST ===')
from psc_database import PSCDatabase
import sqlite3

db = PSCDatabase()
try:
    # Test database connection
    conn = db.connection
    cursor = conn.cursor()
    
    # Check tables exist
    cursor.execute(\"SELECT name FROM sqlite_master WHERE type='table'\")
    tables = [row[0] for row in cursor.fetchall()]
    print(f'✅ Database tables: {tables}')
    
    # Check record counts
    for table in ['signals', 'trades', 'validation']:
        if table in tables:
            cursor.execute(f'SELECT COUNT(*) FROM {table}')
            count = cursor.fetchone()[0]
            print(f'✅ {table}: {count} records')
    
    conn.close()
    print('✅ Database fully functional')
except Exception as e:
    print(f'❌ Database test failed: {e}')
"
```

### **Test 3: Dashboard Access**
```bash
# After starting dashboard, verify:
# 1. Navigate to http://localhost:8502
# 2. Should see:
#    - Live Signals section
#    - Trade History section  
#    - Performance Metrics
#    - No SQL column errors

# If you see errors, check PROJECT_SNAPSHOT.md for solutions
```

---

## 🎯 **IMMEDIATE PRIORITIES FOR NEW AGENT**

### **Priority 1: ML Training Investigation** 🤖
**CRITICAL QUESTION**: Are ML models using historical database data for continuous learning?

**Investigation Steps**:
1. Check ML Engine database integration:
   ```bash
   python -c "
   import sys
   sys.path.insert(0, 'src')
   from ml_engine import MLEngine
   from psc_database import PSCDatabase
   
   db = PSCDatabase()
   ml = MLEngine(data_manager=db)
   print(f'ML predictions loaded: {len(ml.predictions)}')
   print(f'Database integration: {ml.data_manager is not None}')
   "
   ```

2. Test LiveMicrostructureTrainer database usage:
   ```bash
   python -c "
   import sys
   sys.path.insert(0, 'src')
   from models.live_microstructure_trainer import LiveMicrostructureTrainer
   from psc_database import PSCDatabase
   
   db = PSCDatabase()
   trainer = LiveMicrostructureTrainer(data_manager=db)
   print(f'Trainer database: {trainer.data_manager is not None}')
   "
   ```

3. Check if models are actually retraining:
   - Look for retraining logs in `logs/` directory
   - Check if model files in `data/ml/` are being updated
   - Verify prediction accuracy is improving over time

### **Priority 2: System Optimization** ⚡
- Fix Streamlit dashboard deprecation warnings
- Optimize resource usage
- Enhance error handling

### **Priority 3: Production Enhancements** 🛡️
- Add system monitoring
- Implement automated backups
- Create performance alerts

---

## 📁 **KEY FILES REFERENCE**

### **Fixed Issues** ✅
- **`src/__init__.py`** - Package initialization (FIXED)
- **`src/models/__init__.py`** - Models package initialization (FIXED)
- **`Dashboard/database_dashboard.py`** - SQL column errors (FIXED)
- **Signal filtering** - Enhanced confidence thresholds (IMPLEMENTED)

### **Core System Files** 🔧
- **`psc_ton_system.py`** - Main trading bot
- **`src/ml_engine.py`** - ML prediction engine
- **`psc_database.py`** - Database management (PSCDatabase class)
- **`start_production.py`** - System launcher

### **Configuration** ⚙️
- **`config/settings.yaml`** - System settings
- **`requirements.txt`** - Python dependencies
- **Database**: `data/psc_trading.db` - SQLite database

---

## 🎉 **SUCCESS INDICATORS**

After setup, you should see:
- ✅ System generating signals every 45 seconds
- ✅ Dashboard showing live data at http://localhost:8502
- ✅ Database storing all signals and trades
- ✅ ML predictions with enhanced filtering (fewer weak signals)
- ✅ No import or SQL column errors

---

## 🆘 **TROUBLESHOOTING**

### **Import Errors**
- Ensure `__init__.py` files exist in `src/` and `src/models/`
- Check Python path includes current directory

### **Database Errors**
- Verify `data/` directory exists
- Check SQLite database file permissions

### **Dashboard Issues**
- Use port 8502 (not 8501) to avoid conflicts
- Run `start_database_dashboard.py` for guaranteed correct dashboard

---

**🎯 This setup guide provides everything needed to get the PSC Trading System running in a new workspace with all recent fixes and enhancements applied.**