# PSC TON Trading System Dashboard

## 🖥️ Dashboard Features

### ⚙️ Configuration Panel
- **Scan Interval**: Adjust how often the system scans for opportunities (5-300 seconds)
- **Confidence Threshold**: Set minimum confidence required to open positions (0.1-1.0)
- **Ratio Threshold**: Minimum PSC/TON ratio to consider (1.0-5.0)
- **Position Management**: Max concurrent positions and position size
- **Superp Settings**: Configure extreme leverage ranges and time limits
- **ML Configuration**: Enable/disable ML predictions and set retrain intervals

### 📈 Trading Monitor
- **Real-time Metrics**: Total profit, success rate, active positions, recent signals
- **Recent Trades Table**: Latest trading activity with timestamps and results
- **Recent Signals Table**: Latest PSC opportunities detected
- **Cumulative Profit Chart**: Visual profit tracking over time

### 🧠 ML Analytics
- **Model Performance**: Total predictions, accuracy rates, model status
- **Prediction History**: Recent ML predictions with outcomes
- **ML Controls**: Retrain models, save/load model states

### 📋 System Logs
- **Live Log Viewing**: Real-time system logs with filtering
- **Log Level Control**: Filter by INFO, DEBUG, WARNING, ERROR
- **Auto-refresh**: Optional automatic log refreshing
- **Multiple Log Files**: System and trading logs

### 📊 Performance Analytics
- **Comprehensive Metrics**: Profit analysis, win rates, trade statistics
- **Performance Charts**: Profit distribution, confidence vs success correlation
- **Historical Analysis**: Best/worst trades, average performance

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install streamlit plotly pandas pyyaml
```

### 2. Start Dashboard
```bash
# Method 1: Use launcher script
python start_dashboard.py

# Method 2: Direct streamlit
streamlit run dashboard.py --server.port=8501 --server.address=0.0.0.0
```

### 3. Access Dashboard
- **Local**: http://localhost:8501
- **Network**: http://your-ip:8501 (for remote access)

## 📱 Telegram Integration

### New Bot Commands
- `/dashboard` - Get dashboard access information and setup instructions
- `/logs` - View recent system logs directly in Telegram
- `/config` - Show current configuration settings
- `/performance` - Get performance summary with key metrics

### Remote Monitoring
- Access logs remotely via Telegram
- Check configuration without dashboard access
- Get performance updates on mobile
- Dashboard setup instructions via bot

## ⚙️ Configuration Management

### Via Dashboard
1. Navigate to "Configuration" tab
2. Adjust parameters using sliders and inputs
3. Click "Save Configuration" to apply changes
4. Use "Load Defaults" to reset to default values

### Via Bot Commands
- `/config` - View current settings
- `/settings` - Access advanced settings through bot
- Configuration changes via dashboard are immediately applied

### Configuration File
- Location: `config/settings.yaml`
- Manual editing supported
- Dashboard automatically reloads configuration changes

## 📊 Data Management

### Data Sources
- **Trades**: `data/live_trades.csv`
- **Signals**: `data/psc_signals.csv`
- **Predictions**: `data/ml/prediction_history.json`
- **Logs**: `logs/hybrid_system.log`

### Export Options
- Download trading data as CSV
- Export configuration settings
- Access log files for analysis

## 🔧 Advanced Features

### Real-time Updates
- Dashboard auto-refreshes data every 30 seconds
- Live charts update with new trade data
- Real-time log streaming available

### ML Model Management
- Retrain models directly from dashboard
- Save/load trained models
- Monitor prediction accuracy in real-time
- View feature importance and model metrics

### System Controls
- Start/stop trading bot from dashboard
- Restart system with new configuration
- Monitor system health and status
- Control ML engine training schedule

## 🛡️ Security & Access

### Local Access
- Dashboard runs on localhost by default
- No external access unless specifically configured

### Network Access
- Configure `--server.address=0.0.0.0` for network access
- Use firewall rules to restrict access
- Consider VPN for secure remote access

### Data Protection
- All configuration in local files
- No cloud dependencies
- Logs contain no sensitive API keys (when properly configured)

## 🔍 Troubleshooting

### Dashboard Won't Start
1. Check if streamlit is installed: `pip install streamlit`
2. Verify you're in the correct directory (`core_system/`)
3. Check for port conflicts (default: 8501)

### Data Not Loading
1. Verify trading system has been running and generating data
2. Check file permissions for `data/` directory
3. Ensure CSV files are not corrupted

### Configuration Not Saving
1. Check write permissions for `config/` directory
2. Verify YAML format is correct
3. Check for file locks or access issues

### ML Engine Issues
1. Ensure sklearn is installed: `pip install scikit-learn`
2. Check `data/ml/` directory exists and is writable
3. Verify training data is available

## 📝 Usage Examples

### Adjusting Scan Frequency
1. Open dashboard → Configuration tab
2. Adjust "Scan Interval" slider (recommended: 30-60 seconds)
3. Click "Save Configuration"
4. Restart bot for changes to take effect

### Monitoring Performance
1. Navigate to "Performance" tab
2. View profit charts and success rates
3. Use filters to analyze specific time periods
4. Export data for external analysis

### ML Model Training
1. Go to "ML Analytics" tab
2. Check current model performance
3. Click "Retrain Models" when enough new data available
4. Monitor accuracy improvements over time

## 🎯 Best Practices

### Configuration
- Start with conservative settings (lower leverage, higher confidence thresholds)
- Monitor performance before increasing risk parameters
- Save working configurations before making changes

### Monitoring
- Check dashboard daily for performance trends
- Use Telegram commands for quick status updates
- Monitor logs for any error patterns

### Maintenance
- Retrain ML models weekly or after significant market changes
- Export trading data regularly for backup
- Update configuration based on performance analysis

## 🔗 Integration

### With Trading System
- Dashboard reads directly from trading system files
- Configuration changes apply to running system
- Real-time monitoring of active trades

### With Telegram Bot
- All dashboard features accessible via bot commands
- Remote monitoring and control
- Mobile-friendly status updates

### With ML Engine
- Direct model management from dashboard
- Real-time prediction monitoring
- Training data visualization
