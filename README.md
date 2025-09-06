# 🐳 PSC TON Trading System - Docker Deployment

A sophisticated autonomous trading system with ML-driven predictions and SuperP no-liquidation technology, containerized for cloud deployment.

## 🚀 Quick Railway Deployment

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template)

### One-Click Cloud Deployment
1. Click "Deploy on Railway" (or fork this repo)
2. Connect your GitHub account to Railway
3. Configure environment variables (see below)
4. Deploy automatically to the cloud!

## 🏗️ System Architecture

This Docker deployment includes:
- **Trading Engine**: Complete PSC TON trading system with ML integration
- **PostgreSQL Database**: Persistent data storage for trades and predictions  
- **Streamlit Dashboard**: Real-time monitoring and analytics interface

## 📋 Required Environment Variables

Configure these in Railway or your deployment platform:

```bash
# Core System (REQUIRED)
DISABLE_TIMER_ALERTS=true
DATABASE_URL=postgresql://user:password@db:5432/trading_db

# Optional Telegram Integration
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# Trading Parameters (Optional)
MAX_POSITION_SIZE=1000
RISK_PERCENTAGE=2.0
MIN_CONFIDENCE=0.7
```

## � Local Development

### Prerequisites
- Docker and Docker Compose installed
- Git installed

### Quick Start
```bash
# Clone the repository
git clone https://github.com/yourusername/psc-ton-trading-docker.git
cd psc-ton-trading-docker

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f trading-system

# Access dashboard
open http://localhost:8501
```

### Services Overview
- **Trading System**: Main application (port: internal)
- **PostgreSQL**: Database (port: 5432)  
- **Dashboard**: Streamlit interface (port: 8501)

## � Key Features

### Advanced Trading Engine
- **SuperP Technology**: No-liquidation positioning system
- **ML Predictions**: Multi-model ensemble with confidence scoring
- **Risk Management**: Dynamic position sizing and stop-loss
- **TradingView Integration**: Real-time market data and analysis

### Enhanced Prediction Validator
- Real-time validation of ML predictions vs actual market outcomes
- Performance tracking and accuracy metrics
- Automatic model retraining triggers based on accuracy degradation
- Confidence-based position sizing optimization

### Comprehensive Dashboard
- Live trading performance metrics and P&L tracking
- ML prediction accuracy analysis with confidence distributions
- Risk analysis and portfolio exposure overview
- Historical trade analysis with detailed breakdowns

## 🔧 Configuration

### Trading Parameters
Edit `docker-compose.yml` environment variables:

```yaml
environment:
  - DISABLE_TIMER_ALERTS=true  # Disable notification spam
  - MAX_TRADES_PER_DAY=10     # Daily trade limit
  - MIN_CONFIDENCE=0.7        # Minimum ML confidence
  - RISK_PERCENTAGE=2.0       # Risk per trade
```

### Database Configuration
PostgreSQL is pre-configured with:
- Database: `trading_db`
- Username: `trading_user`  
- Password: `secure_password`
- Port: `5432`

## 📈 Monitoring & Health Checks

### Container Status
```bash
# Check all containers
docker-compose ps

# View trading system logs
docker-compose logs trading-system

# Monitor database
docker-compose logs postgres

# Dashboard logs
docker-compose logs dashboard
```

### Access Points
- **Local Dashboard**: http://localhost:8501
- **Railway Dashboard**: Your assigned Railway URL
- **Health Status**: Available in container logs

## 🔒 Security Features

- Environment variables for all sensitive data
- No hardcoded API keys or passwords in code
- Database credentials managed via environment
- Secure inter-container communication
- Minimal attack surface with Alpine Linux base

## 📁 Project Structure

```
psc-trading-docker/
├── docker-compose.yml      # Multi-container orchestration
├── Dockerfile             # Trading system container
├── psc_ton_system.py      # Main trading engine (181KB)
├── requirements.txt       # Python dependencies
├── dashboard.py           # Streamlit monitoring interface
├── data/                  # Trading data storage
│   ├── live_trades.csv    # Active trade records
│   ├── ml_signals.csv     # ML prediction signals
│   └── psc_signals.csv    # PSC system signals
└── README.md              # This file
```

## 🚨 Important Production Notes

### Data Management
- CSV files start with headers-only on fresh deployment
- System generates authentic trading data from live market conditions
- No fake or test data included - all data is real market-derived
- Data persists in Docker volumes for continuity

### Performance Optimization
- **Enhanced Prediction Validator** is the active modern validation system
- Legacy Paper Trading Validator is archived (not used)
- Timer notifications disabled by default to prevent spam
- Optimized for cloud deployment with minimal resource usage

## 🛠️ Troubleshooting

### Common Issues

**Containers won't start:**
```bash
# Full reset
docker-compose down -v
docker-compose pull
docker-compose up -d
```

**Database connection issues:**
```bash
# Reset database with fresh data
docker-compose down -v
docker volume prune -f
docker-compose up -d
```

**Dashboard not accessible:**
- Verify port 8501 is open
- Check Streamlit container: `docker-compose logs dashboard`
- Ensure no firewall blocking

**"Paper Trading Validator not found" warning:**
- This is expected - Enhanced Prediction Validator is the active system
- Warning can be ignored as it doesn't affect functionality

### Debugging Commands
```bash
# All services logs
docker-compose logs

# Follow logs in real-time
docker-compose logs -f

# Specific service debug
docker-compose logs trading-system
docker-compose exec trading-system /bin/bash
```

## 🌐 Railway Deployment Guide

### Step-by-Step Railway Setup

1. **Fork or Clone this repository to your GitHub**

2. **Create Railway Account**: Sign up at [railway.app](https://railway.app)

3. **New Project**: Click "New Project" → "Deploy from GitHub repo"

4. **Select Repository**: Choose your forked psc-ton-trading-docker repo

5. **Configure Services**: Railway will auto-detect docker-compose.yml
   - **trading-system**: Main application
   - **postgres**: Database service  
   - **dashboard**: Streamlit interface

6. **Set Environment Variables** in Railway dashboard:
   ```
   DISABLE_TIMER_ALERTS=true
   TELEGRAM_BOT_TOKEN=your_token_here
   TELEGRAM_CHAT_ID=your_chat_id_here
   ```

7. **Deploy**: Railway automatically builds and deploys

8. **Access Dashboard**: Use Railway-provided URL

### Railway-Specific Configuration

Railway automatically:
- ✅ Builds Docker containers
- ✅ Manages service discovery
- ✅ Provides persistent volumes
- ✅ Handles load balancing
- ✅ Generates HTTPS URLs

## 📜 License

This project is licensed under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Test locally with Docker Compose
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Submit a pull request

## 📞 Support

For deployment issues or questions:
- 🐛 **Issues**: Create an issue in this repository
- 🚂 **Railway**: Check Railway deployment logs in dashboard
- 🐳 **Docker**: Review container health with `docker-compose ps`
- 📊 **Trading**: Monitor system performance via Streamlit dashboard

---

**🚀 Built for autonomous trading with ML intelligence and cloud-first deployment.**

**Excluded:** Development tools, tests, extra utilities, and documentation not needed for production.

**Result:** Lightweight, secure, production-ready trading system that can run anywhere Docker is supported.
