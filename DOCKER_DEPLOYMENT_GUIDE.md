# 🐳 PSC Trading System - Docker Deployment Guide

## 🚀 Quick Start (5 Minutes)

### **Prerequisites**
- Docker installed
- Docker Compose installed  
- Telegram bot token (from @BotFather)

### **1. Setup Environment**
```bash
# Copy environment template
cp .env.template .env

# Edit .env file with your values
nano .env  # or your preferred editor
```

### **2. Deploy with Docker Compose**
```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f psc-trading

# Check status
docker-compose ps
```

### **3. Verify Deployment**
- **Trading System**: http://localhost:8080/health
- **Dashboard**: http://localhost:8501
- **Database**: localhost:5432 (if enabled)

## 📊 **Services Overview**

### **psc-trading** (Main System)
- **Port**: 8080
- **Function**: Core trading system with ML
- **Health Check**: `/health` endpoint
- **Data**: Persistent volumes for data/logs

### **postgres** (Database)
- **Port**: 5432
- **Function**: ML model and trading data storage
- **Data**: Persistent PostgreSQL data

### **dashboard** (Optional Web UI)
- **Port**: 8501
- **Function**: Streamlit web interface
- **Access**: http://localhost:8501

## 🔧 **Configuration**

### **Required Environment Variables**
```env
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id  
POSTGRES_PASSWORD=secure_password
```

### **Optional Configuration**
```env
INITIAL_BALANCE=1000
MIN_CONFIDENCE=0.60
ML_TRAINING_ENABLED=true
LOG_LEVEL=INFO
```

## 🚀 **Cloud Deployment**

### **Docker Hub Deployment**
```bash
# Build image
docker build -t psc-trading .

# Tag for registry
docker tag psc-trading your-registry/psc-trading

# Push to registry
docker push your-registry/psc-trading
```

### **Railway Deployment**
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

### **DigitalOcean App Platform**
```bash
# Use docker-compose.yml for deployment
# Set environment variables in DO dashboard
```

## 🔄 **Management Commands**

### **View Logs**
```bash
docker-compose logs -f psc-trading     # Trading system
docker-compose logs -f postgres        # Database
docker-compose logs -f dashboard       # Web dashboard
```

### **Restart Services**
```bash
docker-compose restart psc-trading     # Restart trading
docker-compose restart                 # Restart all
```

### **Update System**
```bash
docker-compose down                     # Stop services
docker-compose build --no-cache        # Rebuild images
docker-compose up -d                    # Start services
```

### **Backup Data**
```bash
# Backup database
docker-compose exec postgres pg_dump -U postgres psc_trading > backup.sql

# Backup trading data  
docker cp psc-trading-system:/app/data ./data_backup
```

## 🔒 **Security Best Practices**

1. **Change default passwords**
2. **Use strong environment variables**
3. **Enable firewall on cloud deployments**
4. **Regular security updates**
5. **Monitor logs for suspicious activity**

## 🎯 **Production Checklist**

- [ ] Set secure POSTGRES_PASSWORD
- [ ] Configure Telegram bot properly
- [ ] Set appropriate resource limits
- [ ] Enable log rotation
- [ ] Set up monitoring alerts
- [ ] Configure backup schedule
- [ ] Test disaster recovery

## 🆘 **Troubleshooting**

### **Common Issues**

**Service won't start:**
```bash
docker-compose logs psc-trading
```

**Database connection issues:**
```bash
docker-compose exec postgres psql -U postgres -d psc_trading
```

**Permission issues:**
```bash
docker-compose down
docker system prune -f
docker-compose up -d
```

### **Performance Optimization**

**For low-memory systems:**
```yaml
# Add to docker-compose.yml under psc-trading service
deploy:
  resources:
    limits:
      memory: 512M
    reservations:
      memory: 256M
```

**For high-frequency trading:**
```yaml
# Increase resource limits
deploy:
  resources:
    limits:
      memory: 2G
      cpus: '1.0'
```

## 📞 **Support**

- Check logs: `docker-compose logs -f`
- Health status: http://localhost:8080/health
- Database status: `docker-compose exec postgres pg_isready`
- System resources: `docker stats`
