#!/usr/bin/env python3
"""
Railway Deployment Configuration for PSC Trading System
Optimized for Railway cloud deployment with proper Unicode handling
"""

import os
import sys
import logging
from pathlib import Path

# Set UTF-8 encoding for Railway environment
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUNBUFFERED'] = '1'

def setup_railway_environment():
    """Setup Railway-specific environment variables and configurations"""
    
    # Ensure required directories exist
    Path('logs').mkdir(exist_ok=True)
    Path('database').mkdir(exist_ok=True)
    Path('data').mkdir(exist_ok=True)
    Path('data/ml').mkdir(exist_ok=True)
    
    # Configure logging for Railway
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/railway.log', encoding='utf-8')
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("🚄 Railway deployment environment configured")
    
    # Railway-specific configurations
    port = os.environ.get('PORT', '8080')
    os.environ['HEALTH_CHECK_PORT'] = port
    
    # Database configuration for Railway (SQLite for simplicity)
    database_url = os.environ.get('DATABASE_URL', 'sqlite:///database/psc_trading.db')
    
    # Telegram configuration (should be set in Railway environment)
    if not os.environ.get('TELEGRAM_BOT_TOKEN'):
        logger.warning("⚠️ TELEGRAM_BOT_TOKEN not set - using default from config")
    
    if not os.environ.get('TELEGRAM_CHAT_ID'):
        logger.warning("⚠️ TELEGRAM_CHAT_ID not set - using default from config")
    
    logger.info(f"🔧 Railway configuration:")
    logger.info(f"   Port: {port}")
    logger.info(f"   Database: {database_url}")
    logger.info(f"   Telegram Bot: {'✅ Configured' if os.environ.get('TELEGRAM_BOT_TOKEN') else '❌ Not set'}")
    
    return {
        'port': port,
        'database_url': database_url,
        'telegram_configured': bool(os.environ.get('TELEGRAM_BOT_TOKEN'))
    }

def main():
    """Main Railway startup function"""
    try:
        # Setup Railway environment
        config = setup_railway_environment()
        
        # Import and run the main system
        logger = logging.getLogger(__name__)
        logger.info("🚀 Starting PSC Trading System on Railway...")
        
        # Add current directory to Python path
        current_dir = Path(__file__).parent
        sys.path.insert(0, str(current_dir))
        
        # Import the production startup
        import start_production
        
        # Run the production system
        start_production.main()
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"❌ Railway startup failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()
