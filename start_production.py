#!/usr/bin/env python3
"""
PSC Trading System - Production Startup Script
Starts the main trading system in Docker environment
"""

import os
import sys
import logging
import asyncio
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/startup.log')
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main startup function"""
    try:
        logger.info("Starting PSC Trading System in Docker...")
        
        # Import and run the main system
        logger.info("Importing main trading system...")
        
        # Add current directory to Python path
        current_dir = Path(__file__).parent
        sys.path.insert(0, str(current_dir))
        
        # Import the main system
        import psc_ton_system
        
        logger.info("Starting trading system...")
        
        # Run the async main function properly
        asyncio.run(psc_ton_system.main())
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()
