#!/usr/bin/env python3
"""
PSC Trading System - Database Dashboard Direct Launcher
Directly starts the database dashboard without fallback options
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the database dashboard directly"""
    print("🎯 PSC Trading System - Database Dashboard Direct Launcher")
    print("=" * 60)
    
    # Check if streamlit is available
    try:
        import streamlit
        print("✅ Streamlit available")
    except ImportError:
        print("❌ Streamlit not installed")
        print("Installing streamlit...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit", "plotly", "altair"])
        print("✅ Streamlit installed")
    
    # Direct path to database dashboard
    dashboard_path = "Dashboard/database_dashboard.py"
    
    if not Path(dashboard_path).exists():
        print(f"❌ Database dashboard not found at: {dashboard_path}")
        return
    
    print(f"✅ Found database dashboard: {dashboard_path}")
    print(f"\n🚀 Starting Database Dashboard (FORCED)")
    print("📊 Dashboard will open in your browser")
    print("🔗 URL: http://localhost:8501")
    print("\n📋 Controls:")
    print("   • Ctrl+C to stop")
    print("   • Refresh browser to reload")
    
    # Start streamlit dashboard directly
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            dashboard_path,
            "--server.port=8501",
            "--server.address=0.0.0.0",
            "--browser.serverAddress=localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Database Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Database Dashboard failed to start: {e}")

if __name__ == "__main__":
    main()
