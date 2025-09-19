#!/usr/bin/env python3
"""
PSC Trading System - Dashboard Launcher
Start the database-integrated dashboard for monitoring
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the dashboard"""
    print("🎯 PSC Trading System - Dashboard Launcher")
    print("=" * 50)
    
    # Check if streamlit is available
    try:
        import streamlit
        print("✅ Streamlit available")
    except ImportError:
        print("❌ Streamlit not installed")
        print("Installing streamlit...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit", "plotly", "altair"])
        print("✅ Streamlit installed")
    
    # Try to find the best dashboard to use
    dashboard_options = [
        ("Dashboard/database_dashboard.py", "Database Dashboard (Recommended)"),
        ("Dashboard/dashboard.py", "Full Feature Dashboard"),
        ("simple_database_dashboard.py", "Simple Database Dashboard"),
        ("Dashboard/minimal_dashboard.py", "Minimal Dashboard")
    ]
    
    selected_dashboard = None
    for dashboard_path, description in dashboard_options:
        if Path(dashboard_path).exists():
            print(f"✅ Found: {description}")
            selected_dashboard = dashboard_path
            break
        else:
            print(f"❌ Not found: {dashboard_path}")
    
    if not selected_dashboard:
        print("❌ No dashboard found!")
        return

    print(f"\n🚀 Starting: {selected_dashboard}")
    print(f"📊 Dashboard Type: {'DATABASE DASHBOARD' if 'database_dashboard' in selected_dashboard else 'OTHER DASHBOARD'}")
    print("📊 Dashboard will open in your browser")
    print("🔗 URL: http://localhost:8501")
    print("\n📋 Controls:")
    print("   • Ctrl+C to stop")
    print("   • Refresh browser to reload")
    
    # Start streamlit
    try:
        if selected_dashboard == "simple_database_dashboard.py":
            # This is not a streamlit app
            print("⚠️ Simple database dashboard uses built-in HTTP server")
            subprocess.run([sys.executable, selected_dashboard])
        else:
            # Start streamlit dashboard
            subprocess.run([
                sys.executable, "-m", "streamlit", "run", 
                selected_dashboard,
                "--server.port=8501",
                "--server.address=0.0.0.0",
                "--browser.serverAddress=localhost"
            ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Dashboard failed to start: {e}")

if __name__ == "__main__":
    main()
