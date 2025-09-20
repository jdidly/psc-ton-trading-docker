<!-- PSC Trading System - Workspace Instructions -->
## PSC Trading System Development Guidelines

### Project Type
Advanced cryptocurrency trading bot with ML engine, database integration, and real-time dashboard.

### Key Components
- Python-based trading system with asyncio for concurrent operations
- SQLite database for data persistence and ML training
- Streamlit dashboard for real-time monitoring
- TradingView API integration for market data
- Machine learning models for signal prediction

### Development Focus
- Maintain database-driven architecture
- Ensure ML models use historical data for continuous learning
- Keep signal filtering optimized for quality over quantity
- Preserve existing fixes for imports, dashboard, and database integration

### Coding Standards
- Use async/await for API calls and concurrent operations
- Implement proper error handling with logging
- Follow database-first approach for data storage
- Maintain backwards compatibility with existing data structures

### Testing and Validation Protocol
When creating test files or implementing different validation approaches:
1. **Test File Management**: After completing tests and validating that the main system is properly implemented, delete unnecessary test files to keep the workspace clean
2. **Validation Priority**: Focus on verifying core system functionality before running extensive validation tests
3. **File Cleanup**: Remove temporary test scripts, debug files, and validation tools once the main system is confirmed working
4. **System Verification**: Use `test_ml_microstructure.py` as the primary comprehensive test, then remove after validation

### Documentation Update Protocol
When making significant system changes:
1. **README Updates**: Update main README.md and relevant component READMEs to reflect new features, architecture changes, or usage instructions
2. **System Reference**: Update files in SYSTEM_REFERENCE/ folder to document new components, configurations, or integration patterns
3. **Change Log**: Create or update CHANGELOG.md to track major improvements, breaking changes, and system evolution
4. **Architecture Documentation**: Update system diagrams and architecture docs when adding new components or changing data flow
5. **Configuration Updates**: Update config examples and documentation when adding new settings or changing defaults

### Current System Status
- ✅ Enhanced signal filtering implemented (confidence thresholds raised)
- ✅ Database integration with ML engine verified
- ✅ Micro structure and PSC ratio analysis working
- ✅ All core components tested and operational
- ✅ Database-only operation implemented (no CSV/JSON fallbacks)
- ✅ Complete learning pipeline: Signal → Prediction → Validation → Learning
- ✅ Timer-based validation system: Auto-validates predictions after 10-minute Superp cycles
- ✅ ML training data enhanced 8x: 415 → 3,323 examples (ML + PSC + validation + trade data)
- 🔄 Continuous learning from historical database data active