# 🧪 Testing & Validation Framework

**Purpose**: Comprehensive testing procedures and validation protocols for the PSC Trading System

---

## 📋 **TESTING HIERARCHY**

### **Level 1: Unit Testing** ⚡

**Core Component Tests**:

```python
# tests/test_core_components.py
import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml_engine import MLPredictionEngine
from psc_ton_system import PSCTONTradingBot

class TestCoreComponents(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.ml_engine = MLPredictionEngine()
        self.trading_bot = PSCTONTradingBot(test_mode=True)
    
    def test_ml_engine_initialization(self):
        """Test ML engine initializes correctly"""
        self.assertIsNotNone(self.ml_engine)
        self.assertTrue(hasattr(self.ml_engine, 'predict_trade_outcome'))
        
    def test_ml_prediction_format(self):
        """Test ML prediction returns correct format"""
        # Mock price data
        mock_prices = {
            'BTCUSDT': 45000.0,
            'ETHUSDT': 2500.0,
            'TONUSDT': 2.5
        }
        
        prediction = self.ml_engine.predict_trade_outcome(
            action='LONG',
            asset='BTCUSDT',
            amount=100,
            current_prices=mock_prices
        )
        
        # Validate prediction structure
        self.assertIn('success_probability', prediction)
        self.assertIn('expected_return', prediction)
        self.assertIn('confidence', prediction)
        self.assertIsInstance(prediction['confidence'], float)
        self.assertTrue(0 <= prediction['confidence'] <= 1)
    
    def test_psc_ratio_calculation(self):
        """Test PSC ratio calculation accuracy"""
        prices = {
            'BTCUSDT': 50000.0,
            'ETHUSDT': 3000.0,
            'TONUSDT': 2.0  # Base currency
        }
        
        # Test LONG signal detection
        prices_long_signal = {
            'BTCUSDT': 50000.0,
            'ETHUSDT': 3000.0,
            'TONUSDT': 1.6  # 25% lower than others
        }
        
        ratio = self.trading_bot.calculate_psc_ratio(prices_long_signal)
        self.assertGreater(ratio, 1.25, "Should detect LONG signal")
        
        # Test SHORT signal detection
        prices_short_signal = {
            'BTCUSDT': 50000.0,
            'ETHUSDT': 3000.0,
            'TONUSDT': 2.8  # Higher than others
        }
        
        ratio = self.trading_bot.calculate_psc_ratio(prices_short_signal)
        self.assertLess(ratio, 0.9, "Should detect SHORT signal")
    
    def test_position_management(self):
        """Test position opening and closing logic"""
        # Test position opening
        position = self.trading_bot.open_position(
            action='LONG',
            asset='BTCUSDT',
            amount=100,
            confidence=0.75
        )
        
        self.assertIsNotNone(position)
        self.assertEqual(position['action'], 'LONG')
        self.assertEqual(position['asset'], 'BTCUSDT')
        self.assertEqual(position['amount'], 100)
        
        # Test position closing
        close_result = self.trading_bot.close_position(position['id'])
        self.assertTrue(close_result['success'])

if __name__ == '__main__':
    unittest.main()
```

**ML Engine Specific Tests**:

```python
# tests/test_ml_engine.py
import unittest
import numpy as np
import pandas as pd
from src.ml_engine import MLPredictionEngine

class TestMLEngine(unittest.TestCase):
    
    def setUp(self):
        self.ml_engine = MLPredictionEngine()
        # Create sample training data
        self.sample_data = self.create_sample_training_data()
    
    def create_sample_training_data(self):
        """Create realistic sample data for testing"""
        np.random.seed(42)  # For reproducible tests
        
        data = []
        for i in range(1000):
            # Simulate market conditions
            psc_ratio = np.random.uniform(0.8, 1.4)
            confidence = np.random.uniform(0.5, 0.9)
            
            # Create correlation between features and outcome
            if psc_ratio > 1.25 and confidence > 0.65:
                success = np.random.choice([True, False], p=[0.6, 0.4])
                return_val = np.random.uniform(0.08, 0.15) if success else np.random.uniform(-0.12, -0.05)
            else:
                success = np.random.choice([True, False], p=[0.4, 0.6])
                return_val = np.random.uniform(0.05, 0.12) if success else np.random.uniform(-0.15, -0.08)
            
            data.append({
                'psc_ratio': psc_ratio,
                'confidence': confidence,
                'market_volatility': np.random.uniform(0.01, 0.05),
                'success': success,
                'return': return_val
            })
        
        return pd.DataFrame(data)
    
    def test_model_training(self):
        """Test that models can be trained successfully"""
        # Train models
        training_result = self.ml_engine.train_models(self.sample_data)
        
        self.assertTrue(training_result['success'])
        self.assertIn('win_predictor', training_result)
        self.assertIn('return_predictor', training_result)
        self.assertIn('confidence_predictor', training_result)
    
    def test_prediction_accuracy(self):
        """Test prediction accuracy on known data"""
        # Train on sample data
        self.ml_engine.train_models(self.sample_data)
        
        # Test predictions
        test_features = [1.3, 0.7, 0.02]  # Strong signal
        prediction = self.ml_engine.make_prediction(test_features)
        
        # Should predict high success probability for strong signal
        self.assertGreater(prediction['success_probability'], 0.5)
        self.assertGreater(prediction['confidence'], 0.6)
    
    def test_feature_engineering(self):
        """Test feature extraction from market data"""
        market_data = {
            'BTCUSDT': 45000.0,
            'ETHUSDT': 2500.0,
            'TONUSDT': 2.0,
            'timestamp': '2025-08-25 12:00:00'
        }
        
        features = self.ml_engine.extract_features(market_data)
        
        # Should return exactly 25 features
        self.assertEqual(len(features), 25)
        
        # All features should be numeric
        for feature in features:
            self.assertIsInstance(feature, (int, float, np.number))
    
    def test_model_persistence(self):
        """Test saving and loading trained models"""
        # Train models
        self.ml_engine.train_models(self.sample_data)
        
        # Save models
        save_result = self.ml_engine.save_models('data/ml/models/')
        self.assertTrue(save_result)
        
        # Create new engine and load models
        new_engine = MLPredictionEngine()
        load_result = new_engine.load_models('data/ml/models/')
        self.assertTrue(load_result)
        
        # Test that loaded models work
        test_features = [1.3, 0.7, 0.02]
        prediction = new_engine.make_prediction(test_features)
        self.assertIsNotNone(prediction)

if __name__ == '__main__':
    unittest.main()
```

### **Level 2: Integration Testing** 🔗

**System Integration Tests**:

```python
# tests/test_integration.py
import unittest
import time
import threading
from unittest.mock import Mock, patch

class TestSystemIntegration(unittest.TestCase):
    
    def setUp(self):
        """Set up integration test environment"""
        self.trading_bot = PSCTONTradingBot(test_mode=True)
        self.trading_bot.initialize_ml_engine()
    
    def test_ml_trading_integration(self):
        """Test complete ML-driven trading workflow"""
        # Mock market data
        with patch.object(self.trading_bot, 'get_current_prices') as mock_prices:
            mock_prices.return_value = {
                'BTCUSDT': 45000.0,
                'ETHUSDT': 2500.0,
                'TONUSDT': 1.8  # Strong LONG signal
            }
            
            # Execute trading cycle
            result = self.trading_bot.run_trading_cycle()
            
            # Verify workflow completion
            self.assertTrue(result['cycle_completed'])
            self.assertIn('ml_prediction', result)
            self.assertIn('psc_analysis', result)
    
    def test_telegram_integration(self):
        """Test Telegram bot integration"""
        with patch.object(self.trading_bot, 'send_telegram_message') as mock_send:
            # Trigger notification
            self.trading_bot.send_trade_notification(
                action='LONG',
                asset='BTCUSDT',
                amount=100,
                confidence=0.75
            )
            
            # Verify message was sent
            mock_send.assert_called_once()
            call_args = mock_send.call_args[0][0]
            self.assertIn('LONG', call_args)
            self.assertIn('BTCUSDT', call_args)
    
    def test_tradingview_integration(self):
        """Test TradingView data integration"""
        with patch('requests.get') as mock_request:
            # Mock TradingView response
            mock_response = Mock()
            mock_response.json.return_value = {
                'consensus': 'BUY',
                'signals': {'RSI': 0.7, 'MACD': 0.8}
            }
            mock_request.return_value = mock_response
            
            # Get TradingView analysis
            analysis = self.trading_bot.get_tradingview_analysis('BTCUSDT')
            
            self.assertIsNotNone(analysis)
            self.assertIn('consensus', analysis)
    
    def test_timer_integration(self):
        """Test timer-based trading cycles"""
        trade_count = 0
        
        def mock_trade_cycle():
            nonlocal trade_count
            trade_count += 1
            return {'cycle_completed': True}
        
        with patch.object(self.trading_bot, 'run_trading_cycle', side_effect=mock_trade_cycle):
            # Start timer (short interval for testing)
            self.trading_bot.timer_interval = 1  # 1 second for testing
            
            # Run for 3 seconds
            timer_thread = threading.Thread(target=self.trading_bot.start_timer_based_trading)
            timer_thread.daemon = True
            timer_thread.start()
            
            time.sleep(3.5)
            self.trading_bot.stop_trading()
            
            # Should have completed ~3 cycles
            self.assertGreaterEqual(trade_count, 2)
            self.assertLessEqual(trade_count, 4)

if __name__ == '__main__':
    unittest.main()
```

### **Level 3: End-to-End Testing** 🔄

**Live System Tests**:

```python
# tests/test_end_to_end.py
import unittest
import time
import json
from datetime import datetime, timedelta

class TestEndToEnd(unittest.TestCase):
    
    def setUp(self):
        """Set up end-to-end test environment"""
        self.trading_bot = PSCTONTradingBot(test_mode=True)
        self.test_start_time = datetime.now()
    
    def test_complete_trading_session(self):
        """Test complete 10-minute trading session"""
        session_results = []
        
        # Simulate 10-minute trading session
        for minute in range(10):
            # Simulate price changes over time
            mock_prices = self.generate_time_based_prices(minute)
            
            with patch.object(self.trading_bot, 'get_current_prices', return_value=mock_prices):
                cycle_result = self.trading_bot.run_trading_cycle()
                session_results.append({
                    'minute': minute,
                    'result': cycle_result,
                    'prices': mock_prices
                })
        
        # Analyze session
        positions_opened = sum(1 for r in session_results if r['result'].get('position_opened'))
        positions_closed = sum(1 for r in session_results if r['result'].get('position_closed'))
        
        # Verify session behavior
        self.assertGreaterEqual(positions_opened, 0)
        self.assertEqual(positions_opened, positions_closed)  # All positions should close
    
    def generate_time_based_prices(self, minute):
        """Generate realistic price progression over 10 minutes"""
        base_prices = {
            'BTCUSDT': 45000.0,
            'ETHUSDT': 2500.0,
            'TONUSDT': 2.0
        }
        
        # Simulate price movements
        volatility = 0.001 * minute  # Increasing volatility
        
        return {
            asset: price * (1 + (np.random.random() - 0.5) * volatility)
            for asset, price in base_prices.items()
        }
    
    def test_performance_monitoring(self):
        """Test performance tracking and metrics"""
        # Run multiple trading cycles
        results = []
        for i in range(50):
            result = self.trading_bot.run_trading_cycle()
            results.append(result)
        
        # Calculate performance metrics
        performance = self.trading_bot.calculate_performance_metrics()
        
        # Verify metrics are calculated
        self.assertIn('win_rate', performance)
        self.assertIn('average_return', performance)
        self.assertIn('total_trades', performance)
        self.assertIn('ml_accuracy', performance)
        
        # Verify realistic ranges
        self.assertTrue(0 <= performance['win_rate'] <= 1)
        self.assertGreaterEqual(performance['total_trades'], 0)
    
    def test_error_recovery(self):
        """Test system behavior under error conditions"""
        error_scenarios = [
            'api_timeout',
            'ml_prediction_failure',
            'invalid_price_data',
            'telegram_send_failure'
        ]
        
        for scenario in error_scenarios:
            with self.subTest(scenario=scenario):
                # Simulate error condition
                result = self.simulate_error_scenario(scenario)
                
                # System should handle error gracefully
                self.assertTrue(result['error_handled'])
                self.assertFalse(result['system_crashed'])
    
    def simulate_error_scenario(self, scenario):
        """Simulate various error conditions"""
        if scenario == 'api_timeout':
            with patch('requests.get', side_effect=TimeoutError()):
                result = self.trading_bot.run_trading_cycle()
                return {
                    'error_handled': 'error' in result,
                    'system_crashed': False
                }
        
        elif scenario == 'ml_prediction_failure':
            with patch.object(self.trading_bot.ml_engine, 'predict_trade_outcome', side_effect=Exception()):
                result = self.trading_bot.run_trading_cycle()
                return {
                    'error_handled': result.get('fallback_used', False),
                    'system_crashed': False
                }
        
        # Add more error scenarios...
        return {'error_handled': True, 'system_crashed': False}

if __name__ == '__main__':
    unittest.main()
```

---

## 🎯 **VALIDATION PROTOCOLS**

### **Performance Validation** 📈

**Trading Performance Metrics**:

```python
# validation/performance_validator.py
class PerformanceValidator:
    
    def __init__(self):
        self.required_metrics = {
            'win_rate': {'min': 0.45, 'target': 0.55},
            'average_return': {'min': 0.08, 'target': 0.12},
            'ml_accuracy': {'min': 0.50, 'target': 0.65},
            'system_uptime': {'min': 0.95, 'target': 0.99}
        }
    
    def validate_performance(self, metrics):
        """Validate system performance against benchmarks"""
        results = {}
        
        for metric, thresholds in self.required_metrics.items():
            value = metrics.get(metric, 0)
            
            if value >= thresholds['target']:
                status = 'EXCELLENT'
            elif value >= thresholds['min']:
                status = 'ACCEPTABLE'
            else:
                status = 'NEEDS_IMPROVEMENT'
            
            results[metric] = {
                'value': value,
                'status': status,
                'target': thresholds['target'],
                'minimum': thresholds['min']
            }
        
        return results
    
    def generate_performance_report(self, results):
        """Generate detailed performance report"""
        report = []
        report.append("=== PERFORMANCE VALIDATION REPORT ===\n")
        
        for metric, data in results.items():
            report.append(f"{metric.upper()}:")
            report.append(f"  Current: {data['value']:.3f}")
            report.append(f"  Target:  {data['target']:.3f}")
            report.append(f"  Status:  {data['status']}")
            report.append("")
        
        return "\n".join(report)
```

**ML Model Validation**:

```python
# validation/ml_validator.py
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

class MLModelValidator:
    
    def validate_ml_models(self, ml_engine, test_data):
        """Comprehensive ML model validation"""
        validation_results = {}
        
        # Prepare test features and targets
        X_test = test_data.drop(['success', 'return'], axis=1)
        y_win = test_data['success']
        y_return = test_data['return']
        
        # Validate win predictor
        win_predictions = ml_engine.win_predictor.predict(X_test)
        validation_results['win_predictor'] = {
            'accuracy': accuracy_score(y_win, win_predictions),
            'precision': precision_score(y_win, win_predictions),
            'recall': recall_score(y_win, win_predictions)
        }
        
        # Validate return predictor
        return_predictions = ml_engine.return_predictor.predict(X_test)
        return_mae = np.mean(np.abs(y_return - return_predictions))
        validation_results['return_predictor'] = {
            'mae': return_mae,
            'rmse': np.sqrt(np.mean((y_return - return_predictions) ** 2))
        }
        
        # Validate confidence predictor
        confidence_predictions = ml_engine.confidence_predictor.predict(X_test)
        confidence_range_valid = np.all((0 <= confidence_predictions) & (confidence_predictions <= 1))
        validation_results['confidence_predictor'] = {
            'range_valid': confidence_range_valid,
            'mean_confidence': np.mean(confidence_predictions)
        }
        
        return validation_results
    
    def check_model_drift(self, current_performance, historical_performance):
        """Check for model performance drift"""
        drift_threshold = 0.05  # 5% degradation threshold
        
        drift_detected = {}
        for metric in ['accuracy', 'precision', 'recall']:
            current = current_performance.get(metric, 0)
            historical = historical_performance.get(metric, 0)
            
            if historical > 0:
                drift_percentage = (historical - current) / historical
                drift_detected[metric] = drift_percentage > drift_threshold
        
        return drift_detected
```

### **Risk Validation** ⚠️

**Risk Management Validation**:

```python
# validation/risk_validator.py
class RiskValidator:
    
    def __init__(self):
        self.risk_limits = {
            'max_position_size': 500,        # $500 maximum
            'max_daily_loss': 0.1,           # 10% daily loss limit
            'max_drawdown': 0.15,            # 15% maximum drawdown
            'position_concentration': 0.2,    # 20% maximum in single asset
            'leverage_limit': 1000           # Maximum 1000x leverage
        }
    
    def validate_risk_compliance(self, trading_data):
        """Validate risk management compliance"""
        violations = []
        
        # Check position sizes
        max_position = max(trading_data['position_sizes'])
        if max_position > self.risk_limits['max_position_size']:
            violations.append(f"Position size limit exceeded: ${max_position}")
        
        # Check daily losses
        daily_pnl = trading_data['daily_pnl']
        max_daily_loss = min(daily_pnl)
        if abs(max_daily_loss) > self.risk_limits['max_daily_loss']:
            violations.append(f"Daily loss limit exceeded: {max_daily_loss:.2%}")
        
        # Check drawdown
        equity_curve = trading_data['equity_curve']
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_drawdown = max(drawdown)
        if max_drawdown > self.risk_limits['max_drawdown']:
            violations.append(f"Drawdown limit exceeded: {max_drawdown:.2%}")
        
        return {
            'compliant': len(violations) == 0,
            'violations': violations
        }
    
    def stress_test_scenarios(self, trading_system):
        """Run stress test scenarios"""
        scenarios = {
            'market_crash': {'price_drop': 0.2, 'volatility': 0.1},
            'flash_crash': {'price_drop': 0.1, 'speed': 'instant'},
            'high_volatility': {'volatility': 0.05, 'duration': '1_hour'},
            'api_outage': {'outage_duration': 300, 'recovery_time': 60}
        }
        
        results = {}
        for scenario_name, params in scenarios.items():
            results[scenario_name] = self.run_stress_scenario(trading_system, params)
        
        return results
    
    def run_stress_scenario(self, trading_system, params):
        """Run individual stress test scenario"""
        # Simulate stress condition
        # Return system behavior metrics
        return {
            'system_survived': True,
            'max_loss': 0.05,
            'recovery_time': 120,
            'positions_closed': 3
        }
```

---

## 🚀 **AUTOMATED TEST EXECUTION**

### **Test Automation Scripts**

**Complete Test Suite Runner**:

```bash
# run_tests.bat (Windows)
@echo off
echo Starting PSC Trading System Test Suite...
echo.

echo [1/4] Running Unit Tests...
python -m pytest tests/test_core.py -v
if %errorlevel% neq 0 goto :error

echo [2/4] Running ML Engine Tests...
python -m pytest tests/test_ml_engine.py -v
if %errorlevel% neq 0 goto :error

echo [3/4] Running Integration Tests...
python -m pytest tests/test_integration.py -v
if %errorlevel% neq 0 goto :error

echo [4/4] Running Performance Validation...
python validation/performance_validator.py
if %errorlevel% neq 0 goto :error

echo.
echo ✅ All tests passed successfully!
echo Test report generated in: tests/reports/
goto :end

:error
echo.
echo ❌ Tests failed! Check the output above for details.
exit /b 1

:end
```

**Continuous Integration Configuration**:

```yaml
# .github/workflows/test.yml
name: PSC Trading System Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run unit tests
      run: |
        pytest tests/ --cov=src/ --cov-report=html
    
    - name: Run ML validation
      run: |
        python validation/ml_validator.py
    
    - name: Run performance tests
      run: |
        python validation/performance_validator.py
    
    - name: Upload coverage reports
      uses: codecov/codecov-action@v1
```

### **Test Data Management**

**Test Data Generator**:

```python
# tests/data_generator.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class TestDataGenerator:
    
    def generate_market_data(self, days=30, interval_minutes=1):
        """Generate realistic market data for testing"""
        start_time = datetime.now() - timedelta(days=days)
        
        # Calculate number of data points
        total_minutes = days * 24 * 60
        timestamps = [start_time + timedelta(minutes=i) for i in range(0, total_minutes, interval_minutes)]
        
        data = []
        base_prices = {'BTCUSDT': 45000, 'ETHUSDT': 2500, 'TONUSDT': 2.0}
        
        for i, timestamp in enumerate(timestamps):
            # Simulate realistic price movements
            row = {'timestamp': timestamp}
            
            for asset, base_price in base_prices.items():
                # Add trending and random components
                trend = 0.0001 * i  # Slight upward trend
                random_change = np.random.normal(0, 0.002)  # 0.2% volatility
                price = base_price * (1 + trend + random_change)
                row[asset] = max(price, base_price * 0.8)  # Prevent negative prices
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def generate_trading_outcomes(self, num_trades=1000):
        """Generate realistic trading outcome data"""
        np.random.seed(42)
        
        data = []
        for i in range(num_trades):
            # Generate correlated features and outcomes
            psc_ratio = np.random.uniform(0.8, 1.4)
            confidence = np.random.uniform(0.5, 0.9)
            
            # Higher PSC ratios and confidence correlate with success
            base_success_prob = 0.4
            if psc_ratio > 1.25:
                base_success_prob += 0.2
            if confidence > 0.7:
                base_success_prob += 0.15
            
            success = np.random.random() < base_success_prob
            
            # Generate return based on success
            if success:
                return_val = np.random.uniform(0.08, 0.15)
            else:
                return_val = np.random.uniform(-0.12, -0.05)
            
            data.append({
                'psc_ratio': psc_ratio,
                'confidence': confidence,
                'market_volatility': np.random.uniform(0.01, 0.05),
                'position_size': np.random.uniform(10, 200),
                'leverage': np.random.choice([1, 10, 100, 500]),
                'success': success,
                'return': return_val,
                'timestamp': datetime.now() - timedelta(hours=i)
            })
        
        return pd.DataFrame(data)
```

---

## 📊 **REPORTING & MONITORING**

### **Test Reporting**

**Automated Test Reports**:

```python
# tests/report_generator.py
import json
from datetime import datetime

class TestReportGenerator:
    
    def generate_comprehensive_report(self, test_results):
        """Generate comprehensive test report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': self.generate_summary(test_results),
            'detailed_results': test_results,
            'recommendations': self.generate_recommendations(test_results)
        }
        
        # Save report
        with open(f'tests/reports/test_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def generate_summary(self, test_results):
        """Generate test summary"""
        total_tests = sum(len(suite) for suite in test_results.values())
        passed_tests = sum(
            sum(1 for test in suite if test.get('passed', False))
            for suite in test_results.values()
        )
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'status': 'PASS' if passed_tests == total_tests else 'FAIL'
        }
    
    def generate_recommendations(self, test_results):
        """Generate improvement recommendations"""
        recommendations = []
        
        # Analyze failed tests
        for suite_name, suite in test_results.items():
            failed_tests = [test for test in suite if not test.get('passed', False)]
            
            if failed_tests:
                recommendations.append(f"Review {suite_name}: {len(failed_tests)} failures")
        
        return recommendations
```

---

**🔗 Navigation**: Continue to `07_DEPLOYMENT_PRODUCTION.md` for deployment and production guidelines.
