#!/usr/bin/env python3
"""
Test the fully enhanced ML engine with all data sources
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ml_engine import MLEngine
from psc_data_manager import PSCDataManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_full_enhanced_dataset():
    """Test the complete enhanced ML training dataset"""
    
    print("🧪 Testing FULLY Enhanced ML Training Dataset")
    print("=" * 55)
    
    try:
        # Initialize data manager with correct database path
        data_manager = PSCDataManager(db_path="database/psc_trading.db")
        
        # Initialize ML engine with data manager
        ml_engine = MLEngine(data_manager=data_manager)
        
        print(f"\n📊 COMPLETE Training Dataset Summary:")
        print(f"Total predictions loaded: {len(ml_engine.predictions)}")
        
        # Detailed breakdown
        source_counts = {}
        signal_type_counts = {}
        outcome_counts = {}
        
        for pred in ml_engine.predictions:
            source = pred.get('source', 'unknown')
            signal_type = pred.get('signal_type', 'unknown')
            outcome = pred.get('actual_outcome')
            
            source_counts[source] = source_counts.get(source, 0) + 1
            signal_type_counts[signal_type] = signal_type_counts.get(signal_type, 0) + 1
            
            if outcome:
                outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        
        print(f"\n📈 Complete Breakdown by Source:")
        for source, count in sorted(source_counts.items()):
            print(f"   {source}: {count}")
            
        print(f"\n🎯 Complete Breakdown by Signal Type:")  
        for sig_type, count in sorted(signal_type_counts.items()):
            print(f"   {sig_type}: {count}")
            
        print(f"\n✅ Actual Outcomes for Supervised Learning:")
        for outcome, count in sorted(outcome_counts.items()):
            print(f"   {outcome}: {count}")
            
        # Calculate improvements
        original_size = 415
        current_size = len(ml_engine.predictions)
        improvement = ((current_size - original_size) / original_size) * 100
        
        supervised_examples = sum(outcome_counts.values())
        supervised_pct = (supervised_examples / current_size) * 100 if current_size > 0 else 0
        
        print(f"\n🚀 ENHANCEMENT SUMMARY:")
        print(f"   Original dataset: {original_size} predictions")
        print(f"   Enhanced dataset: {current_size} predictions")
        print(f"   Improvement: +{improvement:.1f}% more training data")
        print(f"   Supervised examples: {supervised_examples} ({supervised_pct:.1f}%)")
        
        print(f"\n🎯 TRAINING QUALITY:")
        print(f"   Historical signals: {source_counts.get('psc_historical', 0) + source_counts.get('ml_historical', 0)}")
        print(f"   Validated outcomes: {source_counts.get('validation_data', 0)}")
        print(f"   Trade outcomes: {source_counts.get('trade_outcomes', 0)}")
        
        print(f"\n🧠 ML Models now have access to:")
        print(f"   📊 {current_size:,} total training examples")
        print(f"   ✅ {supervised_examples} examples with known outcomes")
        print(f"   🎯 Multiple signal types for diverse learning")
        print(f"   📈 Real trading results for validation")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_full_enhanced_dataset()