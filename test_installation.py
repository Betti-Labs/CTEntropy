#!/usr/bin/env python3
"""
Installation Test Script for CTEntropy

This script tests that CTEntropy can be installed and used by a new user.
Run this after cloning the repository to verify everything works.
"""

import sys
import traceback

def test_imports():
    """Test that all core modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        # Core modules
        from ctentropy_platform.core.entropy import SymbolicEntropyCalculator
        from ctentropy_platform.core.signals import SignalGenerator, ConditionType
        from ctentropy_platform.core.clinical_validator import ClinicalValidator
        print("  ✅ Core modules imported successfully")
        
        # Data loaders
        from ctentropy_platform.data.physionet_loader import PhysioNetEEGLoader
        from ctentropy_platform.data.uci_alcoholism_loader import UCIAlcoholismLoader
        from ctentropy_platform.data.clinical_loader import ClinicalEEGLoader
        print("  ✅ Data loaders imported successfully")
        
        # Security and reports
        from ctentropy_platform.security.hipaa_compliance import HIPAACompliance
        from ctentropy_platform.reports.clinical_reporter import ClinicalReporter
        print("  ✅ Security and reporting modules imported successfully")
        
        # Main clinical system
        from clinical_ctentropy_system import ClinicalCTEntropySystem
        print("  ✅ Main clinical system imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic entropy calculation functionality"""
    print("\n🧮 Testing basic functionality...")
    
    try:
        import numpy as np
        from ctentropy_platform.core.entropy import SymbolicEntropyCalculator
        from ctentropy_platform.core.signals import SignalGenerator, ConditionType
        
        # Test signal generation
        generator = SignalGenerator(random_seed=42)
        signal = generator.generate_healthy_series(length=1000)
        print(f"  ✅ Signal generation: {len(signal)} samples")
        
        # Test entropy calculation
        calculator = SymbolicEntropyCalculator(window_size=50)
        entropies = calculator.calculate(signal)
        print(f"  ✅ Entropy calculation: {len(entropies)} values, mean={np.mean(entropies):.3f}")
        
        # Test different conditions
        conditions = [ConditionType.HEALTHY, ConditionType.CTE, ConditionType.ALZHEIMERS]
        for condition in conditions:
            test_signal = generator.generate_signal(condition, length=500)
            test_entropies = calculator.calculate(test_signal)
            print(f"  ✅ {condition.value}: mean entropy = {np.mean(test_entropies):.3f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_clinical_system():
    """Test the main clinical system"""
    print("\n🏥 Testing clinical system...")
    
    try:
        import numpy as np
        from clinical_ctentropy_system import ClinicalCTEntropySystem
        
        # Initialize clinical system
        system = ClinicalCTEntropySystem(
            facility_name="Test Lab",
            physician_name="Test Doctor"
        )
        print("  ✅ Clinical system initialized")
        
        # Test with synthetic data
        test_signal = np.random.randn(2560)  # 10 seconds at 256 Hz
        
        result = system.analyze_patient_eeg(
            eeg_signal=test_signal,
            sampling_rate=256.0,
            patient_id="TEST_PATIENT_001",
            user_id="TEST_USER"
        )
        
        if result['status'] == 'success':
            print(f"  ✅ Analysis successful: {result['diagnosis']['condition']} ({result['diagnosis']['confidence']:.1f}%)")
            return True
        else:
            print(f"  ❌ Analysis failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"  ❌ Clinical system test failed: {e}")
        traceback.print_exc()
        return False

def test_dependencies():
    """Test that all required dependencies are available"""
    print("\n📦 Testing dependencies...")
    
    required_packages = [
        'numpy', 'scipy', 'scikit-learn', 'pandas', 
        'matplotlib', 'mne', 'reportlab', 'cryptography'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'scikit-learn':
                __import__('sklearn')  # scikit-learn imports as sklearn
            else:
                __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def main():
    """Run all installation tests"""
    print("🧪 CTEntropy Installation Test")
    print("=" * 50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Imports", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Clinical System", test_clinical_system)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! CTEntropy is ready to use.")
        print("\nNext steps:")
        print("1. Run 'python demo.py' to see the core engine in action")
        print("2. Run 'python test_real_detection.py' to test detection capabilities")
        print("3. Check the README.md for usage examples")
        print("4. See CONTRIBUTING.md for collaboration guidelines")
        return True
    else:
        print(f"\n⚠️  {len(results) - passed} tests failed. Please check the errors above.")
        print("\nCommon solutions:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Make sure you're in the CTEntropy directory")
        print("3. Check that Python 3.8+ is being used")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)