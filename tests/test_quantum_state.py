import sys
sys.path.append('.')

import unittest
import numpy as np
from src.core.quantum_state import QuantumState


class TestQuantumState(unittest.TestCase):
    """Test suite for QuantumState class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.num_qubits = 2
        self.state = QuantumState(self.num_qubits)
        
    def test_initialization(self):
        """Test quantum state initialization."""
        self.assertEqual(self.state.num_qubits, self.num_qubits)
        self.assertIsNotNone(self.state.circuit)
        
    def test_bell_state_creation(self):
        """Test Bell state initialization."""
        self.state.initialize_bell_state(0, 1, 'phi_plus')
        statevector = self.state.compute_statevector()
        
        # |Φ+⟩ = (|00⟩ + |11⟩)/√2
        expected = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
        
        np.testing.assert_array_almost_equal(
            np.abs(statevector.data), 
            np.abs(expected), 
            decimal=5
        )
        
    def test_entanglement_entropy(self):
        """Test entanglement entropy calculation."""
        self.state.initialize_bell_state(0, 1)
        entropy = self.state.calculate_entanglement_entropy([0])
        
        # For maximally entangled state, entropy should be 1
        self.assertAlmostEqual(entropy, 1.0, places=5)
        
    def test_concurrence(self):
        """Test concurrence calculation for Bell state."""
        self.state.initialize_bell_state(0, 1)
        concurrence = self.state.calculate_concurrence(0, 1)
        
        # Bell state should have maximal concurrence = 1
        self.assertAlmostEqual(concurrence, 1.0, places=5)
        
    def test_ghz_state(self):
        """Test GHZ state creation."""
        state_3q = QuantumState(3)
        state_3q.initialize_ghz_state([0, 1, 2])
        statevector = state_3q.compute_statevector()
        
        # |GHZ⟩ = (|000⟩ + |111⟩)/√2
        expected_positions = [0, 7]  # |000⟩ and |111⟩
        
        for i, amp in enumerate(statevector.data):
            if i in expected_positions:
                self.assertAlmostEqual(np.abs(amp), 1/np.sqrt(2), places=5)
            else:
                self.assertAlmostEqual(np.abs(amp), 0, places=5)


class TestEntanglementMeasures(unittest.TestCase):
    """Test entanglement quantification methods."""
    
    def test_separable_state_entropy(self):
        """Test that separable states have zero entropy."""
        state = QuantumState(2)
        # Product state |00⟩
        entropy = state.calculate_entanglement_entropy([0])
        self.assertAlmostEqual(entropy, 0.0, places=5)
        
    def test_separable_state_concurrence(self):
        """Test that separable states have zero concurrence."""
        state = QuantumState(2)
        concurrence = state.calculate_concurrence(0, 1)
        self.assertAlmostEqual(concurrence, 0.0, places=5)


if __name__ == '__main__':
    unittest.main()