import sys
sys.path.append('.')

import unittest
import numpy as np
from src.optimization.state_controller import QuantumStateController


class TestStateController(unittest.TestCase):
    """Test quantum state control and optimization."""
    
    def test_variational_state_preparation(self):
        """Test variational quantum state preparation."""
        target_distribution = {
            '00': 0.5,
            '11': 0.5
        }
        
        controller = QuantumStateController(
            num_qubits=2,
            target_distribution=target_distribution
        )
        
        circuit, params = controller.variational_state_preparation(depth=2)
        
        self.assertIsNotNone(circuit)
        self.assertIsNotNone(params)
        self.assertGreater(len(params), 0)
        
    def test_bell_violation_optimization(self):
        """Test optimization of Bell violation."""
        controller = QuantumStateController(num_qubits=2)
        
        results = controller.optimize_bell_violation('CHSH')
        
        self.assertIn('theta_a0', results)
        self.assertIn('theta_a1', results)
        self.assertIn('theta_b0', results)
        self.assertIn('theta_b1', results)
        self.assertIn('max_violation', results)
        
        # Should achieve near-maximal violation
        self.assertGreater(results['max_violation'], 2.5)


if __name__ == '__main__':
    unittest.main()