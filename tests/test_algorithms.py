import sys
sys.path.append('.')

import unittest
import numpy as np
from src.algorithms.grover import GroverAlgorithm
from src.algorithms.bell_violation import BellInequalityTest
from qiskit import execute
from qiskit_aer import AerSimulator


class TestGroverAlgorithm(unittest.TestCase):
    """Test Grover's search algorithm."""
    
    def test_grover_single_solution(self):
        """Test Grover's algorithm with single marked state."""
        num_qubits = 3
        grover = GroverAlgorithm(num_qubits)
        marked_state = [5]  # Search for |101⟩
        
        circuit = grover.run(marked_state)
        
        # Execute circuit
        simulator = AerSimulator()
        job = execute(circuit, simulator, shots=1024)
        result = job.result()
        counts = result.get_counts()
        
        # Most common outcome should be the marked state
        most_common = max(counts, key=counts.get)
        self.assertEqual(int(most_common, 2), marked_state[0])
        
    def test_grover_multiple_solutions(self):
        """Test Grover with multiple marked states."""
        num_qubits = 2
        grover = GroverAlgorithm(num_qubits)
        marked_states = [1, 3]  # Search for |01⟩ and |11⟩
        
        circuit = grover.run(marked_states)
        
        simulator = AerSimulator()
        job = execute(circuit, simulator, shots=1024)
        result = job.result()
        counts = result.get_counts()
        
        # Check that marked states have high probability
        total_marked = sum(counts.get(format(s, '02b'), 0) for s in marked_states)
        self.assertGreater(total_marked / 1024, 0.5)


class TestBellInequality(unittest.TestCase):
    """Test Bell inequality violations."""
    
    def test_chsh_violation(self):
        """Test that CHSH inequality is violated."""
        bell_test = BellInequalityTest(shots=4096)
        
        # Use optimal angles
        optimal_angles = bell_test.optimal_chsh_angles()
        S = bell_test.chsh_inequality(
            optimal_angles['theta_a0'],
            optimal_angles['theta_a1'],
            optimal_angles['theta_b0'],
            optimal_angles['theta_b1']
        )
        
        # S should exceed classical bound of 2
        self.assertGreater(S, 2.0)
        
        # S should be close to quantum maximum 2√2
        self.assertLess(abs(S - 2*np.sqrt(2)), 0.2)
        
    def test_chsh_classical_bound(self):
        """Test that certain angles give classical correlations."""
        bell_test = BellInequalityTest(shots=4096)
        
        # Aligned measurements (should give classical-like correlations)
        S = bell_test.chsh_inequality(0, 0, 0, 0)
        
        # Should be close to or below classical bound
        self.assertLessEqual(S, 2.5)


if __name__ == '__main__':
    unittest.main()