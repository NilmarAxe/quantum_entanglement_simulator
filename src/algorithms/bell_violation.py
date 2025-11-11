import numpy as np
from typing import Dict, Tuple, List
from qiskit import QuantumCircuit, execute
from qiskit_aer import AerSimulator


class BellInequalityTest:
    """Tests for Bell inequality violations."""
    
    def __init__(self, shots: int = 8192):
        self.shots = shots
        self.simulator = AerSimulator()
        
    def chsh_inequality(self, theta_a0: float, theta_a1: float,
                       theta_b0: float, theta_b1: float) -> float:
        """Compute CHSH correlation for measurement angles.
        
        CHSH: S = |E(a0,b0) + E(a0,b1) + E(a1,b0) - E(a1,b1)| ≤ 2 (classical)
        Quantum maximum: 2√2
        
        Returns:
            CHSH value S
        """
        angles = [
            (theta_a0, theta_b0),
            (theta_a0, theta_b1),
            (theta_a1, theta_b0),
            (theta_a1, theta_b1)
        ]
        
        correlations = []
        
        for theta_a, theta_b in angles:
            E = self._measure_correlation(theta_a, theta_b)
            correlations.append(E)
            
        S = abs(correlations[0] + correlations[1] + 
                correlations[2] - correlations[3])
        
        return S
        
    def _measure_correlation(self, theta_a: float, theta_b: float) -> float:
        """Measure correlation E(a,b) for Bell pair with given angles."""
        circuit = QuantumCircuit(2, 2)
        
        # Create Bell pair
        circuit.h(0)
        circuit.cx(0, 1)
        
        # Rotate measurement bases
        circuit.ry(theta_a, 0)
        circuit.ry(theta_b, 1)
        
        # Measure
        circuit.measure([0, 1], [0, 1])
        
        # Execute
        job = execute(circuit, self.simulator, shots=self.shots)
        result = job.result()
        counts = result.get_counts()
        
        # Calculate correlation E = P(same) - P(different)
        same = counts.get('00', 0) + counts.get('11', 0)
        different = counts.get('01', 0) + counts.get('10', 0)
        
        E = (same - different) / self.shots
        
        return E
        
    def optimal_chsh_angles(self) -> Dict[str, float]:
        """Return optimal angles for maximal CHSH violation.
        
        Returns:
            Dictionary with angles that give S = 2√2
        """
        return {
            'theta_a0': 0,
            'theta_a1': np.pi / 2,
            'theta_b0': np.pi / 4,
            'theta_b1': -np.pi / 4
        }
        
    def mermin_inequality(self, num_parties: int = 3) -> float:
        """Test Mermin inequality for multi-party entanglement.
        
        Mermin-GHZ inequality for n parties:
        Classical bound: 2^(n-1)
        Quantum maximum: 2^(n/2) * √2 for even n
        
        Args:
            num_parties: Number of entangled parties
            
        Returns:
            Mermin value
        """
        circuit = QuantumCircuit(num_parties, num_parties)
        
        # Create GHZ state
        circuit.h(0)
        for i in range(1, num_parties):
            circuit.cx(0, i)
            
        # Measure in X and Y bases
        measurements = []
        
        for basis_choice in range(2 ** num_parties):
            meas_circuit = circuit.copy()
            
            for qubit in range(num_parties):
                if (basis_choice >> qubit) & 1:
                    # Y measurement
                    meas_circuit.sdg(qubit)
                    meas_circuit.h(qubit)
                else:
                    # X measurement
                    meas_circuit.h(qubit)
                    
            meas_circuit.measure(range(num_parties), range(num_parties))
            
            job = execute(meas_circuit, self.simulator, shots=self.shots)
            result = job.result()
            counts = result.get_counts()
            
            measurements.append(counts)
            
        # Compute Mermin polynomial (simplified)
        mermin_value = self._compute_mermin_value(measurements, num_parties)
        
        return mermin_value
        
    def _compute_mermin_value(self, measurements: List[Dict], 
                             num_parties: int) -> float:
        """Compute Mermin polynomial value from measurements."""
        # Simplified computation - full implementation requires
        # proper Mermin operator construction
        total_correlation = 0.0
        
        for counts in measurements:
            for bitstring, count in counts.items():
                parity = sum(int(bit) for bit in bitstring) % 2
                total_correlation += (-1) ** parity * count / self.shots
                
        return abs(total_correlation)
        
    def eberhard_inequality(self) -> Tuple[float, bool]:
        """Test Eberhard inequality without fair sampling assumption.
        
        Returns:
            (inequality_value, is_violated)
        """
        circuit = QuantumCircuit(2, 2)
        
        # Create Bell pair
        circuit.h(0)
        circuit.cx(0, 1)
        
        # Measure with different settings
        angles_alice = [0, np.pi/4]
        angles_bob = [np.pi/8, -np.pi/8]
        
        probabilities = {}
        
        for a_idx, theta_a in enumerate(angles_alice):
            for b_idx, theta_b in enumerate(angles_bob):
                test_circuit = circuit.copy()
                test_circuit.ry(theta_a, 0)
                test_circuit.ry(theta_b, 1)
                test_circuit.measure([0, 1], [0, 1])
                
                job = execute(test_circuit, self.simulator, shots=self.shots)
                result = job.result()
                counts = result.get_counts()
                
                # Store probabilities
                for outcome in ['00', '01', '10', '11']:
                    key = f'P{a_idx}{b_idx}_{outcome}'
                    probabilities[key] = counts.get(outcome, 0) / self.shots
                    
        # Compute Eberhard inequality
        E = self._compute_eberhard_value(probabilities)
        
        # Classical bound is 0, quantum can be negative
        is_violated = E < -1e-6
        
        return E, is_violated
        
    def _compute_eberhard_value(self, probs: Dict[str, float]) -> float:
        """Compute Eberhard inequality value."""
        # Simplified Eberhard computation
        value = (probs.get('P00_00', 0) + probs.get('P00_11', 0) +
                probs.get('P11_00', 0) + probs.get('P11_11', 0) -
                probs.get('P01_00', 0) - probs.get('P01_11', 0) -
                probs.get('P10_00', 0) - probs.get('P10_11', 0))
        
        return value