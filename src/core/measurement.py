import numpy as np
from typing import Dict, List, Tuple
from qiskit import execute
from qiskit_aer import AerSimulator
from scipy.stats import chi2


class MeasurementSystem:
    """Handles quantum measurements and statistical analysis."""
    
    def __init__(self, shots: int = 8192):
        self.shots = shots
        self.simulator = AerSimulator()
        self.measurement_history: List[Dict] = []
        
    def measure_all(self, circuit) -> Dict[str, int]:
        """Perform computational basis measurement on all qubits."""
        job = execute(circuit, self.simulator, shots=self.shots)
        result = job.result()
        counts = result.get_counts()
        return counts
        
    def measure_pauli_expectation(self, circuit, pauli_string: str) -> float:
        """Measure expectation value of Pauli operator.
        
        Args:
            pauli_string: String like 'XYZ' for σ_x ⊗ σ_y ⊗ σ_z
        """
        # Create measurement circuit
        meas_circuit = circuit.copy()
        
        for idx, pauli in enumerate(pauli_string):
            if pauli == 'X':
                meas_circuit.h(idx)
            elif pauli == 'Y':
                meas_circuit.sdg(idx)
                meas_circuit.h(idx)
            # Z basis is computational basis (no transformation needed)
            
        # Measure
        for idx in range(len(pauli_string)):
            meas_circuit.measure(idx, idx)
            
        counts = self.measure_all(meas_circuit)
        
        # Calculate expectation value
        expectation = 0.0
        for bitstring, count in counts.items():
            parity = sum(int(bit) for bit in bitstring) % 2
            sign = 1 if parity == 0 else -1
            expectation += sign * count / self.shots
            
        return expectation
        
    def tomography_reconstruction(self, circuit, num_qubits: int) -> np.ndarray:
        """Perform quantum state tomography.
        
        Returns reconstructed density matrix.
        """
        # Pauli basis measurements
        paulis = ['I', 'X', 'Y', 'Z']
        pauli_measurements = {}
        
        # Generate all Pauli strings
        for basis in self._generate_pauli_basis(num_qubits):
            expectation = self.measure_pauli_expectation(circuit, basis)
            pauli_measurements[basis] = expectation
            
        # Reconstruct density matrix using linear inversion
        dim = 2 ** num_qubits
        rho = np.zeros((dim, dim), dtype=complex)
        
        for pauli_string, expectation in pauli_measurements.items():
            pauli_matrix = self._pauli_string_to_matrix(pauli_string)
            rho += expectation * pauli_matrix
            
        rho /= dim
        
        # Make hermitian
        rho = (rho + rho.conj().T) / 2
        
        return rho
        
    def _generate_pauli_basis(self, num_qubits: int) -> List[str]:
        """Generate all Pauli strings for tomography."""
        if num_qubits == 1:
            return ['I', 'X', 'Y', 'Z']
            
        basis = []
        sub_basis = self._generate_pauli_basis(num_qubits - 1)
        
        for pauli in ['I', 'X', 'Y', 'Z']:
            for sub_string in sub_basis:
                basis.append(pauli + sub_string)
                
        return basis
        
    def _pauli_string_to_matrix(self, pauli_string: str) -> np.ndarray:
        """Convert Pauli string to matrix representation."""
        pauli_dict = {
            'I': np.array([[1, 0], [0, 1]]),
            'X': np.array([[0, 1], [1, 0]]),
            'Y': np.array([[0, -1j], [1j, 0]]),
            'Z': np.array([[1, 0], [0, -1]])
        }
        
        matrix = pauli_dict[pauli_string[0]]
        for pauli in pauli_string[1:]:
            matrix = np.kron(matrix, pauli_dict[pauli])
            
        return matrix
        
    def statistical_significance_test(self, observed: Dict[str, int], 
                                      expected: Dict[str, float]) -> Tuple[float, float]:
        """Perform chi-squared test for measurement outcomes.
        
        Returns: (chi-squared statistic, p-value)
        """
        chi_sq = 0.0
        degrees_of_freedom = len(observed) - 1
        
        for bitstring in observed.keys():
            obs = observed[bitstring]
            exp = expected.get(bitstring, 0) * self.shots
            
            if exp > 0:
                chi_sq += (obs - exp) ** 2 / exp
                
        p_value = 1 - chi2.cdf(chi_sq, degrees_of_freedom)
        
        return chi_sq, p_value
        
    def estimate_fidelity(self, rho_target: np.ndarray, 
                         rho_measured: np.ndarray) -> float:
        """Estimate fidelity F = Tr(√(√ρ₁ρ₂√ρ₁))²"""
        sqrt_rho1 = self._matrix_sqrt(rho_target)
        temp = sqrt_rho1 @ rho_measured @ sqrt_rho1
        sqrt_temp = self._matrix_sqrt(temp)
        
        fidelity = np.trace(sqrt_temp).real ** 2
        return float(fidelity)
        
    def _matrix_sqrt(self, matrix: np.ndarray) -> np.ndarray:
        """Compute matrix square root."""
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues = np.maximum(eigenvalues, 0)  # Ensure non-negative
        return eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.conj().T