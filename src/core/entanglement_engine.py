import numpy as np
from typing import List, Tuple, Dict
from qiskit import QuantumCircuit, execute
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from .quantum_state import QuantumState


class EntanglementEngine:
    """Manages entanglement operations and measurements."""
    
    def __init__(self, num_qubits: int, shots: int = 8192):
        self.num_qubits = num_qubits
        self.shots = shots
        self.simulator = AerSimulator()
        self.state = QuantumState(num_qubits)
        
    def create_epr_pairs(self, pairs: List[Tuple[int, int]]) -> None:
        """Create multiple EPR pairs simultaneously."""
        for qubit_a, qubit_b in pairs:
            self.state.initialize_bell_state(qubit_a, qubit_b)
            
    def entanglement_swapping(self, pairs: List[Tuple[int, int, int, int]]) -> None:
        """Perform entanglement swapping between pairs.
        
        Args:
            pairs: List of (q1, q2, q3, q4) where (q1,q2) and (q3,q4) are entangled,
                   and we swap to entangle (q1,q4).
        """
        for q1, q2, q3, q4 in pairs:
            # Initial entanglement
            self.state.initialize_bell_state(q1, q2)
            self.state.initialize_bell_state(q3, q4)
            
            # Bell measurement on q2, q3
            self.state.circuit.cx(q2, q3)
            self.state.circuit.h(q2)
            
    def quantum_teleportation(self, source: int, ancilla1: int, 
                             ancilla2: int, target: int) -> None:
        """Implement quantum teleportation protocol.
        
        Teleport state of 'source' qubit to 'target' qubit using
        entangled ancilla pair.
        """
        # Prepare Bell pair between ancillas
        self.state.initialize_bell_state(ancilla1, ancilla2)
        
        # Bell measurement on source and ancilla1
        self.state.circuit.cx(source, ancilla1)
        self.state.circuit.h(source)
        
        # Conditional operations on target based on measurements
        self.state.circuit.cx(ancilla1, target)
        self.state.circuit.cz(source, target)
        
    def generate_cluster_state(self, graph: Dict[int, List[int]]) -> None:
        """Generate cluster state based on graph connectivity.
        
        Args:
            graph: Dictionary mapping qubit indices to their neighbors
        """
        # Initialize all qubits in |+⟩ state
        for qubit in graph.keys():
            self.state.circuit.h(qubit)
            
        # Apply CZ gates according to graph edges
        for qubit, neighbors in graph.items():
            for neighbor in neighbors:
                if qubit < neighbor:  # Avoid double application
                    self.state.circuit.cz(qubit, neighbor)
                    
    def measure_bell_basis(self, qubit_a: int, qubit_b: int) -> Tuple[int, int]:
        """Measure two qubits in Bell basis."""
        circuit = self.state.circuit.copy()
        
        # Transform to Bell basis
        circuit.cx(qubit_a, qubit_b)
        circuit.h(qubit_a)
        
        # Measure
        circuit.measure(qubit_a, qubit_a)
        circuit.measure(qubit_b, qubit_b)
        
        job = execute(circuit, self.simulator, shots=1)
        result = job.result()
        counts = result.get_counts()
        
        # Extract measurement outcomes
        outcome = list(counts.keys())[0]
        return int(outcome[self.num_qubits - qubit_b - 1]), \
               int(outcome[self.num_qubits - qubit_a - 1])
               
    def calculate_mutual_information(self, subsystem_a: List[int], 
                                    subsystem_b: List[int]) -> float:
        """Calculate mutual information I(A:B) = S(A) + S(B) - S(AB)."""
        s_a = self.state.calculate_entanglement_entropy(subsystem_a)
        s_b = self.state.calculate_entanglement_entropy(subsystem_b)
        s_ab = self.state.calculate_entanglement_entropy(subsystem_a + subsystem_b)
        
        return s_a + s_b - s_ab
        
    def entanglement_witness(self, observable_matrix: np.ndarray) -> float:
        """Compute entanglement witness expectation value.
        
        A witness W satisfies: Tr(Wρ) < 0 for entangled states.
        """
        rho = self.state.compute_density_matrix()
        expectation = np.trace(observable_matrix @ rho.data)
        return float(np.real(expectation))
        
    def get_entanglement_spectrum(self) -> np.ndarray:
        """Return spectrum of reduced density matrix."""
        rho = self.state.compute_density_matrix()
        eigenvalues = np.linalg.eigvalsh(rho.data)
        return np.sort(eigenvalues)[::-1]