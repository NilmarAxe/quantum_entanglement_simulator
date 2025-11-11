import numpy as np
from typing import List, Tuple, Optional
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace


class QuantumState:
    """Represents and manipulates quantum states with entanglement tracking."""
    
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.qr = QuantumRegister(num_qubits, 'q')
        self.cr = ClassicalRegister(num_qubits, 'c')
        self.circuit = QuantumCircuit(self.qr, self.cr)
        self._statevector: Optional[Statevector] = None
        self._density_matrix: Optional[DensityMatrix] = None
        
    def initialize_bell_state(self, qubit_a: int, qubit_b: int, 
                              bell_type: str = 'phi_plus') -> None:
        """Initialize Bell state pairs for entanglement studies.
        
        Bell states:
        |Φ+⟩ = (|00⟩ + |11⟩)/√2
        |Φ-⟩ = (|00⟩ - |11⟩)/√2
        |Ψ+⟩ = (|01⟩ + |10⟩)/√2
        |Ψ-⟩ = (|01⟩ - |10⟩)/√2
        
        Args:
            qubit_a: First qubit index
            qubit_b: Second qubit index
            bell_type: Type of Bell state ('phi_plus', 'phi_minus', 'psi_plus', 'psi_minus')
            
        Raises:
            ValueError: If qubit indices are invalid or bell_type is unknown
        """
        if qubit_a >= self.num_qubits or qubit_b >= self.num_qubits:
            raise ValueError(f"Qubit indices must be < {self.num_qubits}")
        
        if qubit_a == qubit_b:
            raise ValueError("Qubit indices must be different")
        
        valid_types = ['phi_plus', 'phi_minus', 'psi_plus', 'psi_minus']
        if bell_type not in valid_types:
            raise ValueError(f"bell_type must be one of {valid_types}")
        
        self.circuit.h(qubit_a)
        self.circuit.cx(qubit_a, qubit_b)
        
        if bell_type == 'phi_minus':
            self.circuit.z(qubit_b)
        elif bell_type == 'psi_plus':
            self.circuit.x(qubit_b)
        elif bell_type == 'psi_minus':
            self.circuit.x(qubit_b)
            self.circuit.z(qubit_b)
            
    def initialize_ghz_state(self, qubits: List[int]) -> None:
        """Initialize GHZ state: (|000...⟩ + |111...⟩)/√2
        
        Args:
            qubits: List of qubit indices to entangle
            
        Raises:
            ValueError: If fewer than 2 qubits provided or indices invalid
        """
        if len(qubits) < 2:
            raise ValueError("GHZ state requires at least 2 qubits")
        
        if any(q >= self.num_qubits or q < 0 for q in qubits):
            raise ValueError(f"All qubit indices must be in range [0, {self.num_qubits})")
        
        if len(set(qubits)) != len(qubits):
            raise ValueError("Qubit indices must be unique")
            
        self.circuit.h(qubits[0])
        for i in range(1, len(qubits)):
            self.circuit.cx(qubits[0], qubits[i])
            
    def apply_controlled_rotation(self, control: int, target: int, 
                                  theta: float, phi: float, lambda_: float) -> None:
        """Apply controlled arbitrary rotation."""
        self.circuit.cu(theta, phi, lambda_, 0, control, target)
        
    def apply_entangling_gate(self, qubit_a: int, qubit_b: int, 
                             gate_type: str = 'cnot') -> None:
        """Apply two-qubit entangling gates."""
        if gate_type == 'cnot':
            self.circuit.cx(qubit_a, qubit_b)
        elif gate_type == 'cz':
            self.circuit.cz(qubit_a, qubit_b)
        elif gate_type == 'swap':
            self.circuit.swap(qubit_a, qubit_b)
        elif gate_type == 'iswap':
            self.circuit.iswap(qubit_a, qubit_b)
            
    def compute_statevector(self) -> Statevector:
        """Compute current statevector."""
        self._statevector = Statevector.from_instruction(self.circuit)
        return self._statevector
        
    def compute_density_matrix(self) -> DensityMatrix:
        """Compute density matrix representation."""
        if self._statevector is None:
            self.compute_statevector()
        self._density_matrix = DensityMatrix(self._statevector)
        return self._density_matrix
        
    def calculate_entanglement_entropy(self, partition: List[int]) -> float:
        """Calculate von Neumann entropy for subsystem.
        
        S(ρ) = -Tr(ρ log₂ ρ)
        """
        rho = self.compute_density_matrix()
        
        # Get qubits to trace out
        trace_qubits = [i for i in range(self.num_qubits) if i not in partition]
        
        if not trace_qubits:
            return 0.0
            
        rho_reduced = partial_trace(rho, trace_qubits)
        eigenvalues = np.linalg.eigvalsh(rho_reduced.data)
        
        # Remove near-zero eigenvalues for numerical stability
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        
        entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
        return float(entropy)
        
    def calculate_concurrence(self, qubit_a: int, qubit_b: int) -> float:
        """Calculate concurrence for two-qubit entanglement measure.
        
        C = max(0, λ₁ - λ₂ - λ₃ - λ₄)
        where λᵢ are eigenvalues of ρ(σ_y ⊗ σ_y)ρ*(σ_y ⊗ σ_y)
        """
        rho = self.compute_density_matrix()
        
        # Get reduced density matrix for the two qubits
        trace_qubits = [i for i in range(self.num_qubits) 
                       if i not in [qubit_a, qubit_b]]
        
        if trace_qubits:
            rho_ab = partial_trace(rho, trace_qubits)
        else:
            rho_ab = rho
            
        rho_matrix = rho_ab.data
        
        # Pauli Y operator
        sigma_y = np.array([[0, -1j], [1j, 0]])
        sigma_yy = np.kron(sigma_y, sigma_y)
        
        # Compute R = ρ(σ_y ⊗ σ_y)ρ*(σ_y ⊗ σ_y)
        R = rho_matrix @ sigma_yy @ np.conj(rho_matrix) @ sigma_yy
        
        eigenvalues = np.linalg.eigvalsh(R)
        eigenvalues = np.sqrt(np.maximum(eigenvalues, 0))
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        concurrence = max(0, eigenvalues[0] - eigenvalues[1] 
                         - eigenvalues[2] - eigenvalues[3])
        
        return float(concurrence)
        
    def get_circuit(self) -> QuantumCircuit:
        """Return the quantum circuit."""
        return self.circuit
        
    def reset_circuit(self) -> None:
        """Reset circuit to initial state."""
        self.circuit = QuantumCircuit(self.qr, self.cr)
        self._statevector = None
        self._density_matrix = None