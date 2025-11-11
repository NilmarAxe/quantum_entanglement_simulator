import numpy as np
from typing import List, Callable
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


class GroverAlgorithm:
    """Implements Grover's quantum search algorithm."""
    
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.qr = QuantumRegister(num_qubits, 'q')
        self.cr = ClassicalRegister(num_qubits, 'c')
        self.circuit = QuantumCircuit(self.qr, self.cr)
        
    def oracle(self, marked_states: List[int]) -> None:
        """Create oracle that marks target states."""
        for state in marked_states:
            # Convert state to binary
            binary = format(state, f'0{self.num_qubits}b')
            
            # Apply X gates to qubits that should be 0
            for i, bit in enumerate(binary):
                if bit == '0':
                    self.circuit.x(i)
                    
            # Multi-controlled Z gate
            if self.num_qubits == 2:
                self.circuit.cz(0, 1)
            else:
                self.circuit.h(self.num_qubits - 1)
                self.circuit.mcx(list(range(self.num_qubits - 1)), 
                                self.num_qubits - 1)
                self.circuit.h(self.num_qubits - 1)
                
            # Undo X gates
            for i, bit in enumerate(binary):
                if bit == '0':
                    self.circuit.x(i)
                    
    def diffusion_operator(self) -> None:
        """Apply Grover diffusion operator (inversion about average)."""
        # Apply H gates
        for i in range(self.num_qubits):
            self.circuit.h(i)
            
        # Apply X gates
        for i in range(self.num_qubits):
            self.circuit.x(i)
            
        # Multi-controlled Z
        self.circuit.h(self.num_qubits - 1)
        if self.num_qubits == 2:
            self.circuit.cx(0, 1)
        else:
            self.circuit.mcx(list(range(self.num_qubits - 1)), 
                            self.num_qubits - 1)
        self.circuit.h(self.num_qubits - 1)
        
        # Apply X gates
        for i in range(self.num_qubits):
            self.circuit.x(i)
            
        # Apply H gates
        for i in range(self.num_qubits):
            self.circuit.h(i)
            
    def run(self, marked_states: List[int]) -> QuantumCircuit:
        """Execute Grover's algorithm.
        
        Args:
            marked_states: List of states to search for
            
        Returns:
            Quantum circuit implementing Grover's algorithm
        """
        # Initialize in superposition
        for i in range(self.num_qubits):
            self.circuit.h(i)
            
        # Calculate optimal number of iterations
        N = 2 ** self.num_qubits
        M = len(marked_states)
        iterations = int(np.pi / 4 * np.sqrt(N / M))
        
        # Apply Grover iterations
        for _ in range(iterations):
            self.oracle(marked_states)
            self.diffusion_operator()
            
        # Measure
        self.circuit.measure(self.qr, self.cr)
        
        return self.circuit
        
    def adaptive_grover(self, oracle_func: Callable, confidence: float = 0.95) -> QuantumCircuit:
        """Adaptive Grover search with unknown number of solutions."""
        # Initialize
        for i in range(self.num_qubits):
            self.circuit.h(i)
            
        m = 1
        lambda_param = 6/5
        
        # Adaptive iteration
        while m <= self.num_qubits:
            j = np.random.randint(0, m)
            iterations = int(lambda_param ** j)
            
            for _ in range(iterations):
                oracle_func(self.circuit)
                self.diffusion_operator()
                
            m += 1
            
        self.circuit.measure(self.qr, self.cr)
        return self.circuit