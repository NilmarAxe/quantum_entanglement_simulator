import numpy as np
from typing import Tuple, Optional
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from math import gcd
from fractions import Fraction


class AdaptedShorAlgorithm:
    """Implements adapted Shor's algorithm for period finding and factorization."""
    
    def __init__(self, N: int):
        """Initialize for factoring number N."""
        self.N = N
        self.n_count = int(np.ceil(np.log2(N))) * 2  # Counting qubits
        self.n_work = int(np.ceil(np.log2(N)))  # Work qubits
        
    def quantum_period_finding(self, a: int) -> QuantumCircuit:
        """Find period r where a^r mod N = 1 using quantum phase estimation.
        
        Args:
            a: Base for modular exponentiation (coprime to N)
            
        Returns:
            Quantum circuit for period finding
        """
        qr_count = QuantumRegister(self.n_count, 'count')
        qr_work = QuantumRegister(self.n_work, 'work')
        cr = ClassicalRegister(self.n_count, 'c')
        circuit = QuantumCircuit(qr_count, qr_work, cr)
        
        # Initialize work register to |1⟩
        circuit.x(qr_work[0])
        
        # Apply Hadamard to counting qubits
        for i in range(self.n_count):
            circuit.h(qr_count[i])
            
        # Controlled modular exponentiation
        for i in range(self.n_count):
            exponent = 2 ** i
            self._controlled_modular_exp(circuit, qr_count[i], qr_work, 
                                        a, exponent, self.N)
            
        # Inverse QFT on counting register
        self._inverse_qft(circuit, qr_count)
        
        # Measure counting qubits
        circuit.measure(qr_count, cr)
        
        return circuit
        
    def _controlled_modular_exp(self, circuit: QuantumCircuit, 
                               control: int, work_qubits: QuantumRegister,
                               a: int, exponent: int, N: int) -> None:
        """Apply controlled U^(2^j) where U|y⟩ = |ay mod N⟩."""
        # Compute a^exponent mod N classically
        power = pow(a, exponent, N)
        
        # Implement controlled multiplication
        # This is simplified; full implementation requires modular arithmetic gates
        if power % 2 == 1:
            circuit.cx(control, work_qubits[0])
            
    def _inverse_qft(self, circuit: QuantumCircuit, 
                    qubits: QuantumRegister) -> None:
        """Apply inverse Quantum Fourier Transform."""
        n = len(qubits)
        
        for j in range(n // 2):
            circuit.swap(qubits[j], qubits[n - j - 1])
            
        for j in range(n):
            for k in range(j):
                circuit.cp(-np.pi / (2 ** (j - k)), qubits[k], qubits[j])
            circuit.h(qubits[j])
            
    def classical_post_processing(self, measured_phase: int) -> Optional[int]:
        """Extract period from measured phase.
        
        Args:
            measured_phase: Integer measured from counting register
            
        Returns:
            Period r, or None if not found
        """
        if measured_phase == 0:
            return None
            
        # Convert to fraction
        phase = measured_phase / (2 ** self.n_count)
        frac = Fraction(phase).limit_denominator(self.N)
        
        r = frac.denominator
        
        # Verify period
        if pow(2, r, self.N) == 1:
            return r
            
        return None
        
    def factor(self, a: Optional[int] = None) -> Tuple[Optional[int], Optional[int]]:
        """Attempt to factor N using quantum period finding.
        
        Args:
            a: Base for period finding (random if None)
            
        Returns:
            Tuple of factors (p, q) where N = p*q, or (None, None)
        """
        if a is None:
            # Choose random a coprime to N
            while True:
                a = np.random.randint(2, self.N)
                if gcd(a, self.N) == 1:
                    break
                    
        # Check if a is already a factor
        g = gcd(a, self.N)
        if g != 1:
            return (g, self.N // g)
            
        # Find period (would use quantum circuit here)
        # For demonstration, using classical algorithm
        r = self._classical_period_finding(a, self.N)
        
        if r is None or r % 2 != 0:
            return (None, None)
            
        # Try to extract factors
        x = pow(a, r // 2, self.N)
        
        if x == 1 or x == self.N - 1:
            return (None, None)
            
        p = gcd(x - 1, self.N)
        q = gcd(x + 1, self.N)
        
        if p * q == self.N:
            return (p, q)
            
        return (None, None)
        
    def _classical_period_finding(self, a: int, N: int) -> Optional[int]:
        """Classical period finding for small N (fallback)."""
        result = 1
        for r in range(1, N):
            result = (result * a) % N
            if result == 1:
                return r
        return None