import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('.')

from src.core.quantum_state import QuantumState
from src.core.entanglement_engine import EntanglementEngine
from src.core.measurement import MeasurementSystem
from src.algorithms.grover import GroverAlgorithm
from src.algorithms.bell_violation import BellInequalityTest
from src.optimization.state_controller import QuantumStateController
from src.visualization.plotter import QuantumPlotter
from src.visualization.bloch_sphere import BlochSphereVisualizer
from src.utils.config import Config
from src.utils.logger import setup_logger

import numpy as np
from typing import Dict
import json


def setup_environment():
    """Set up necessary directories and configuration."""
    # Create results directory
    Path('results').mkdir(exist_ok=True)
    Path('results/plots').mkdir(exist_ok=True)
    Path('results/data').mkdir(exist_ok=True)
    
    # Load or create configuration
    config = Config()
    config.to_file('results/config.json')
    
    return config


def run_complete_simulation(config: Config) -> Dict:
    """Run complete quantum entanglement simulation suite."""
    logger = setup_logger('MainSimulation', 'results/main_simulation.log')
    logger.info("=" * 60)
    logger.info("QUANTUM ENTANGLEMENT SIMULATOR - MAIN EXECUTION")
    logger.info("=" * 60)
    
    results = {}
    
    # 1. Basic Entanglement Tests
    logger.info("\n[1/6] Testing basic entanglement properties...")
    results['entanglement'] = test_entanglement_properties(logger)
    
    # 2. Bell Inequality Violations
    logger.info("\n[2/6] Testing Bell inequality violations...")
    results['bell_violations'] = test_bell_violations(logger)
    
    # 3. Quantum Algorithms
    logger.info("\n[3/6] Running quantum algorithms...")
    results['algorithms'] = test_quantum_algorithms(logger)
    
    # 4. State Control and Optimization
    logger.info("\n[4/6] Testing state control and optimization...")
    results['optimization'] = test_state_optimization(logger)
    
    # 5. Advanced Protocols
    logger.info("\n[5/6] Testing advanced quantum protocols...")
    results['protocols'] = test_quantum_protocols(logger)
    
    # 6. Performance Analysis
    logger.info("\n[6/6] Analyzing system performance...")
    results['performance'] = analyze_performance(logger)
    
    # Save results
    with open('results/data/simulation_results.json', 'w') as f:
        json.dump(results, f, indent=4, default=str)
    
    logger.info("\n" + "=" * 60)
    logger.info("SIMULATION COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info(f"Results saved to: results/data/simulation_results.json")
    logger.info(f"Plots saved to: results/plots/")
    
    return results


def test_entanglement_properties(logger) -> Dict:
    """Test fundamental entanglement properties."""
    logger.info("Creating Bell states and measuring entanglement...")
    
    engine = EntanglementEngine(num_qubits=3, shots=8192)
    plotter = QuantumPlotter()
    
    # Test different Bell states
    bell_states = ['phi_plus', 'phi_minus', 'psi_plus', 'psi_minus']
    results = {}
    
    for bell_type in bell_states:
        engine.state.reset_circuit()
        engine.state.initialize_bell_state(0, 1, bell_type)
        
        entropy = engine.state.calculate_entanglement_entropy([0])
        concurrence = engine.state.calculate_concurrence(0, 1)
        
        results[bell_type] = {
            'entropy': float(entropy),
            'concurrence': float(concurrence)
        }
        
        logger.info(f"  {bell_type}: Entropy={entropy:.4f}, Concurrence={concurrence:.4f}")
    
    # Create GHZ state
    engine.state.reset_circuit()
    engine.state.initialize_ghz_state([0, 1, 2])
    
    ghz_entropy = engine.state.calculate_entanglement_entropy([0])
    results['ghz'] = {'entropy': float(ghz_entropy)}
    
    logger.info(f"  GHZ state: Entropy={ghz_entropy:.4f}")
    
    return results


def test_bell_violations(logger) -> Dict:
    """Test various Bell inequality violations."""
    logger.info("Testing CHSH and other Bell inequalities...")
    
    bell_test = BellInequalityTest(shots=8192)
    plotter = QuantumPlotter()
    
    # CHSH test
    optimal_angles = bell_test.optimal_chsh_angles()
    S = bell_test.chsh_inequality(
        optimal_angles['theta_a0'],
        optimal_angles['theta_a1'],
        optimal_angles['theta_b0'],
        optimal_angles['theta_b1']
    )
    
    logger.info(f"  CHSH value: S = {S:.4f}")
    logger.info(f"  Classical bound: 2.0000")
    logger.info(f"  Quantum maximum: {2*np.sqrt(2):.4f}")
    
    violation = (S > 2.0)
    logger.info(f"  Bell inequality violated: {violation}")
    
    # Mermin test
    mermin_value = bell_test.mermin_inequality(num_parties=3)
    logger.info(f"  Mermin value: M = {mermin_value:.4f}")
    
    results = {
        'chsh_value': float(S),
        'chsh_violated': violation,
        'mermin_value': float(mermin_value),
        'quantum_supremacy_demonstrated': violation
    }
    
    # Plot
    chsh_data = {'Optimal': S, 'Classical Bound': 2.0, 
                'Quantum Bound': 2*np.sqrt(2)}
    plotter.plot_bell_violation(chsh_data, 
                               save_path='results/plots/bell_violation.png')
    
    return results


def test_quantum_algorithms(logger) -> Dict:
    """Test quantum algorithm implementations."""
    logger.info("Running Grover's search algorithm...")
    
    # Grover's algorithm
    num_qubits = 3
    grover = GroverAlgorithm(num_qubits)
    marked_states = [5, 7]  # Search for |101⟩ and |111⟩
    
    circuit = grover.run(marked_states)
    
    logger.info(f"  Search space: {2**num_qubits} states")
    logger.info(f"  Marked states: {marked_states}")
    logger.info(f"  Grover iterations: {int(np.pi/4 * np.sqrt(2**num_qubits / len(marked_states)))}")
    
    # Measure results
    measurement = MeasurementSystem(shots=8192)
    counts = measurement.measure_all(circuit)
    
    success_count = sum(counts.get(format(s, f'0{num_qubits}b'), 0) 
                       for s in marked_states)
    success_rate = success_count / 8192
    
    logger.info(f"  Success rate: {success_rate*100:.2f}%")
    
    # Plot
    plotter = QuantumPlotter()
    plotter.plot_measurement_histogram(counts,
                                      title="Grover's Algorithm Results",
                                      save_path='results/plots/grover_results.png')
    
    return {
        'success_rate': float(success_rate),
        'marked_states': marked_states,
        'measurements': {k: int(v) for k, v in counts.items()}
    }


def test_state_optimization(logger) -> Dict:
    """Test quantum state control and optimization."""
    logger.info("Optimizing quantum states for target distributions...")
    
    controller = QuantumStateController(
        num_qubits=2,
        target_distribution={'00': 0.5, '11': 0.5}
    )
    
    circuit, params = controller.variational_state_preparation(depth=3)
    
    logger.info(f"  Variational circuit depth: 3")
    logger.info(f"  Optimized parameters: {len(params)} angles")
    
    # Optimize Bell violation
    logger.info("Optimizing for maximum Bell violation...")
    bell_results = controller.optimize_bell_violation('CHSH')
    
    logger.info(f"  Maximum CHSH value achieved: {bell_results['max_violation']:.4f}")
    
    return {
        'variational_params': len(params),
        'max_bell_violation': float(bell_results['max_violation']),
        'optimal_angles': {k: float(v) for k, v in bell_results.items() if 'theta' in k}
    }


def test_quantum_protocols(logger) -> Dict:
    """Test advanced quantum information protocols."""
    logger.info("Testing quantum teleportation and entanglement swapping...")
    
    engine = EntanglementEngine(num_qubits=4, shots=8192)
    
    # Entanglement swapping
    engine.state.reset_circuit()
    engine.state.initialize_bell_state(0, 1)
    engine.state.initialize_bell_state(2, 3)
    
    initial_c03 = engine.state.calculate_concurrence(0, 3)
    
    engine.entanglement_swapping([(0, 1, 2, 3)])
    
    final_c03 = engine.state.calculate_concurrence(0, 3)
    
    logger.info(f"  Entanglement swapping:")
    logger.info(f"    Initial C(0,3): {initial_c03:.4f}")
    logger.info(f"    Final C(0,3): {final_c03:.4f}")
    
    swap_success = (final_c03 > 0.7)
    logger.info(f"    Success: {swap_success}")
    
    return {
        'entanglement_swapping': {
            'initial_concurrence': float(initial_c03),
            'final_concurrence': float(final_c03),
            'success': swap_success
        }
    }


def analyze_performance(logger) -> Dict:
    """Analyze computational performance."""
    import time
    
    logger.info("Running performance benchmarks...")
    
    # Benchmark state creation
    start = time.time()
    for _ in range(100):
        state = QuantumState(4)
        state.initialize_bell_state(0, 1)
    bell_time = time.time() - start
    
    logger.info(f"  Bell state creation (100x): {bell_time:.3f}s")
    
    # Benchmark entanglement calculation
    state = QuantumState(4)
    state.initialize_ghz_state([0, 1, 2, 3])
    
    start = time.time()
    for _ in range(10):
        entropy = state.calculate_entanglement_entropy([0, 1])
    entropy_time = time.time() - start
    
    logger.info(f"  Entropy calculation (10x): {entropy_time:.3f}s")
    
    return {
        'bell_state_creation_time': float(bell_time),
        'entropy_calculation_time': float(entropy_time)
    }


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print(" " * 15 + "QUANTUM ENTANGLEMENT SIMULATOR")
    print(" " * 10 + "Inspired by Scientific Curiosity and Precision")
    print("=" * 70 + "\n")
    
    # Setup
    config = setup_environment()
    
    # Run simulation
    results = run_complete_simulation(config)
    
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    print(f"\nKey Results:")
    print(f"  - Bell Violation: {results['bell_violations']['chsh_violated']}")
    print(f"  - CHSH Value: {results['bell_violations']['chsh_value']:.4f}")
    print(f"  - Grover Success Rate: {results['algorithms']['success_rate']*100:.1f}%")
    print(f"  - Max Optimized Violation: {results['optimization']['max_bell_violation']:.4f}")
    print(f"\nFull results in: results/data/simulation_results.json")
    print(f"Visualizations in: results/plots/\n")