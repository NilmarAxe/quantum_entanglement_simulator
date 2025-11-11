import sys
sys.path.append('.')

from src.algorithms.bell_violation import BellInequalityTest
from src.optimization.state_controller import QuantumStateController
from src.visualization.plotter import QuantumPlotter
from src.utils.logger import setup_logger
import numpy as np
from typing import Dict


def run_chsh_experiment() -> Dict:
    """Run CHSH inequality violation experiment."""
    logger = setup_logger('CHSH_Experiment', 'results/chsh_experiment.log')
    logger.info("Starting CHSH inequality violation experiment")
    
    bell_test = BellInequalityTest(shots=8192)
    plotter = QuantumPlotter()
    
    # Test optimal angles
    logger.info("Testing optimal CHSH angles")
    optimal_angles = bell_test.optimal_chsh_angles()
    
    S_optimal = bell_test.chsh_inequality(
        optimal_angles['theta_a0'],
        optimal_angles['theta_a1'],
        optimal_angles['theta_b0'],
        optimal_angles['theta_b1']
    )
    
    logger.info(f"Optimal CHSH value: S = {S_optimal:.4f}")
    logger.info(f"Quantum maximum: S = {2*np.sqrt(2):.4f}")
    logger.info(f"Classical bound: S = 2.0000")
    
    violation_percentage = (S_optimal - 2) / (2*np.sqrt(2) - 2) * 100
    logger.info(f"Violation: {violation_percentage:.2f}% of maximum possible")
    
    # Test multiple angle combinations
    logger.info("Testing various angle combinations")
    chsh_values = {}
    
    for i, theta_a0 in enumerate(np.linspace(0, np.pi, 5)):
        for j, theta_b0 in enumerate(np.linspace(0, np.pi, 5)):
            S = bell_test.chsh_inequality(
                theta_a0, np.pi/2, theta_b0, -theta_b0
            )
            chsh_values[f'Config_{i}_{j}'] = S
            
    # Plot results
    plotter.plot_bell_violation(chsh_values, save_path='results/chsh_violation.png')
    
    logger.info("CHSH experiment completed successfully")
    
    return {
        'optimal_S': S_optimal,
        'classical_bound': 2.0,
        'quantum_bound': 2*np.sqrt(2),
        'violation_percentage': violation_percentage,
        'all_values': chsh_values
    }


def run_optimization_experiment() -> Dict:
    """Run optimization to find maximal Bell violation."""
    logger = setup_logger('Optimization_Experiment', 'results/optimization_experiment.log')
    logger.info("Starting Bell inequality optimization")
    
    controller = QuantumStateController(num_qubits=2)
    
    logger.info("Optimizing CHSH violation...")
    optimal_results = controller.optimize_bell_violation('CHSH')
    
    logger.info(f"Optimized angles:")
    logger.info(f"  theta_a0: {optimal_results['theta_a0']:.4f} rad")
    logger.info(f"  theta_a1: {optimal_results['theta_a1']:.4f} rad")
    logger.info(f"  theta_b0: {optimal_results['theta_b0']:.4f} rad")
    logger.info(f"  theta_b1: {optimal_results['theta_b1']:.4f} rad")
    logger.info(f"Maximum violation: S = {optimal_results['max_violation']:.4f}")
    
    return optimal_results


def run_mermin_experiment() -> Dict:
    """Run Mermin inequality test for 3-party entanglement."""
    logger = setup_logger('Mermin_Experiment', 'results/mermin_experiment.log')
    logger.info("Starting Mermin inequality experiment")
    
    bell_test = BellInequalityTest(shots=8192)
    
    mermin_value = bell_test.mermin_inequality(num_parties=3)
    
    classical_bound = 2
    quantum_bound = 4.0
    
    logger.info(f"Mermin value: M = {mermin_value:.4f}")
    logger.info(f"Classical bound: M = {classical_bound}")
    logger.info(f"Quantum bound: M = {quantum_bound}")
    
    if mermin_value > classical_bound:
        logger.info("Mermin inequality VIOLATED - Quantum correlations confirmed")
    else:
        logger.info("No Mermin violation detected")
    
    return {
        'mermin_value': mermin_value,
        'classical_bound': classical_bound,
        'quantum_bound': quantum_bound
    }


if __name__ == '__main__':
    print("=" * 60)
    print("BELL INEQUALITY VIOLATION EXPERIMENTS")
    print("=" * 60)
    
    # Run experiments
    chsh_results = run_chsh_experiment()
    optimization_results = run_optimization_experiment()
    mermin_results = run_mermin_experiment()
    
    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENTAL SUMMARY")
    print("=" * 60)
    print(f"\nCHSH Violation: {chsh_results['violation_percentage']:.2f}%")
    print(f"Optimized S value: {optimization_results['max_violation']:.4f}")
    print(f"Mermin value: {mermin_results['mermin_value']:.4f}")
    print("\nAll results saved to results/")