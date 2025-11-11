import sys
sys.path.append('.')

from typing import Dict
import numpy as np
from src.core.entanglement_engine import EntanglementEngine
from src.core.measurement import MeasurementSystem
from src.visualization.plotter import QuantumPlotter
from src.utils.logger import setup_logger


def run_entanglement_swapping() -> Dict:
    """Demonstrate entanglement swapping protocol."""
    logger = setup_logger('Entanglement_Swapping', 'results/entanglement_swapping.log')
    logger.info("Starting entanglement swapping experiment")
    
    # Initialize with 4 qubits
    engine = EntanglementEngine(num_qubits=4, shots=8192)
    measurement = MeasurementSystem(shots=8192)
    
    # Create two independent EPR pairs: (0,1) and (2,3)
    logger.info("Creating two independent EPR pairs")
    engine.state.initialize_bell_state(0, 1)
    engine.state.initialize_bell_state(2, 3)
    
    # Calculate initial entanglement
    initial_concurrence_01 = engine.state.calculate_concurrence(0, 1)
    initial_concurrence_23 = engine.state.calculate_concurrence(2, 3)
    initial_concurrence_03 = engine.state.calculate_concurrence(0, 3)
    
    logger.info(f"Initial entanglement:")
    logger.info(f"  C(0,1) = {initial_concurrence_01:.4f}")
    logger.info(f"  C(2,3) = {initial_concurrence_23:.4f}")
    logger.info(f"  C(0,3) = {initial_concurrence_03:.4f} (should be ~0)")
    
    # Perform entanglement swapping
    logger.info("Performing entanglement swapping via Bell measurement on qubits 1,2")
    engine.entanglement_swapping([(0, 1, 2, 3)])
    
    # Calculate final entanglement
    final_concurrence_01 = engine.state.calculate_concurrence(0, 1)
    final_concurrence_23 = engine.state.calculate_concurrence(2, 3)
    final_concurrence_03 = engine.state.calculate_concurrence(0, 3)
    
    logger.info(f"Final entanglement:")
    logger.info(f"  C(0,1) = {final_concurrence_01:.4f}")
    logger.info(f"  C(2,3) = {final_concurrence_23:.4f}")
    logger.info(f"  C(0,3) = {final_concurrence_03:.4f} (should be ~1)")
    
    # Verify entanglement transfer
    if final_concurrence_03 > 0.8:
        logger.info("SUCCESS: Entanglement successfully swapped from (0,1)(2,3) to (0,3)")
    else:
        logger.warning("WARNING: Entanglement swapping may have failed")
    
    return {
        'initial': {
            'C_01': initial_concurrence_01,
            'C_23': initial_concurrence_23,
            'C_03': initial_concurrence_03
        },
        'final': {
            'C_01': final_concurrence_01,
            'C_23': final_concurrence_23,
            'C_03': final_concurrence_03
        }
    }


if __name__ == '__main__':
    print("=" * 60)
    print("ENTANGLEMENT SWAPPING PROTOCOL")
    print("=" * 60)
    
    results = run_entanglement_swapping()
    
    print("\nResults:")
    print(f"Entanglement transferred from pairs to distant qubits")
    print(f"Final C(0,3) = {results['final']['C_03']:.4f}")