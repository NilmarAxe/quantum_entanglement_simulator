from dataclasses import dataclass
from typing import Dict, Any
import json


@dataclass
class Config:
    """Global configuration for quantum simulator."""
    
    # Simulation parameters
    default_shots: int = 8192
    max_qubits: int = 20
    optimization_tolerance: float = 1e-6
    
    # Algorithm parameters
    grover_max_iterations: int = 100
    variational_depth: int = 3
    learning_rate: float = 0.1
    
    # Visualization parameters
    figure_dpi: int = 300
    plot_style: str = 'seaborn'
    
    # Performance parameters
    use_cpp_acceleration: bool = True
    parallel_execution: bool = True
    num_threads: int = 4
    
    # Output parameters
    results_dir: str = 'results'
    save_plots: bool = True
    save_data: bool = True
    
    @classmethod
    def from_file(cls, filepath: str) -> 'Config':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)
        
    def to_file(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            
    def update(self, **kwargs: Any) -> None:
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")