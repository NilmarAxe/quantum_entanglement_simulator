import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec


class QuantumPlotter:
    """Creates publication-quality plots for quantum experiments."""
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        plt.style.use('default')
        sns.set_palette("husl")
        self.fig_size = (12, 8)
        
    def plot_measurement_histogram(self, counts: Dict[str, int], 
                                   title: str = "Measurement Outcomes",
                                   save_path: Optional[str] = None) -> Figure:
        """Plot histogram of measurement outcomes."""
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        states = list(counts.keys())
        values = list(counts.values())
        
        bars = ax.bar(states, values, color='steelblue', alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Quantum State', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.xticks(rotation=45 if len(states) > 8 else 0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig