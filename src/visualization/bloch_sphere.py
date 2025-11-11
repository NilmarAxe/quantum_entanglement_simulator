import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, List, Tuple


class BlochSphereVisualizer:
    """Visualize quantum states on the Bloch sphere."""
    
    def __init__(self, fig_size: Tuple[int, int] = (10, 10)):
        self.fig_size = fig_size
        
    def plot_state(self, statevector: np.ndarray, 
                  title: str = "Quantum State on Bloch Sphere",
                  save_path: Optional[str] = None) -> plt.Figure:
        """Plot single qubit state on Bloch sphere.
        
        Args:
            statevector: Complex amplitude [α, β] where |ψ⟩ = α|0⟩ + β|1⟩
        """
        fig = plt.figure(figsize=self.fig_size)
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw Bloch sphere
        self._draw_sphere(ax)
        
        # Convert statevector to Bloch vector
        bloch_vec = self._statevector_to_bloch(statevector)
        
        # Plot state vector
        ax.quiver(0, 0, 0, bloch_vec[0], bloch_vec[1], bloch_vec[2],
                 color='red', arrow_length_ratio=0.1, linewidth=3,
                 label='|ψ⟩')
        
        # Add coordinate axes
        ax.plot([0, 1.3], [0, 0], [0, 0], 'k-', linewidth=1, alpha=0.3)
        ax.plot([0, 0], [0, 1.3], [0, 0], 'k-', linewidth=1, alpha=0.3)
        ax.plot([0, 0], [0, 0], [0, 1.3], 'k-', linewidth=1, alpha=0.3)
        
        # Labels
        ax.text(1.4, 0, 0, 'X', fontsize=12, fontweight='bold')
        ax.text(0, 1.4, 0, 'Y', fontsize=12, fontweight='bold')
        ax.text(0, 0, 1.4, '|0⟩', fontsize=12, fontweight='bold')
        ax.text(0, 0, -1.4, '|1⟩', fontsize=12, fontweight='bold')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.set_zlim([-1.2, 1.2])
        
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def plot_trajectory(self, statevectors: List[np.ndarray],
                       title: str = "Quantum State Evolution",
                       save_path: Optional[str] = None) -> plt.Figure:
        """Plot evolution of quantum state on Bloch sphere."""
        fig = plt.figure(figsize=self.fig_size)
        ax = fig.add_subplot(111, projection='3d')
        
        self._draw_sphere(ax)
        
        # Convert all states to Bloch vectors
        trajectory = [self._statevector_to_bloch(sv) for sv in statevectors]
        trajectory = np.array(trajectory)
        
        # Plot trajectory
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
               'b-', linewidth=2, alpha=0.6, label='Evolution Path')
        
        # Mark initial and final states
        ax.scatter(*trajectory[0], color='green', s=100, 
                  label='Initial State', marker='o')
        ax.scatter(*trajectory[-1], color='red', s=100,
                  label='Final State', marker='s')
        
        # Add coordinate axes
        ax.plot([0, 1.3], [0, 0], [0, 0], 'k-', linewidth=1, alpha=0.3)
        ax.plot([0, 0], [0, 1.3], [0, 0], 'k-', linewidth=1, alpha=0.3)
        ax.plot([0, 0], [0, 0], [0, 1.3], 'k-', linewidth=1, alpha=0.3)
        
        ax.text(1.4, 0, 0, 'X', fontsize=12, fontweight='bold')
        ax.text(0, 1.4, 0, 'Y', fontsize=12, fontweight='bold')
        ax.text(0, 0, 1.4, '|0⟩', fontsize=12, fontweight='bold')
        ax.text(0, 0, -1.4, '|1⟩', fontsize=12, fontweight='bold')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.set_zlim([-1.2, 1.2])
        
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def _draw_sphere(self, ax: Axes3D) -> None:
        """Draw the Bloch sphere surface."""
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax.plot_surface(x, y, z, color='cyan', alpha=0.1, 
                       linewidth=0, antialiased=True)
        
        # Draw equator and meridians
        theta = np.linspace(0, 2*np.pi, 100)
        
        # Equator (XY plane)
        ax.plot(np.cos(theta), np.sin(theta), 0, 'b--', linewidth=1, alpha=0.3)
        
        # XZ meridian
        ax.plot(np.cos(theta), 0, np.sin(theta), 'b--', linewidth=1, alpha=0.3)
        
        # YZ meridian
        ax.plot(0, np.cos(theta), np.sin(theta), 'b--', linewidth=1, alpha=0.3)
        
    def _statevector_to_bloch(self, statevector: np.ndarray) -> np.ndarray:
        """Convert statevector to Bloch sphere coordinates.
        
        For |ψ⟩ = α|0⟩ + β|1⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩
        Bloch vector: (sin(θ)cos(φ), sin(θ)sin(φ), cos(θ))
        """
        alpha, beta = statevector[0], statevector[1]
        
        # Handle numerical precision
        if abs(alpha) < 1e-10:
            theta = np.pi
            phi = 0
        elif abs(beta) < 1e-10:
            theta = 0
            phi = 0
        else:
            theta = 2 * np.arctan2(abs(beta), abs(alpha))
            phi = np.angle(beta) - np.angle(alpha)
        
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        
        return np.array([x, y, z])