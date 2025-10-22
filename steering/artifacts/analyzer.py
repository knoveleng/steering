"""
Analysis and visualization tools for steering calibration
"""

import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import json

from ..utils.logger import setup_logger


class ActivationAnalyzer:
    """
    Analyze and visualize activation steering calibration data
    """
    
    def __init__(self, output_dir: str = "analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logger
        self.logger = setup_logger(obj=self)

        self.stats = {}
    
    def plot_activation_norms(
        self,
        harmful_acts: Dict[str, torch.Tensor],
        harmless_acts: Dict[str, torch.Tensor],
        save_name: Optional[str] = None
    ) -> Dict:
        """
        Plot activation norms across layers
        Shows how activation magnitudes grow through the network
        """
        layer_names = list(harmful_acts.keys())

        # Calculate mean and variance for each layer
        harmful_means = []
        harmful_stds = []
        harmless_means = []
        harmless_stds = []

        for layer in layer_names:
            harmful_norms = harmful_acts[layer].norm(dim=-1)
            harmless_norms = harmless_acts[layer].norm(dim=-1)

            harmful_means.append(harmful_norms.mean().item())
            harmful_stds.append(harmful_norms.std().item())
            harmless_means.append(harmless_norms.mean().item())
            harmless_stds.append(harmless_norms.std().item())

        x = list(range(len(layer_names)))

        # Create Plotly figure
        fig = go.Figure()

        # Add harmful trace with shaded variance area
        fig.add_trace(go.Scatter(
            x=x + x[::-1],  # x, then x reversed
            y=[harmful_means[i] + harmful_stds[i] for i in range(len(x))] +
              [harmful_means[i] - harmful_stds[i] for i in range(len(x)-1, -1, -1)],
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False,
            name='Harmful Variance'
        ))

        # Add harmless trace with shaded variance area
        fig.add_trace(go.Scatter(
            x=x + x[::-1],  # x, then x reversed
            y=[harmless_means[i] + harmless_stds[i] for i in range(len(x))] +
              [harmless_means[i] - harmless_stds[i] for i in range(len(x)-1, -1, -1)],
            fill='toself',
            fillcolor='rgba(0, 0, 255, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False,
            name='Harmless Variance'
        ))

        # Add mean lines
        fig.add_trace(go.Scatter(
            x=x,
            y=harmful_means,
            mode='lines+markers',
            name='Harmful',
            line=dict(color='red', width=3),
            marker=dict(size=6)
        ))

        fig.add_trace(go.Scatter(
            x=x,
            y=harmless_means,
            mode='lines+markers',
            name='Harmless',
            line=dict(color='blue', width=3),
            marker=dict(size=6)
        ))

        # Update layout
        fig.update_layout(
            title=dict(text='Activation Norms Across Layers', font=dict(size=16)),
            xaxis=dict(title='Extraction Point', showgrid=True, gridcolor='lightgray'),
            yaxis=dict(title='Activation Norm', showgrid=True, gridcolor='lightgray'),
            showlegend=True,
            width=900,
            height=500,
            template='plotly_white'
        )

        if save_name:
            fig.write_image(self.output_dir / f'activation_norms_{save_name}.png', format='png')
            self.logger.info(f"  ✓ activation_norms_{save_name}.png")

        # Store original values for compatibility
        harmful_norms = harmful_means
        harmless_norms = harmless_means
        
        stats = {
            'harmful_norms': harmful_norms,
            'harmless_norms': harmless_norms,
            'layer_names': layer_names,
            'max_harmful': max(harmful_norms),
            'max_harmless': max(harmless_norms)
        }
        
        self.stats['activation_norms'] = stats
        return stats
    
    def plot_direction_projections_local(
        self,
        harmful_acts: Dict[str, torch.Tensor],
        harmless_acts: Dict[str, torch.Tensor],
        candidates: Dict[str, torch.Tensor],
        save_name: Optional[str] = None
    ) -> Dict:
        """
        Plot scalar projections on layer-specific candidate directions
        Shows how activations align with local feature directions
        """
        layer_names = list(harmful_acts.keys())

        # Calculate mean and variance for projections
        harmful_proj_means = []
        harmful_proj_stds = []
        harmless_proj_means = []
        harmless_proj_stds = []

        for layer in layer_names:
            harmful_norm = harmful_acts[layer] / (harmful_acts[layer].norm(dim=-1, keepdim=True) + 1e-8)
            harmless_norm = harmless_acts[layer] / (harmless_acts[layer].norm(dim=-1, keepdim=True) + 1e-8)
            direction = candidates[layer] / (candidates[layer].norm() + 1e-8)

            harmful_proj = harmful_norm @ direction
            harmless_proj = harmless_norm @ direction

            harmful_proj_means.append(harmful_proj.mean().item())
            harmful_proj_stds.append(harmful_proj.std().item())
            harmless_proj_means.append(harmless_proj.mean().item())
            harmless_proj_stds.append(harmless_proj.std().item())

        x = list(range(len(layer_names)))

        # Create Plotly figure
        fig = go.Figure()

        # Add harmful variance area
        fig.add_trace(go.Scatter(
            x=x + x[::-1],
            y=[harmful_proj_means[i] + harmful_proj_stds[i] for i in range(len(x))] +
              [harmful_proj_means[i] - harmful_proj_stds[i] for i in range(len(x)-1, -1, -1)],
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ))

        # Add harmless variance area
        fig.add_trace(go.Scatter(
            x=x + x[::-1],
            y=[harmless_proj_means[i] + harmless_proj_stds[i] for i in range(len(x))] +
              [harmless_proj_means[i] - harmless_proj_stds[i] for i in range(len(x)-1, -1, -1)],
            fill='toself',
            fillcolor='rgba(0, 0, 255, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ))

        # Add mean lines
        fig.add_trace(go.Scatter(
            x=x,
            y=harmful_proj_means,
            mode='lines+markers',
            name='Harmful',
            line=dict(color='red', width=3),
            marker=dict(size=6)
        ))

        fig.add_trace(go.Scatter(
            x=x,
            y=harmless_proj_means,
            mode='lines+markers',
            name='Harmless',
            line=dict(color='blue', width=3),
            marker=dict(size=6)
        ))

        # Add zero reference line
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)

        # Update layout
        fig.update_layout(
            title=dict(text='Projections on Local Candidate Directions', font=dict(size=16)),
            xaxis=dict(title='Layer Index', showgrid=True, gridcolor='lightgray'),
            yaxis=dict(title='Scalar Projection', showgrid=True, gridcolor='lightgray'),
            showlegend=True,
            width=900,
            height=500,
            template='plotly_white'
        )

        if save_name:
            fig.write_image(self.output_dir / f'projections_local_{save_name}.png', format='png')
            self.logger.info(f"  ✓ projections_local_{save_name}.png")

        # Store original values for compatibility
        harmful_projs = harmful_proj_means
        harmless_projs = harmless_proj_means
        

        
        stats = {
            'harmful_projections': harmful_projs,
            'harmless_projections': harmless_projs,
            'layer_names': layer_names,
            'mean_separation': abs(np.mean(harmful_projs) - np.mean(harmless_projs))
        }
        
        self.stats['projections_local'] = stats
        return stats
    
    def plot_direction_statistics(
        self,
        candidates: Dict[str, torch.Tensor],
        save_name: Optional[str] = None
    ) -> Dict:
        """
        Plot direction norms and inter-layer similarities
        Left: Magnitude of each candidate direction
        Right: How similar each direction is to others
        """
        layer_names = list(candidates.keys())
        
        norms = [candidates[l].norm().item() for l in layer_names]
        
        similarities = []
        for i, layer_i in enumerate(layer_names):
            sims = []
            for j, layer_j in enumerate(layer_names):
                if i != j:
                    cos_sim = torch.nn.functional.cosine_similarity(
                        candidates[layer_i].unsqueeze(0),
                        candidates[layer_j].unsqueeze(0)
                    ).item()
                    sims.append(cos_sim)
            similarities.append(np.mean(sims))
        
        x = list(range(len(layer_names)))
        best_idx = np.argmax(similarities)

        # Create Plotly figure with subplots
        fig = make_subplots(rows=1, cols=2, subplot_titles=(
            'Candidate Direction Magnitudes', 'Inter-Layer Direction Similarity'
        ))

        # Norms plot
        fig.add_trace(go.Scatter(
            x=x, y=norms, mode='lines+markers', name='Norm',
            line=dict(color='green', width=3), marker=dict(size=6)
        ), row=1, col=1)

        # Similarities plot
        fig.add_trace(go.Scatter(
            x=x, y=similarities, mode='lines+markers', name='Similarity',
            line=dict(color='purple', width=3), marker=dict(size=6)
        ), row=1, col=2)

        # Highlight best layer
        fig.add_trace(go.Scatter(
            x=[best_idx], y=[similarities[best_idx]], mode='markers',
            name=f'Best (layer {best_idx})',
            marker=dict(color='red', size=12, symbol='star', line=dict(width=1, color='black')),
            showlegend=True
        ), row=1, col=2)

        # Update layout
        fig.update_layout(
            title=dict(text='Candidate Direction Analysis', font=dict(size=16)),
            width=1000, height=500, template='plotly_white',
            showlegend=False
        )

        fig.update_xaxes(title_text='Layer Index', row=1, col=1)
        fig.update_yaxes(title_text='Direction Norm', row=1, col=1)
        fig.update_xaxes(title_text='Layer Index', row=1, col=2)
        fig.update_yaxes(title_text='Mean Cosine Similarity', row=1, col=2)

        if save_name:
            fig.write_image(self.output_dir / f'direction_statistics_{save_name}.png', format='png')
            self.logger.info(f"  ✓ direction_statistics_{save_name}.png")
        
        stats = {
            'norms': norms,
            'similarities': similarities,
            'best_layer_idx': best_idx,
            'best_layer_name': layer_names[best_idx],
            'layer_names': layer_names
        }
        
        self.stats['direction_statistics'] = stats
        return stats
    
    def plot_plane_evolution(
        self,
        candidates: Dict[str, torch.Tensor],
        basis: Tuple[torch.Tensor, torch.Tensor],
        chosen_direction: Optional[torch.Tensor] = None,
        save_name: Optional[str] = None
    ) -> Dict:
        """
        Plot how candidate directions project onto steering plane
        Shows trajectory of directions in 2D steering space
        """
        layer_names = list(candidates.keys())
        b1, b2 = basis
        
        projections = []
        for layer in layer_names:
            direction = candidates[layer]
            coeff_b1 = torch.dot(direction, b1).item()
            coeff_b2 = torch.dot(direction, b2).item()
            projections.append((coeff_b1, coeff_b2))
        
        proj_array = np.array(projections)
        colors = list(range(len(projections)))

        # Create Plotly figure
        fig = go.Figure()

        # Add trajectory lines
        for i in range(len(projections) - 1):
            fig.add_trace(go.Scatter(
                x=[projections[i][0], projections[i+1][0]],
                y=[projections[i][1], projections[i+1][1]],
                mode='lines',
                line=dict(color='gray', width=2, dash='solid'),
                showlegend=False,
                hoverinfo='skip'
            ))

        # Add scatter plot with color scale
        fig.add_trace(go.Scatter(
            x=proj_array[:, 0],
            y=proj_array[:, 1],
            mode='markers',
            marker=dict(
                color=colors,
                colorscale='viridis',
                size=12,
                opacity=0.8,
                colorbar=dict(title="Layer Index")
            ),
            name='Layer Projections',
            text=[f'Layer {i}' for i in range(len(projections))],
            hovertemplate='Layer %{text}<br>b₁: %{x:.3f}<br>b₂: %{y:.3f}<extra></extra>'
        ))

        # Add chosen direction if provided
        if chosen_direction is not None:
            chosen_b1 = torch.dot(chosen_direction, b1).item()
            chosen_b2 = torch.dot(chosen_direction, b2).item()
            fig.add_trace(go.Scatter(
                x=[chosen_b1],
                y=[chosen_b2],
                mode='markers',
                marker=dict(color='red', size=20, symbol='star',
                           line=dict(width=2, color='black')),
                name='Selected Direction'
            ))

        # Add basis vectors as arrows (approximated with lines)
        fig.add_trace(go.Scatter(
            x=[0, 0.8], y=[0, 0],
            mode='lines+text',
            line=dict(color='blue', width=3),
            text=['', 'b₁'],
            textposition='middle right',
            textfont=dict(color='blue', size=14),
            showlegend=False,
            name='Basis b₁'
        ))

        fig.add_trace(go.Scatter(
            x=[0, 0], y=[0, 0.8],
            mode='lines+text',
            line=dict(color='green', width=3),
            text=['', 'b₂'],
            textposition='top center',
            textfont=dict(color='green', size=14),
            showlegend=False,
            name='Basis b₂'
        ))

        # Add reference lines
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.3)
        fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.3)

        # Update layout
        fig.update_layout(
            title=dict(text='Steering Plane Evolution', font=dict(size=16)),
            xaxis=dict(title='Projection on b₁', showgrid=True, gridcolor='lightgray', scaleanchor="y", scaleratio=1),
            yaxis=dict(title='Projection on b₂', showgrid=True, gridcolor='lightgray'),
            showlegend=True,
            width=700,
            height=700,
            template='plotly_white'
        )

        if save_name:
            fig.write_image(self.output_dir / f'plane_evolution_{save_name}.png', format='png')
            self.logger.info(f"  ✓ plane_evolution_{save_name}.png")
        
        stats = {
            'projections': projections,
            'layer_names': layer_names
        }
        
        self.stats['plane_evolution'] = stats
        return stats
    
    def plot_feature_alignment(
        self,
        harmful_acts: Dict[str, torch.Tensor],
        harmless_acts: Dict[str, torch.Tensor],
        chosen_direction: torch.Tensor,
        save_name: Optional[str] = None
    ) -> Dict:
        """
        Plot projections on the selected feature direction
        Shows how well activations align with chosen direction across layers
        """
        layer_names = list(harmful_acts.keys())
        chosen_norm = chosen_direction / (chosen_direction.norm() + 1e-8)

        # Calculate mean and variance for projections
        harmful_proj_means = []
        harmful_proj_stds = []
        harmless_proj_means = []
        harmless_proj_stds = []

        for layer in layer_names:
            harmful_norm = harmful_acts[layer] / (harmful_acts[layer].norm(dim=-1, keepdim=True) + 1e-8)
            harmless_norm = harmless_acts[layer] / (harmless_acts[layer].norm(dim=-1, keepdim=True) + 1e-8)

            harmful_proj = harmful_norm @ chosen_norm
            harmless_proj = harmless_norm @ chosen_norm

            harmful_proj_means.append(harmful_proj.mean().item())
            harmful_proj_stds.append(harmful_proj.std().item())
            harmless_proj_means.append(harmless_proj.mean().item())
            harmless_proj_stds.append(harmless_proj.std().item())

        x = list(range(len(layer_names)))

        # Create Plotly figure
        fig = go.Figure()

        # Add harmful variance area
        fig.add_trace(go.Scatter(
            x=x + x[::-1],
            y=[harmful_proj_means[i] + harmful_proj_stds[i] for i in range(len(x))] +
              [harmful_proj_means[i] - harmful_proj_stds[i] for i in range(len(x)-1, -1, -1)],
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ))

        # Add harmless variance area
        fig.add_trace(go.Scatter(
            x=x + x[::-1],
            y=[harmless_proj_means[i] + harmless_proj_stds[i] for i in range(len(x))] +
              [harmless_proj_means[i] - harmless_proj_stds[i] for i in range(len(x)-1, -1, -1)],
            fill='toself',
            fillcolor='rgba(0, 0, 255, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ))

        # Add mean lines
        fig.add_trace(go.Scatter(
            x=x,
            y=harmful_proj_means,
            mode='lines+markers',
            name='Harmful',
            line=dict(color='red', width=3),
            marker=dict(size=6)
        ))

        fig.add_trace(go.Scatter(
            x=x,
            y=harmless_proj_means,
            mode='lines+markers',
            name='Harmless',
            line=dict(color='blue', width=3),
            marker=dict(size=6)
        ))

        # Add zero reference line
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)

        # Update layout
        fig.update_layout(
            title=dict(text='Alignment with Selected Feature Direction', font=dict(size=16)),
            xaxis=dict(title='Layer Index', showgrid=True, gridcolor='lightgray'),
            yaxis=dict(title='Scalar Projection', showgrid=True, gridcolor='lightgray'),
            showlegend=True,
            width=900,
            height=500,
            template='plotly_white'
        )

        if save_name:
            fig.write_image(self.output_dir / f'feature_alignment_{save_name}.png', format='png')
            self.logger.info(f"  ✓ feature_alignment_{save_name}.png")

        # Store original values for compatibility
        harmful_projs = harmful_proj_means
        harmless_projs = harmless_proj_means
        
        stats = {
            'harmful_projections': harmful_projs,
            'harmless_projections': harmless_projs,
            'layer_names': layer_names,
            'mean_separation': abs(np.mean(harmful_projs) - np.mean(harmless_projs))
        }
        
        self.stats['feature_alignment'] = stats
        return stats
    
    def plot_similarity_matrix(
        self,
        candidates: Dict[str, torch.Tensor],
        save_name: Optional[str] = None
    ) -> Dict:
        """
        Plot full pairwise similarity matrix between candidate directions
        """
        layer_names = list(candidates.keys())
        n_layers = len(layer_names)
        
        sim_matrix = torch.zeros(n_layers, n_layers)
        
        for i, layer_i in enumerate(layer_names):
            for j, layer_j in enumerate(layer_names):
                vec_i = candidates[layer_i].float()
                vec_j = candidates[layer_j].float()
                
                sim = torch.nn.functional.cosine_similarity(
                    vec_i.unsqueeze(0),
                    vec_j.unsqueeze(0)
                )
                sim_matrix[i, j] = sim.item()
        
        # Create Plotly heatmap
        fig = go.Figure(data=go.Heatmap(
            z=sim_matrix.numpy(),
            x=list(range(n_layers)),
            y=list(range(n_layers)),
            colorscale='RdYlGn',
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Cosine Similarity"),
            hovertemplate='Layer %{x} vs Layer %{y}<br>Similarity: %{z:.3f}<extra></extra>'
        ))

        # Update layout
        fig.update_layout(
            title=dict(text='Direction Similarity Matrix', font=dict(size=16)),
            xaxis=dict(title='Layer Index'),
            yaxis=dict(title='Layer Index'),
            width=600,
            height=600,
            template='plotly_white'
        )

        if save_name:
            fig.write_image(self.output_dir / f'similarity_matrix_{save_name}.png', format='png')
            self.logger.info(f"  ✓ similarity_matrix_{save_name}.png")
        
        stats = {
            'similarity_matrix': sim_matrix.numpy().tolist(),
            'layer_names': layer_names
        }
        
        self.stats['similarity_matrix'] = stats
        return stats
    
    def analyze_all(
        self,
        harmful_acts: Dict[str, torch.Tensor],
        harmless_acts: Dict[str, torch.Tensor],
        candidates: Dict[str, torch.Tensor],
        chosen_direction: torch.Tensor,
        basis: Tuple[torch.Tensor, torch.Tensor],
        save_name: str
    ) -> Dict:
        """Run all analyses and generate all visualizations"""
        self.logger.info("Generating Analysis Plots:")
        self.logger.info("="*60)

        self.logger.info("[1/6] Activation norms...")
        self.plot_activation_norms(harmful_acts, harmless_acts, save_name)

        self.logger.info("[2/6] Local direction projections...")
        self.plot_direction_projections_local(harmful_acts, harmless_acts, candidates, save_name)

        self.logger.info("[3/6] Direction statistics...")
        self.plot_direction_statistics(candidates, save_name)

        self.logger.info("[4/6] Plane evolution...")
        self.plot_plane_evolution(candidates, basis, chosen_direction, save_name)

        self.logger.info("[5/6] Feature alignment...")
        self.plot_feature_alignment(harmful_acts, harmless_acts, chosen_direction, save_name)

        self.logger.info("[6/6] Similarity matrix...")
        self.plot_similarity_matrix(candidates, save_name)

        self.save_statistics(save_name)

        self.logger.info(f"✓ Analysis complete! See {self.output_dir}/")
        return self.stats
    
    def save_statistics(self, save_name: str):
        """Save all computed statistics to JSON"""
        def convert(obj):
            # Handle numpy/torch tensors
            if isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            # Handle numpy scalar types
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            # Handle containers
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert(item) for item in obj]
            # Return as-is for basic types
            else:
                return obj
        
        stats_clean = convert(self.stats)
        
        filepath = self.output_dir / f'statistics_{save_name}.json'
        with open(filepath, 'w') as f:
            json.dump(stats_clean, f, indent=2)
        
        self.logger.info(f"  ✓ statistics_{save_name}.json")