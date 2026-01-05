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
    
    def set_session_output_dir(self, session_name: str) -> None:
        """
        Set output directory to a session-specific subdirectory.
        
        This ensures analysis files don't overwrite each other when running
        multiple calibrations for the same model.
        
        Args:
            session_name: Unique session identifier (e.g., 'analysis_model_timestamp')
        """
        self.output_dir = self.output_dir / session_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Analysis output directory set to: {self.output_dir}")
    
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
            title=dict(text='Activation Norms Across Layers', font=dict(size=28)),
            xaxis=dict(title=dict(text='Extraction Point', font=dict(size=22)), showgrid=True, gridcolor='lightgray', tickfont=dict(size=18)),
            yaxis=dict(title=dict(text='Activation Norm', font=dict(size=22)), showgrid=True, gridcolor='lightgray', tickfont=dict(size=18)),
            legend=dict(font=dict(size=20)),
            showlegend=True,
            width=900,
            height=500,
            template='plotly_white'
        )

        if save_name:
            fig.write_image(self.output_dir / f'activation_norms_{save_name}.pdf')
            self.logger.info(f"  ✓ activation_norms_{save_name}.pdf")

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
            title=dict(text='Projections on Local Candidate Directions', font=dict(size=28)),
            xaxis=dict(title=dict(text='Layer Index', font=dict(size=22)), showgrid=True, gridcolor='lightgray', tickfont=dict(size=18)),
            yaxis=dict(title=dict(text='Scalar Projection', font=dict(size=22)), showgrid=True, gridcolor='lightgray', tickfont=dict(size=18)),
            legend=dict(font=dict(size=20)),
            showlegend=True,
            width=900,
            height=500,
            template='plotly_white'
        )

        if save_name:
            fig.write_image(self.output_dir / f'projections_local_{save_name}.pdf')
            self.logger.info(f"  ✓ projections_local_{save_name}.pdf")

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
            title=dict(text='Candidate Direction Analysis', font=dict(size=28)),
            width=1000, height=500, template='plotly_white',
            showlegend=False
        )

        fig.update_xaxes(title_text='Layer Index', title_font=dict(size=22), tickfont=dict(size=18), row=1, col=1)
        fig.update_yaxes(title_text='Direction Norm', title_font=dict(size=22), tickfont=dict(size=18), row=1, col=1)
        fig.update_xaxes(title_text='Layer Index', title_font=dict(size=22), tickfont=dict(size=18), row=1, col=2)
        fig.update_yaxes(title_text='Mean Cosine Similarity', title_font=dict(size=22), tickfont=dict(size=18), row=1, col=2)

        if save_name:
            fig.write_image(self.output_dir / f'direction_statistics_{save_name}.pdf')
            self.logger.info(f"  ✓ direction_statistics_{save_name}.pdf")
        
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
            textfont=dict(color='blue', size=24),
            showlegend=False,
            name='Basis b₁'
        ))

        fig.add_trace(go.Scatter(
            x=[0, 0], y=[0, 0.8],
            mode='lines+text',
            line=dict(color='green', width=3),
            text=['', 'b₂'],
            textposition='top center',
            textfont=dict(color='green', size=24),
            showlegend=False,
            name='Basis b₂'
        ))

        # Add reference lines
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.3)
        fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.3)

        # Update layout
        fig.update_layout(
            title=dict(text='Steering Plane Evolution', font=dict(size=28)),
            xaxis=dict(title=dict(text='Projection on b₁', font=dict(size=22)), showgrid=True, gridcolor='lightgray', scaleanchor="y", scaleratio=1, tickfont=dict(size=18)),
            yaxis=dict(title=dict(text='Projection on b₂', font=dict(size=22)), showgrid=True, gridcolor='lightgray', tickfont=dict(size=18)),
            legend=dict(font=dict(size=20)),
            showlegend=True,
            width=700,
            height=700,
            template='plotly_white'
        )

        if save_name:
            fig.write_image(self.output_dir / f'plane_evolution_{save_name}.pdf')
            self.logger.info(f"  ✓ plane_evolution_{save_name}.pdf")
        
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
            title=dict(text='Alignment with Selected Feature Direction', font=dict(size=28)),
            xaxis=dict(title=dict(text='Layer Index', font=dict(size=22)), showgrid=True, gridcolor='lightgray', tickfont=dict(size=18)),
            yaxis=dict(title=dict(text='Scalar Projection', font=dict(size=22)), showgrid=True, gridcolor='lightgray', tickfont=dict(size=18)),
            legend=dict(font=dict(size=20)),
            showlegend=True,
            width=900,
            height=500,
            template='plotly_white'
        )

        if save_name:
            fig.write_image(self.output_dir / f'feature_alignment_{save_name}.pdf')
            self.logger.info(f"  ✓ feature_alignment_{save_name}.pdf")

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
            title=dict(text='Direction Similarity Matrix', font=dict(size=28)),
            xaxis=dict(title=dict(text='Layer Index', font=dict(size=22)), tickfont=dict(size=18)),
            yaxis=dict(title=dict(text='Layer Index', font=dict(size=22)), tickfont=dict(size=18)),
            width=600,
            height=600,
            template='plotly_white'
        )

        if save_name:
            fig.write_image(self.output_dir / f'similarity_matrix_{save_name}.pdf')
            self.logger.info(f"  ✓ similarity_matrix_{save_name}.pdf")
        
        stats = {
            'similarity_matrix': sim_matrix.numpy().tolist(),
            'layer_names': layer_names
        }
        
        self.stats['similarity_matrix'] = stats
        return stats

    def plot_activations_on_steering_plane(
        self,
        positive_activations: Dict[str, torch.Tensor],
        negative_activations: Dict[str, torch.Tensor],
        basis: Tuple[torch.Tensor, torch.Tensor],
        save_name: Optional[str] = None
    ) -> Dict:
        """
        Plot mean activations from all layers projected onto the steering plane
        
        Creates a 2D scatter plot showing how the mean activation of each layer
        projects onto the steering plane defined by basis vectors (b1, b2).
        Each point represents one layer's mean activation.
        
        Args:
            positive_activations: Dict[layer, Tensor(n_samples, hidden_dim)]
            negative_activations: Dict[layer, Tensor(n_samples, hidden_dim)]
            basis: Tuple of (b1, b2) basis vectors defining the steering plane
            save_name: Optional name for saving the plot
            
        Returns:
            Dict containing projection statistics
        """
        layer_names = list(positive_activations.keys())
        
        # Unpack basis vectors
        b1, b2 = basis
        b1 = b1.float()
        b2 = b2.float()
        
        # Compute mean activation for each layer
        pos_means_b1 = []
        pos_means_b2 = []
        neg_means_b1 = []
        neg_means_b2 = []
        valid_layers = []
        
        for layer_name in layer_names:
            if layer_name not in positive_activations or layer_name not in negative_activations:
                continue
            
            # Compute mean activation for this layer
            pos_mean = positive_activations[layer_name].float().mean(dim=0)  # Shape: (hidden_dim,)
            neg_mean = negative_activations[layer_name].float().mean(dim=0)  # Shape: (hidden_dim,)
            
            # Project mean activations onto basis
            pos_proj_b1 = torch.dot(pos_mean, b1).item()
            pos_proj_b2 = torch.dot(pos_mean, b2).item()
            neg_proj_b1 = torch.dot(neg_mean, b1).item()
            neg_proj_b2 = torch.dot(neg_mean, b2).item()
            
            pos_means_b1.append(pos_proj_b1)
            pos_means_b2.append(pos_proj_b2)
            neg_means_b1.append(neg_proj_b1)
            neg_means_b2.append(neg_proj_b2)
            valid_layers.append(layer_name)
        
        if len(valid_layers) == 0:
            raise ValueError("No valid layers found with both positive and negative activations")
        
        # Convert to numpy arrays - these should be arrays, not single floats
        pos_x = np.array(pos_means_b1, dtype=np.float32)
        pos_y = np.array(pos_means_b2, dtype=np.float32)
        neg_x = np.array(neg_means_b1, dtype=np.float32)
        neg_y = np.array(neg_means_b2, dtype=np.float32)
        
        # Debug: verify these are arrays
        assert pos_x.ndim == 1, f"pos_x should be 1D array, got shape {pos_x.shape}"
        assert pos_y.ndim == 1, f"pos_y should be 1D array, got shape {pos_y.shape}"
        assert neg_x.ndim == 1, f"neg_x should be 1D array, got shape {neg_x.shape}"
        assert neg_y.ndim == 1, f"neg_y should be 1D array, got shape {neg_y.shape}"
        
        # Create color scale for layers (like in plot_plane_evolution)
        n_layers = len(valid_layers)
        colors_pos = list(range(n_layers))
        colors_neg = list(range(n_layers))
        
        # Create Plotly scatter plot
        fig = go.Figure()
        
        # Add trajectory lines for positive activations
        for i in range(n_layers - 1):
            fig.add_trace(go.Scatter(
                x=[float(pos_x[i]), float(pos_x[i+1])],
                y=[float(pos_y[i]), float(pos_y[i+1])],
                mode='lines',
                line=dict(color='rgba(255, 0, 0, 0.3)', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Add trajectory lines for negative activations
        for i in range(n_layers - 1):
            fig.add_trace(go.Scatter(
                x=[float(neg_x[i]), float(neg_x[i+1])],
                y=[float(neg_y[i]), float(neg_y[i+1])],
                mode='lines',
                line=dict(color='rgba(0, 0, 255, 0.3)', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Add positive mean activations with gradient coloring
        fig.add_trace(go.Scatter(
            x=pos_x.tolist(),  # Convert to list explicitly
            y=pos_y.tolist(),
            mode='markers',
            name='Positive Means',
            marker=dict(
                size=10,
                color=colors_pos,
                colorscale='Reds',
                opacity=0.8,
                line=dict(width=1, color='darkred'),
                colorbar=dict(title="Layer", x=1.15, len=0.4, y=0.75)
            ),
            text=[f'Layer {i}' for i in range(n_layers)],
            hovertemplate='Positive - %{text}<br>b₁: %{x:.3f}<br>b₂: %{y:.3f}<extra></extra>'
        ))
        
        # Add negative mean activations with gradient coloring
        fig.add_trace(go.Scatter(
            x=neg_x.tolist(),  # Convert to list explicitly
            y=neg_y.tolist(),
            mode='markers',
            name='Negative Means',
            marker=dict(
                size=10,
                color=colors_neg,
                colorscale='Blues',
                opacity=0.8,
                line=dict(width=1, color='darkblue'),
                colorbar=dict(title="Layer", x=1.0, len=0.4, y=0.25)
            ),
            text=[f'Layer {i}' for i in range(n_layers)],
            hovertemplate='Negative - %{text}<br>b₁: %{x:.3f}<br>b₂: %{y:.3f}<extra></extra>'
        ))
        
        # Add overall centroids (mean of all layer means)
        pos_centroid_x = float(pos_x.mean())
        pos_centroid_y = float(pos_y.mean())
        neg_centroid_x = float(neg_x.mean())
        neg_centroid_y = float(neg_y.mean())
        
        fig.add_trace(go.Scatter(
            x=[pos_centroid_x],
            y=[pos_centroid_y],
            mode='markers',
            name='Positive Centroid',
            marker=dict(
                size=18,
                color='red',
                symbol='x',
                line=dict(width=3, color='darkred')
            ),
            hovertemplate='Pos Centroid<br>b₁: %{x:.3f}<br>b₂: %{y:.3f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=[neg_centroid_x],
            y=[neg_centroid_y],
            mode='markers',
            name='Negative Centroid',
            marker=dict(
                size=18,
                color='blue',
                symbol='x',
                line=dict(width=3, color='darkblue')
            ),
            hovertemplate='Neg Centroid<br>b₁: %{x:.3f}<br>b₂: %{y:.3f}<extra></extra>'
        ))
        
        # Add line connecting centroids
        fig.add_trace(go.Scatter(
            x=[neg_centroid_x, pos_centroid_x],
            y=[neg_centroid_y, pos_centroid_y],
            mode='lines',
            name='Separation Vector',
            line=dict(color='black', width=2, dash='dash'),
            hoverinfo='skip',
            showlegend=True
        ))
        
        # Add basis vectors as arrows
        # Calculate scale based on data range
        all_x = np.concatenate([pos_x, neg_x])
        all_y = np.concatenate([pos_y, neg_y])
        x_range = float(max(abs(all_x.min()), abs(all_x.max()))) if len(all_x) > 0 else 1.0
        y_range = float(max(abs(all_y.min()), abs(all_y.max()))) if len(all_y) > 0 else 1.0
        arrow_scale = float(min(x_range, y_range) * 0.3)
        
        # Make sure arrow_scale is reasonable
        if arrow_scale < 0.01:
            arrow_scale = 0.5
        
        fig.add_trace(go.Scatter(
            x=[0.0, arrow_scale],
            y=[0.0, 0.0],
            mode='lines+text',
            line=dict(color='blue', width=3),
            text=['', 'b₁'],
            textposition='middle right',
            textfont=dict(color='blue', size=24),
            showlegend=False,
            name='Basis b₁',
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=[0.0, 0.0],
            y=[0.0, arrow_scale],
            mode='lines+text',
            line=dict(color='green', width=3),
            text=['', 'b₂'],
            textposition='top center',
            textfont=dict(color='green', size=24),
            showlegend=False,
            name='Basis b₂',
            hoverinfo='skip'
        ))
        
        # Add axes through origin
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.3, annotation_text="")
        fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.3, annotation_text="")
        
        # Update layout
        fig.update_layout(
            title=dict(
                text='Mean Activations on Steering Plane (All Layers)',
                font=dict(size=28)
            ),
            xaxis=dict(
                title=dict(text='Basis Vector b₁', font=dict(size=22)),
                showgrid=True,
                gridcolor='lightgray',
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='gray',
                tickfont=dict(size=18)
            ),
            yaxis=dict(
                title=dict(text='Basis Vector b₂', font=dict(size=22)),
                showgrid=True,
                gridcolor='lightgray',
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='gray',
                tickfont=dict(size=18)
            ),
            legend=dict(font=dict(size=20)),
            showlegend=True,
            width=1000,
            height=800,
            template='plotly_white',
            hovermode='closest'
        )
        
        # Make axes equal for better visualization
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        
        if save_name:
            fig.write_image(self.output_dir / f'steering_plane_{save_name}.pdf')
            
            if hasattr(self, 'logger'):
                self.logger.info(f"  ✓ steering_plane_{save_name}.pdf")
        
        # Compute statistics
        separation_distance = np.sqrt((pos_centroid_x - neg_centroid_x)**2 + 
                                    (pos_centroid_y - neg_centroid_y)**2)
        
        # Compute variance along each basis axis (variance of layer means)
        pos_var_b1 = float(pos_x.var())
        pos_var_b2 = float(pos_y.var())
        neg_var_b1 = float(neg_x.var())
        neg_var_b2 = float(neg_y.var())
        
        # Compute trajectory lengths (how much the means move across layers)
        pos_trajectory_length = float(np.sum(np.sqrt(np.diff(pos_x)**2 + np.diff(pos_y)**2)))
        neg_trajectory_length = float(np.sum(np.sqrt(np.diff(neg_x)**2 + np.diff(neg_y)**2)))
        
        stats = {
            'n_layers': n_layers,
            'layer_names': valid_layers,
            'positive_centroid': [pos_centroid_x, pos_centroid_y],
            'negative_centroid': [neg_centroid_x, neg_centroid_y],
            'separation_distance': float(separation_distance),
            'positive_variance_b1': pos_var_b1,
            'positive_variance_b2': pos_var_b2,
            'negative_variance_b1': neg_var_b1,
            'negative_variance_b2': neg_var_b2,
            'positive_trajectory_length': pos_trajectory_length,
            'negative_trajectory_length': neg_trajectory_length,
            'positive_projections_b1': pos_x.tolist(),
            'positive_projections_b2': pos_y.tolist(),
            'negative_projections_b1': neg_x.tolist(),
            'negative_projections_b2': neg_y.tolist(),
        }
        
        if hasattr(self, 'stats'):
            self.stats['steering_plane'] = stats
        
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

        self.logger.info("[7/6] Activation on steering plane...")
        self.plot_activations_on_steering_plane(harmful_acts, harmless_acts, basis, save_name)

        self.save_statistics(save_name)

        self.logger.info(f"✓ Analysis complete! See {self.output_dir}/")
        return self.stats
    
    def plot_selective_layer_steering(
        self,
        projection_stats: Dict[str, Dict[str, float]],
        layer_steering_mask: Dict[str, bool],
        save_name: Optional[str] = None,
        selection_method: str = 'opposite_signs'
    ) -> Dict:
        """
        Plot layer selection for selective steering.

        Shows which layers are selected for steering. For range-based methods,
        highlights the contiguous range selection.

        Args:
            projection_stats: Dict from SelectiveSteeringOperator.compute_layer_projection_stats()
                Each entry contains 'pos_mean', 'neg_mean', 'pos_std', 'neg_std',
                'opposite_signs', 'separation'
            layer_steering_mask: Dict mapping layer names to bool (True = selected for steering)
            save_name: Optional name for saving the plot
            selection_method: Selection method used ('opposite_signs', 'weighted_quality', etc.)

        Returns:
            Dict containing statistics about layer selection
        """
        layer_names = list(projection_stats.keys())
        n_layers = len(layer_names)

        # Extract data
        pos_means = [projection_stats[l]['pos_mean'] for l in layer_names]
        neg_means = [projection_stats[l]['neg_mean'] for l in layer_names]
        pos_stds = [projection_stats[l]['pos_std'] for l in layer_names]
        neg_stds = [projection_stats[l]['neg_std'] for l in layer_names]
        selected = [layer_steering_mask[l] for l in layer_names]
        separations = [projection_stats[l]['separation'] for l in layer_names]

        x = list(range(n_layers))

        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                'Mean Projections on Feature Direction (Selected Layers Highlighted)',
                'Projection Separation by Layer'
            ),
            vertical_spacing=0.12,
            row_heights=[0.6, 0.4]
        )

        # Top plot: Mean projections with variance
        # Add variance bands for positive
        fig.add_trace(go.Scatter(
            x=x + x[::-1],
            y=[pos_means[i] + pos_stds[i] for i in range(n_layers)] +
              [pos_means[i] - pos_stds[i] for i in range(n_layers-1, -1, -1)],
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False,
            name='Positive Variance'
        ), row=1, col=1)

        # Add variance bands for negative
        fig.add_trace(go.Scatter(
            x=x + x[::-1],
            y=[neg_means[i] + neg_stds[i] for i in range(n_layers)] +
              [neg_means[i] - neg_stds[i] for i in range(n_layers-1, -1, -1)],
            fill='toself',
            fillcolor='rgba(0, 0, 255, 0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False,
            name='Negative Variance'
        ), row=1, col=1)

        # Add mean lines with markers - color based on selection
        # Positive projections
        pos_colors = ['red' if sel else 'rgba(255, 0, 0, 0.3)' for sel in selected]
        for i in range(n_layers):
            showlegend = i == 0 or (i == 1 and not selected[0])
            fig.add_trace(go.Scatter(
                x=[x[i]],
                y=[pos_means[i]],
                mode='markers+lines' if i < n_layers - 1 else 'markers',
                name='Positive (selected)' if selected[i] and showlegend else 'Positive (not selected)',
                marker=dict(
                    size=10 if selected[i] else 6,
                    color=pos_colors[i],
                    symbol='circle'
                ),
                line=dict(color=pos_colors[i], width=2) if i < n_layers - 1 else None,
                showlegend=showlegend and (selected[i] or i == 1),
                legendgroup='positive',
                hovertemplate=f'Layer {i}<br>Pos Mean: %{{y:.3f}}<br>Selected: {selected[i]}<extra></extra>'
            ), row=1, col=1)

        # Negative projections
        neg_colors = ['blue' if sel else 'rgba(0, 0, 255, 0.3)' for sel in selected]
        for i in range(n_layers):
            showlegend = i == 0 or (i == 1 and not selected[0])
            fig.add_trace(go.Scatter(
                x=[x[i]],
                y=[neg_means[i]],
                mode='markers+lines' if i < n_layers - 1 else 'markers',
                name='Negative (selected)' if selected[i] and showlegend else 'Negative (not selected)',
                marker=dict(
                    size=10 if selected[i] else 6,
                    color=neg_colors[i],
                    symbol='circle'
                ),
                line=dict(color=neg_colors[i], width=2) if i < n_layers - 1 else None,
                showlegend=showlegend and (selected[i] or i == 1),
                legendgroup='negative',
                hovertemplate=f'Layer {i}<br>Neg Mean: %{{y:.3f}}<br>Selected: {selected[i]}<extra></extra>'
            ), row=1, col=1)

        # Add zero reference line
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.3, row=1, col=1)

        # Highlight selected layers with vertical spans
        for i, sel in enumerate(selected):
            if sel:
                fig.add_vrect(
                    x0=i - 0.4, x1=i + 0.4,
                    fillcolor="rgba(0, 255, 0, 0.1)",
                    layer="below",
                    line_width=0,
                    row=1, col=1
                )

        # Bottom plot: Separation bars
        bar_colors = ['green' if sel else 'lightgray' for sel in selected]
        fig.add_trace(go.Bar(
            x=x,
            y=separations,
            marker=dict(color=bar_colors),
            name='Separation',
            hovertemplate='Layer %{x}<br>Separation: %{y:.3f}<br>Selected: ' +
                         '<br>'.join([str(s) for s in selected]) + '<extra></extra>',
            showlegend=False
        ), row=2, col=1)

        # Update axes
        fig.update_xaxes(title_text='Layer Index', title_font=dict(size=22), tickfont=dict(size=18), row=1, col=1, showgrid=True, gridcolor='lightgray')
        fig.update_yaxes(title_text='Mean Projection', title_font=dict(size=22), tickfont=dict(size=18), row=1, col=1, showgrid=True, gridcolor='lightgray')
        fig.update_xaxes(title_text='Layer Index', title_font=dict(size=22), tickfont=dict(size=18), row=2, col=1, showgrid=True, gridcolor='lightgray')
        fig.update_yaxes(title_text='|Pos Mean - Neg Mean|', title_font=dict(size=22), tickfont=dict(size=18), row=2, col=1, showgrid=True, gridcolor='lightgray')

        # Check if selection is contiguous range
        selected_indices = [i for i, s in enumerate(selected) if s]
        is_contiguous = (len(selected_indices) > 0 and
                        selected_indices == list(range(selected_indices[0], selected_indices[-1] + 1)))

        # Create title
        if is_contiguous and len(selected_indices) > 0:
            title_text = (f'Selective Layer Steering Analysis: {selection_method}\n'
                         f'Range [{selected_indices[0]}:{selected_indices[-1]}] '
                         f'({sum(selected)}/{n_layers} layers)')
        else:
            title_text = (f'Selective Layer Steering Analysis: {selection_method}\n'
                         f'{sum(selected)}/{n_layers} layers selected')

        # Update layout
        fig.update_layout(
            title=dict(
                text=title_text,
                font=dict(size=28)
            ),
            showlegend=True,
            width=1000,
            height=900,
            template='plotly_white',
            hovermode='closest'
        )

        if save_name:
            fig.write_image(self.output_dir / f'selective_steering_{save_name}.pdf')
            self.logger.info(f"  ✓ selective_steering_{save_name}.pdf")

        # Compute statistics
        n_selected = sum(selected)
        selected_layers = [layer_names[i] for i, sel in enumerate(selected) if sel]
        avg_separation_selected = np.mean([separations[i] for i, sel in enumerate(selected) if sel]) if n_selected > 0 else 0
        avg_separation_all = np.mean(separations)

        stats = {
            'n_total_layers': n_layers,
            'n_selected_layers': n_selected,
            'selection_ratio': n_selected / n_layers if n_layers > 0 else 0,
            'selected_layer_names': selected_layers,
            'avg_separation_selected': float(avg_separation_selected),
            'avg_separation_all': float(avg_separation_all),
            'layer_stats': {
                layer: {
                    'pos_mean': pos_means[i],
                    'neg_mean': neg_means[i],
                    'separation': separations[i],
                    'selected': selected[i]
                }
                for i, layer in enumerate(layer_names)
            }
        }

        self.stats['selective_steering'] = stats
        return stats

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