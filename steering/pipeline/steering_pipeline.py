"""
Complete Angular Steering Pipeline
"""

import torch
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from tqdm import tqdm
import logging
import math

from ..data import DataManager
from ..extraction import ActivationExtractor
from ..direction import FeatureDirectionCalculator
from ..plane import SteeringPlaneConstructor
from ..steering import AngularSteeringOperator, AdaptiveSteeringOperator, SelectiveSteeringOperator
from ..hooks import ModelHookManager
from ..artifacts import ArtifactsManager, ActivationAnalyzer
from ..utils.logger import setup_logger


class AngularSteeringPipeline:
    """
    Complete pipeline for Angular Steering
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        config: Dict[str, Any]
    ):
        """
        Initialize pipeline

        Args:
            model: Language model
            tokenizer: Model tokenizer
            config: Configuration dictionary
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        # Setup logger
        self.logger = setup_logger(obj=self)
        
        # Chat template settings
        chat_config = config.get('chat_template', {})
        self.use_chat_template = chat_config.get('enabled', False)
        self.system_prompt = chat_config.get('system_prompt', None)
        self.add_generation_prompt = chat_config.get('add_generation_prompt', True)

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.logger.info("[Warning] Tokenizer has no pad_token. Using eos_token as pad_token.")
            # If eos_token is also missing, add a default pad token
            if self.tokenizer.eos_token is None:
                self.logger.info("[Warning] Tokenizer also has no eos_token. Adding '[PAD]' as pad token.")
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            else:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize components
        self.data_manager = DataManager(
            tokenizer=tokenizer,
            use_chat_template=self.use_chat_template,
            system_prompt=self.system_prompt
        )
        self.activation_extractor = ActivationExtractor(
            model,
            tokenizer,
            device=config.get('device', 'cuda')
        )
        self.direction_calculator = FeatureDirectionCalculator(
            method=config.get('direction_method', 'diff_in_means')
        )

        # Artifacts management
        model_name = config['model']['name'].split('/')[-1]
        self.artifacts = ArtifactsManager(
            output_dir=config.get('artifacts_dir', 'artifacts'),
            name=model_name
        )
        self.analyzer = ActivationAnalyzer(
            output_dir=config.get('analysis_dir', 'analysis_results')
        )
        
        # These will be set during calibration
        self.plane_constructor: Optional[SteeringPlaneConstructor] = None
        self.steering_operator: Optional[AngularSteeringOperator] = None
        self.hook_manager: Optional[ModelHookManager] = None
        
        # State
        self.is_calibrated = False
        self.feature_direction = None
        self.best_layer = None
    
    def calibrate(
        self,
        harmful_path: Optional[str] = None,
        harmless_path: Optional[str] = None,
        harmful_samples: Optional[int] = None,
        harmless_samples: Optional[int] = None,
        save_artifacts: bool = True, 
        run_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        Calibrate the steering system
        
        Args:
            harmful_path: Path to harmful dataset
            harmless_path: Path to harmless dataset
            harmful_samples: Number of harmful samples
            harmless_samples: Number of harmless samples
            save_artifacts: Whether to save calibration artifacts
            run_analysis: Whether to run activation analysis
            
        Returns:
            Dictionary with calibration info
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting Angular Steering Calibration")
        self.logger.info("=" * 60)

        # Use paths from config if not provided
        harmful_path = harmful_path or self.config.get('harmful_dataset')
        harmless_path = harmless_path or self.config.get('harmless_dataset')

        if not harmful_path or not harmless_path:
            raise ValueError("Dataset paths must be provided")

        # Step 1: Load datasets
        self.logger.info("[1/6] Loading datasets...")
        harmful_prompts, harmless_prompts = self.data_manager.load_datasets(
            harmful_path,
            harmless_path,
            harmful_samples,
            harmless_samples
        )
        self.logger.info(f"  Loaded {len(harmful_prompts)} harmful prompts")
        self.logger.info(f"  Loaded {len(harmless_prompts)} harmless prompts")

        # Step 2: Extract activations
        self.logger.info("[2/6] Extracting activations...")
        extraction_layers = self._get_extraction_layers()
        self.activation_extractor.register_hooks(extraction_layers)

        self.logger.info(f"  Extracting from {len(extraction_layers)} layers...")
        harmful_acts = self.activation_extractor.extract_activations(
            harmful_prompts
        )
        harmless_acts = self.activation_extractor.extract_activations(
            harmless_prompts
        )
        self.logger.info("  ✓ Extracted activations")

        # Step 3: Compute candidate directions
        self.logger.info("[3/6] Computing feature directions...")
        candidates = self.direction_calculator.compute_candidate_directions(
            harmful_acts,
            harmless_acts
        )
        self.logger.info(f"  Computed {len(candidates)} candidate directions")

        # Step 4: Select best direction
        self.logger.info("[4/6] Selecting best feature direction...")
        self.feature_direction, self.best_layer = \
            self.direction_calculator.select_best_direction(candidates)
        self.logger.info(f"  ✓ Selected direction from layer: {self.best_layer}")

        # Step 5: Construct steering plane
        self.logger.info("[5/6] Constructing steering plane...")
        
        # Choose plane constructor based on config
        plane_method = self.config.get('plane_constructor', 'pca')  # 'pca' or 'grassmannian'
        
        if plane_method == 'grassmannian':
            from ..plane import GrassmannianPlaneConstructor
            
            # Get Grassmannian hyperparameters from config
            grassmann_config = self.config.get('grassmannian', {})
            
            self.plane_constructor = GrassmannianPlaneConstructor(
                # self.feature_direction,
                alpha=grassmann_config.get('alpha', 1.0),
                beta=grassmann_config.get('beta', 0.1),
                lr=grassmann_config.get('lr', 0.1),
                max_iterations=grassmann_config.get('max_iterations', 50),
                convergence_threshold=grassmann_config.get('convergence_threshold', 1e-4),
                use_geoopt=grassmann_config.get('use_geoopt', True),
                verbose=True
            )
            
            # Grassmannian optimization needs actual activations
            self.plane_constructor.construct_plane(
                self.feature_direction,
                candidates,
                harmful_acts,  # Pass activations for optimization
                harmless_acts
            )
            
            self.logger.info("  ✓ Optimized steering plane with Grassmannian method")
            
            # Log convergence info
            q = self.plane_constructor.measure_contraction_constant()
            if q is not None:
                self.logger.info(f"  Empirical contraction constant: q = {q:.4f}")
                if q < 1:
                    self.logger.info("  ✓ Convergence guarantee satisfied (q < 1)")
                else:
                    self.logger.warning("  ⚠ Contraction constant >= 1 (may not converge)")
        
        else:
            # Use original PCA-based constructor
            from ..plane import SteeringPlaneConstructor
            
            self.plane_constructor = SteeringPlaneConstructor(
                self.feature_direction
            )
            self.plane_constructor.construct_plane(
                self.feature_direction,
                candidates,
                # harmful_acts,  # Pass for compatibility
                # harmless_acts
            )
            self.logger.info("  ✓ Constructed 2D steering plane with PCA")
        
        b1, b2 = self.plane_constructor.get_basis()
        self.logger.info(f"  Basis vectors: b1.shape={b1.shape}, b2.shape={b2.shape}")
        self.logger.info("  ✓ Constructed 2D steering plane")

        # Store calibration data for later use by save_calibration_session()
        self._harmful_activations = harmful_acts
        self._harmless_activations = harmless_acts
        self._direction_candidates = candidates
        self._steering_basis = (b1, b2)

        # Step 6: Initialize steering components
        self.logger.info("[6/6] Initializing steering operator...")
        mode = self.config.get('steering', {}).get('mode', 'standard')

        if mode == 'adaptive':
            self.steering_operator = AdaptiveSteeringOperator(
                b1, b2,
                cache_rotations=True
            )
        elif mode == 'selective':
            # Get selective steering configuration
            steering_config = self.config.get('steering', {})

            # Create SelectiveSteeringOperator using activations
            self.logger.info(f"  Computing layer selection using '{mode}' method...")
            self.steering_operator = SelectiveSteeringOperator.from_activations(
                positive_activations=harmful_acts,
                negative_activations=harmless_acts,
                feature_direction=self.feature_direction,
                b1=b1,
                b2=b2,
                cache_rotations=True,
            )

            # Get selection info
            selected_layers = self.steering_operator.get_selected_layers()
            n_selected = len(selected_layers)
            n_total = len(harmful_acts)

            self.logger.info(f"  ✓ Selected {n_selected}/{n_total} layers for steering")

            # Compute and store projection stats for visualization
            self._projection_stats = SelectiveSteeringOperator.compute_layer_projection_stats(
                harmful_acts,
                harmless_acts,
                self.feature_direction
            )
            self._layer_steering_mask = self.steering_operator.layer_steering_mask
        else:
            self.steering_operator = AngularSteeringOperator(
                b1, b2,
                cache_rotations=True
            )

        # Initialize hook manager for transformers
        self.hook_manager = ModelHookManager(
            self.model,
            self.steering_operator
        )

        self.logger.info("  ✓ Initialized %s steering operator", mode)

        self.is_calibrated = True

        # Save artifacts and run analysis AFTER operator is initialized
        if save_artifacts or run_analysis:
            self.save_calibration_session(save_artifacts=save_artifacts, run_analysis=run_analysis)

        self.logger.info("=" * 60)
        self.logger.info("✓ Calibration Complete!")
        self.logger.info("=" * 60)

        return {
            'best_layer': self.best_layer,
            'n_candidates': len(candidates),
            'steering_mode': mode
        }
    
    def _format_prompts(
        self,
        prompts: Union[str, List[str]],
        system_prompt: Optional[str] = None
    ) -> List[str]:
        """
        Format prompts using chat template if enabled
        
        Args:
            prompts: Single prompt or list of prompts
            system_prompt: Optional system prompt override
            
        Returns:
            List of formatted prompts
        """
        # Convert single prompt to list
        if isinstance(prompts, str):
            prompts = [prompts]
        
        # Return as-is if chat template disabled
        if not self.use_chat_template:
            return prompts
        
        # Use DataManager's format_prompt for consistency
        formatted_prompts = [
            self.data_manager.format_prompt(
                p, 
                system_prompt=system_prompt or self.system_prompt,
                add_generation_prompt=self.add_generation_prompt
            )
            for p in prompts
        ]
        
        return formatted_prompts
    
    def steer_and_generate(
        self,
        prompts: Union[str, List[str]],
        theta: float,
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        use_chat_template: Optional[bool] = None,
        calculate_perplexity: bool = False,
        **generation_kwargs
    ) -> Union[List[str], List[Dict[str, Any]]]:
        """
        Generate text with steering

        Args:
            prompts: Input prompt(s) (string or list)
            theta: Steering angle in degrees
            max_length: Maximum total generation length (deprecated, use max_new_tokens)
            max_new_tokens: Maximum number of new tokens to generate
            system_prompt: Optional system prompt for chat template
            use_chat_template: Override global chat template setting
            calculate_perplexity: If True, calculate and return perplexity scores
            **generation_kwargs: Additional generation parameters

        Returns:
            If calculate_perplexity is False: List of generated texts
            If calculate_perplexity is True: List of dicts with 'response' and 'perplexity' keys
        """
        if not self.is_calibrated:
            raise RuntimeError(
                "Pipeline not calibrated. Call calibrate() first."
            )
        
        # Override chat template setting if specified
        original_use_chat = self.use_chat_template
        if use_chat_template is not None:
            self.use_chat_template = use_chat_template
        
        # Convert single prompt to list for tracking
        original_prompts = [prompts] if isinstance(prompts, str) else prompts
        
        # Format prompts
        formatted_prompts = self._format_prompts(prompts, system_prompt)
        
        # Restore original setting
        self.use_chat_template = original_use_chat
        
        # Generate
        results = self._generate_transformers(
            formatted_prompts,
            theta,
            original_prompts=original_prompts,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            calculate_perplexity=calculate_perplexity,
            **generation_kwargs
        )

        if not calculate_perplexity:
            return [res['response'] for res in results]

        return results
    
    def _generate_transformers(
        self,
        prompts: List[str],
        theta: float,
        original_prompts: Optional[List[str]] = None,
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        calculate_perplexity: bool = False,
        **generation_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate using transformers
        
        Args:
            prompts: Formatted prompts (may include system/special tokens)
            theta: Steering angle in degrees
            original_prompts: Original unformatted prompts (without system/special tokens)
            max_length: Maximum total length (deprecated)
            max_new_tokens: Maximum new tokens to generate
            calculate_perplexity: Whether to calculate perplexity
            **generation_kwargs: Additional generation parameters
        """
        # Register hooks with steering
        target_layers = self._get_target_layers()

        # Debug: Log target layers being hooked
        self.logger.debug(f"Hooking {len(target_layers)} layers for steering")
        self.logger.debug(f"First 3 layers: {target_layers[:3] if len(target_layers) > 0 else 'none'}")

        self.hook_manager.register_hooks(
            target_layers,
            steering_params={'theta': theta}
        )

        # Generate outputs
        outputs = []
        self.model.eval()

        with torch.no_grad():
            # Create progress bar
            pbar = tqdm(
                prompts, 
                desc=f"Generating (θ={theta}°)",
                unit="prompt",
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            )
            
            for idx, prompt in enumerate(pbar):
                # Get original unformatted prompt (without system/special tokens)
                if original_prompts and idx < len(original_prompts):
                    original_prompt = original_prompts[idx]
                else:
                    # Fallback: use formatted prompt if original not provided
                    original_prompt = prompt
                
                print(f"Prompt: {prompt}")
                # Tokenize prompt (this includes system prompt and special tokens if chat template was used)
                inputs = self.tokenizer(
                    prompt,
                    return_tensors='pt',
                    padding=True,
                    truncation=True
                ).to(self.model.device)
                
                # Calculate prompt length (includes system tokens, special tokens, and assistant start tokens)
                # This is the full input that goes to the model, including:
                # - System prompt tokens (if any)
                # - Chat template special tokens (e.g., <|im_start|>, <|im_end|>)
                # - User message tokens
                # - Assistant start tokens (if add_generation_prompt=True)
                prompt_length = inputs['input_ids'].shape[1]

                # Prepare generation parameters
                # Prefer max_new_tokens over max_length to avoid input length issues
                gen_kwargs = generation_kwargs.copy()
                if max_new_tokens is not None:
                    gen_kwargs['max_new_tokens'] = max_new_tokens
                elif max_length is not None:
                    gen_kwargs['max_length'] = max_length
                else:
                    # Default fallback
                    gen_kwargs['max_new_tokens'] = 100

                # Generate
                output_ids = self.model.generate(
                    **inputs,
                    **gen_kwargs
                )

                # Decode generated text (everything after prompt_length)
                output_text = self.tokenizer.decode(
                    output_ids[0][prompt_length:],
                    skip_special_tokens=True
                )

                # Store original prompt (without system/special tokens) and response
                result = {'prompt': original_prompt, 'response': output_text}

                if calculate_perplexity:
                    # Update progress bar for perplexity calculation
                    pbar.set_postfix_str("calculating perplexity")
                    
                    # Remove hooks for perplexity calculation (optional but recommended)
                    self.hook_manager.remove_hooks()
                    
                    # Calculate perplexity only on generated tokens
                    # Mask all prompt tokens including:
                    # - System prompt tokens (if any)
                    # - Chat template special tokens (<|im_start|>, <|im_end|>, etc.)
                    # - User message tokens
                    # - Assistant start tokens (if add_generation_prompt=True)
                    # All of these are part of the input and should not be included in perplexity
                    labels = output_ids.clone()
                    labels[:, :prompt_length] = -100  # -100 means ignore in loss calculation
                    
                    # Calculate loss with proper masking
                    model_outputs = self.model(
                        input_ids=output_ids,
                        labels=labels
                    )
                    loss = model_outputs.loss
                    
                    # Calculate perplexity
                    perplexity = torch.exp(loss).item()
                    result['perplexity'] = perplexity
                    
                    # Update progress bar with perplexity info
                    pbar.set_postfix_str(f"ppl={perplexity:.2f}")
                    
                    # Re-register hooks if more prompts to process
                    if prompt != prompts[-1]:
                        self.hook_manager.register_hooks(
                            target_layers,
                            steering_params={'theta': theta}
                        )
                else:
                    # Clear postfix if not calculating perplexity
                    pbar.set_postfix_str("")

                outputs.append(result)

        # Clean up
        self.hook_manager.remove_hooks()

        return outputs
    
    def _get_extraction_layers(self) -> List[str]:
        """Get layer names for activation extraction"""
        if 'extraction_layers' in self.config:
            return self.config['extraction_layers']
        
        # Auto-detect normalization layers
        return self.activation_extractor.get_normalization_layers()
    
    def _get_target_layers(self) -> List[str]:
        """Get layer names for steering"""
        if 'target_layers' in self.config:
            return self.config['target_layers']
        
        # Use same as extraction by default
        return self._get_extraction_layers()

    def save_calibration_session(self, save_artifacts: bool = True, run_analysis: bool = False) -> str:
        """
        Universal function to save all calibration artifacts and optionally run analysis.

        This consolidates all artifact saving operations into a single method that:
        - Uses consistent session-based timestamping
        - Eliminates duplication from the calibrate() method
        - Can be called standalone or during calibration
        - Optionally runs analysis with the same session data

        Args:
            save_artifacts: Whether to save calibration artifacts
            run_analysis: Whether to run analysis and visualization

        Returns:
            Path to the session directory containing all artifacts

        Raises:
            RuntimeError: If calibration hasn't been completed yet
        """
        if not hasattr(self, 'feature_direction') or self.feature_direction is None:
            raise RuntimeError("Cannot save calibration session - calibration not completed yet. Call calibrate() first.")

        model_name = self.config['model']['name'].split('/')[-1]
        session_path = None

        if save_artifacts:
            self.logger.info("[Saving Calibration Session]")

            # Get required data from calibration
            harmful_acts = getattr(self, '_harmful_activations', None)
            harmless_acts = getattr(self, '_harmless_activations', None)
            candidates = getattr(self, '_direction_candidates', None)
            b1, b2 = getattr(self, '_steering_basis', (None, None))

            if any(x is None for x in [harmful_acts, harmless_acts, candidates, b1, b2]):
                raise RuntimeError("Calibration data not available - ensure calibrate() was called with save_artifacts=True")

            # Calculate projections
            projections = self.plane_constructor.project_candidates_onto_plane(candidates)

            # Get layer steering mask if in selective mode
            mode = self.config.get('steering', {}).get('mode', 'standard')
            extra_info = {}
            if mode == 'selective' and hasattr(self.steering_operator, 'layer_steering_mask'):
                extra_info['layer_steering_mask'] = self.steering_operator.layer_steering_mask

            # Save all artifacts using the universal function
            session_path = self.artifacts.save_calibration_artifacts(
                harmful_acts, harmless_acts, candidates,
                self.feature_direction, self.best_layer,
                (b1, b2), projections, self.config,
                mode, extra_info
            )

        if run_analysis:
            self.logger.info("[Analysis & Visualization]")

            # Get required data for analysis
            harmful_acts = getattr(self, '_harmful_activations', None)
            harmless_acts = getattr(self, '_harmless_activations', None)
            candidates = getattr(self, '_direction_candidates', None)
            b1, b2 = getattr(self, '_steering_basis', (None, None))

            if any(x is None for x in [harmful_acts, harmless_acts, candidates, b1, b2]):
                raise RuntimeError("Calibration data not available for analysis")

            self.analyzer.analyze_all(
                harmful_acts, harmless_acts, candidates,
                self.feature_direction, (b1, b2), model_name
            )

            # Add selective steering visualization if in selective mode
            mode = self.config.get('steering', {}).get('mode', 'standard')
            if mode == 'selective':
                projection_stats = getattr(self, '_projection_stats', None)
                layer_mask = getattr(self, '_layer_steering_mask', None)

                if projection_stats is not None and layer_mask is not None:
                    self.logger.info("[Selective Steering Analysis]")
                    selection_method = getattr(self, '_selection_method', 'opposite_signs')
                    self.analyzer.plot_selective_layer_steering(
                        projection_stats=projection_stats,
                        layer_steering_mask=layer_mask,
                        save_name=model_name,
                        selection_method=selection_method
                    )

        return session_path or "No artifacts saved"

    def save_calibration(self, save_artifacts: bool = True, run_analysis: bool = False) -> None:
        """
        Save calibration state
        
        Args:
            save_artifacts: Whether to save calibration artifacts
            run_analysis: Whether to run analysis and visualization
        """
        if not self.is_calibrated:
            raise RuntimeError("No calibration to save")
        
        self.save_calibration_session(save_artifacts, run_analysis)
        self.logger.info(f"Calibration saved to {str(self.artifacts.session_dir)}")

    def load_calibration(self, bundle_dir: str) -> None:
        """
        Load complete calibration bundle
        """
        bundle = self.artifacts.load_calibration(bundle_dir)

        b1, b2 = bundle['plane']['basis']

        # Use mode from saved config, not current config
        saved_config = bundle.get('config', {})
        mode = saved_config.get('steering', {}).get('mode', 'standard')

        self.logger.info(f"Loading calibration with mode: {mode}")

        # Get extraction/target layers from saved calibration
        # This ensures we hook the same layers that were used during calibration
        saved_extraction_layers = saved_config.get('extraction_layers', None)
        saved_target_layers = saved_config.get('target_layers', None)

        # Override current config with saved layer configuration
        if saved_extraction_layers is not None:
            self.config['extraction_layers'] = saved_extraction_layers
        if saved_target_layers is not None:
            self.config['target_layers'] = saved_target_layers

        # For selective mode, use layers from layer_steering_mask if available
        extra_info = bundle['plane'].get('extra_info', {})
        layer_steering_mask = extra_info.get('layer_steering_mask', None)

        if mode == 'selective' and layer_steering_mask is not None:
            # Override target_layers to match the layers in the mask
            calibration_layers = list(layer_steering_mask.keys())
            self.config['target_layers'] = calibration_layers
            self.logger.info(f"  Using {len(calibration_layers)} layers from calibration")

            # Debug: Log which layers will be steered
            selected_layers = [name for name, should_steer in layer_steering_mask.items() if should_steer]
            self.logger.info(f"  Layers to be steered: {selected_layers[:3]}... ({len(selected_layers)} total)")

        # Initialize components based on saved mode
        if mode == 'adaptive':
            self.steering_operator = AdaptiveSteeringOperator(b1, b2, cache_rotations=True)
        elif mode == 'selective':
            if layer_steering_mask is None:
                self.logger.warning("Selective mode requested but no layer_steering_mask found in calibration. Falling back to standard mode.")
                self.steering_operator = AngularSteeringOperator(b1, b2, cache_rotations=True)
            else:
                self.steering_operator = SelectiveSteeringOperator(
                    b1, b2,
                    layer_steering_mask=layer_steering_mask,
                    cache_rotations=True
                )
                n_selected = sum(layer_steering_mask.values())
                n_total = len(layer_steering_mask)
                self.logger.info(f"  ✓ Loaded selective steering with {n_selected}/{n_total} layers selected")
        else:
            self.steering_operator = AngularSteeringOperator(b1, b2, cache_rotations=True)

        self.hook_manager = ModelHookManager(
            self.model,
            self.steering_operator
        )

        self.is_calibrated = True
        self.logger.info(f"Calibration loaded from {bundle_dir}")