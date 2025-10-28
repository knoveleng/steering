"""
Complete Angular Steering Pipeline
"""

import torch
from typing import Dict, List, Optional, Any, Tuple, Literal, Union
from pathlib import Path
from tqdm import tqdm
import logging
import math

from ..data import DataManager
from ..extraction import ActivationExtractor
from ..direction import FeatureDirectionCalculator
from ..plane import SteeringPlaneConstructor
from ..steering import AngularSteeringOperator, AdaptiveSteeringOperator
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
        config: Dict[str, Any],
        backend: Literal["transformers", "vllm"] = "transformers"
    ):
        """
        Initialize pipeline

        Args:
            model: Language model (ignored if backend='vllm')
            tokenizer: Model tokenizer
            config: Configuration dictionary
            backend: Generation backend ('transformers' or 'vllm')
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.backend = backend

        # Setup logger
        self.logger = setup_logger(obj=self)
        
        # Chat template settings
        chat_config = config.get('chat_template', {})
        self.use_chat_template = chat_config.get('enabled', False)
        self.system_prompt = chat_config.get('system_prompt', None)
        self.add_generation_prompt = chat_config.get('add_generation_prompt', True)
        
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
        self.vllm_server = None
        

        
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

        # Use universal save function to eliminate duplication
        if save_artifacts or run_analysis:
            self.save_calibration_session(save_artifacts=save_artifacts, run_analysis=run_analysis)

        # Step 6: Initialize steering components
        self.logger.info("[6/6] Initializing steering operator...")
        mode = self.config.get('steering_mode', 'adaptive')

        if mode == 'adaptive':
            self.steering_operator = AdaptiveSteeringOperator(
                b1, b2,
                cache_rotations=True
            )
        else:
            self.steering_operator = AngularSteeringOperator(
                b1, b2,
                cache_rotations=True
            )

        # Initialize backend-specific components
        if self.backend == "vllm":
            self._initialize_vllm()
        else:
            # Initialize hook manager for transformers
            self.hook_manager = ModelHookManager(
                self.model,
                self.steering_operator
            )

        self.logger.info(f"  ✓ Initialized {mode} steering operator with {self.backend} backend")

        self.is_calibrated = True

        self.logger.info("=" * 60)
        self.logger.info("✓ Calibration Complete!")
        self.logger.info("=" * 60)
        
        return {
            'best_layer': self.best_layer,
            'n_candidates': len(candidates),
            'steering_mode': mode,
            'backend': self.backend
        }
    
    def _initialize_vllm(self):
        """Initialize vLLM server with steering"""
        from ..serving import VLLMSteeringServer

        self.logger.info("  Initializing vLLM server...")

        target_layers = self._get_target_layers()

        # Get vLLM config
        vllm_config = self.config.get('vllm', {})

        self.vllm_server = VLLMSteeringServer(
            model_name=self.config['model']['name'],
            steering_operator=self.steering_operator,
            target_layers=target_layers,
            tensor_parallel_size=vllm_config.get('tensor_parallel_size', 1),
            gpu_memory_utilization=vllm_config.get('gpu_memory_utilization', 0.9),
            dtype=self.config['model'].get('dtype', 'bfloat16'),
        )

        # Clean up original model to free memory
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()
    
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
        max_length: int = 100,
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
            max_length: Maximum generation length
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
        
        # Format prompts
        formatted_prompts = self._format_prompts(prompts, system_prompt)
        
        # Restore original setting
        self.use_chat_template = original_use_chat
        
        # Generate based on backend
        if self.backend == "vllm":
            results = self._generate_vllm(formatted_prompts, theta, max_length, calculate_perplexity, **generation_kwargs)
        else:
            results = self._generate_transformers(formatted_prompts, theta, max_length, calculate_perplexity, **generation_kwargs)

        if not calculate_perplexity:
            return [res['response'] for res in results]

        return results
    
    def _generate_vllm(
        self,
        prompts: List[str],
        theta: float,
        max_length: int = 100,
        calculate_perplexity: bool = False,
        **generation_kwargs
    ) -> List[Dict[str, str]]:
        """Generate using vLLM"""
        if calculate_perplexity:
            raise NotImplementedError(
                "Perplexity calculation is not supported for the vLLM backend."
            )
        # Set steering parameters
        self.vllm_server.set_steering(theta)
        self.vllm_server.enable_steering()

        # Generate
        outputs = self.vllm_server.generate(
            prompts,
            max_tokens=max_length,
            temperature=generation_kwargs.get('temperature', 0.7),
            top_p=generation_kwargs.get('top_p', 0.9),
        )

        # Return as list of dictionaries with prompt and response
        return [
            {'prompt': prompt, 'response': output}
            for prompt, output in zip(prompts, outputs)
        ]
    
    def _generate_transformers(
        self,
        prompts: List[str],
        theta: float,
        max_length: int = 100,
        calculate_perplexity: bool = False,
        **generation_kwargs
    ) -> List[Dict[str, Any]]:
        """Generate using transformers"""
        # Register hooks with steering
        target_layers = self._get_target_layers()
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
            
            for prompt in pbar:
                # Tokenize prompt
                inputs = self.tokenizer(
                    prompt,
                    return_tensors='pt',
                    padding=True,
                    truncation=True
                ).to(self.model.device)
                
                prompt_length = inputs['input_ids'].shape[1]

                # Generate
                output_ids = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    **generation_kwargs
                )

                # Decode
                output_text = self.tokenizer.decode(
                    output_ids[0][prompt_length:],
                    skip_special_tokens=True
                )

                result = {'prompt': prompt, 'response': output_text}

                if calculate_perplexity:
                    # Update progress bar for perplexity calculation
                    pbar.set_postfix_str("calculating perplexity")
                    
                    # Remove hooks for perplexity calculation (optional but recommended)
                    self.hook_manager.remove_hooks()
                    
                    # Calculate perplexity only on generated tokens
                    labels = output_ids.clone()
                    # Mask prompt tokens (set to -100 so they're ignored in loss)
                    labels[:, :prompt_length] = -100
                    
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
    
    def evaluate_steering(
        self,
        eval_prompts: List[str],
        theta_range: List[float],
        max_length: int = 100,
        system_prompt: Optional[str] = None
    ) -> Dict[float, Dict[str, Any]]:
        """
        Evaluate steering across multiple angles
        
        Args:
            eval_prompts: Prompts for evaluation
            theta_range: List of angles to evaluate
            max_length: Generation length
            system_prompt: Optional system prompt
            
        Returns:
            Dict mapping theta to evaluation results
        """
        results = {}
        
        for theta in theta_range:
            self.logger.info(f"Evaluating at θ = {theta}°...")

            # Generate with steering
            outputs = self.steer_and_generate(
                eval_prompts,
                theta,
                max_length=max_length,
                system_prompt=system_prompt
            )

            # Evaluate
            metrics = self.evaluator.evaluate_all(
                outputs,
                compute_perplexity=False  # Can be slow
            )

            results[theta] = {
                'metrics': metrics,
                'sample_outputs': outputs[:3]  # Store a few samples
            }

            self.logger.info(f"  Refusal Score: {metrics['refusal_score']:.3f}")
        
        return results
    
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

            # Save all artifacts using the universal function
            session_path = self.artifacts.save_calibration_artifacts(
                harmful_acts, harmless_acts, candidates,
                self.feature_direction, self.best_layer,
                (b1, b2), projections, self.config,
                self.config.get('steering_mode', 'adaptive')
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

        # Initialize components
        mode = self.config.get('steering_mode', 'adaptive')

        if mode == 'adaptive':
            self.steering_operator = AdaptiveSteeringOperator(b1, b2)
        else:
            self.steering_operator = AngularSteeringOperator(b1, b2)

        if self.backend == "vllm":
            self._initialize_vllm()
        else:
            self.hook_manager = ModelHookManager(
                self.model,
                self.steering_operator
            )

        self.is_calibrated = True
        self.logger.info(f"Calibration loaded from {bundle_dir}")