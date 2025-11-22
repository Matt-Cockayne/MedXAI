"""
Interactive Gradio interface for comparing explainability methods.
"""

import gradio as gr
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from explainers import (
    GradCAM, GradCAMPlusPlus, IntegratedGradients,
    RISE, Occlusion
)
from metrics import (
    PointingGame, DeletionInsertion,
    FaithfulnessMetrics, PlausibilityMetrics
)
from utils import (
    load_model, get_target_layer_name, prepare_input,
    overlay_heatmap, get_default_transforms
)


class ExplainabilityApp:
    """Interactive application for comparing explainability methods."""
    
    def __init__(self):
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.explainers = {}
        self.current_image = None
        self.current_class = None
        
    def load_model_fn(self, model_name: str):
        """Load a pre-trained model."""
        try:
            self.model = load_model(model_name, pretrained=True, device=self.device)
            target_layer = get_target_layer_name(self.model, model_name)
            
            # Initialize explainers
            self.explainers = {
                'GradCAM': GradCAM(self.model, target_layer, self.device),
                'GradCAM++': GradCAMPlusPlus(self.model, target_layer, self.device),
                'Integrated Gradients': IntegratedGradients(self.model, self.device),
                'RISE': RISE(self.model, self.device, n_masks=1000),
                'Occlusion': Occlusion(self.model, self.device)
            }
            
            return f"✅ Model {model_name} loaded successfully!"
        except Exception as e:
            return f"❌ Error loading model: {str(e)}"
    
    def predict(self, image: Image.Image) -> tuple:
        """Get model prediction for an image."""
        if self.model is None:
            return "Please load a model first", None
        
        # Preprocess image
        input_tensor = prepare_input(image, device=self.device)
        self.current_image = input_tensor
        
        # Get prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)[0]
            top5_probs, top5_indices = torch.topk(probs, 5)
        
        # Format results
        results = []
        for prob, idx in zip(top5_probs, top5_indices):
            results.append(f"Class {idx}: {prob:.3f}")
        
        self.current_class = int(top5_indices[0])
        
        return "\n".join(results), int(top5_indices[0])
    
    def generate_explanations(
        self,
        image: Image.Image,
        target_class: Optional[int],
        methods: List[str],
        progress=gr.Progress()
    ) -> List[Image.Image]:
        """Generate explanations using selected methods."""
        if self.model is None:
            return [None] * len(methods)
        
        if target_class is None:
            target_class = self.current_class
        
        # Preprocess image
        input_tensor = prepare_input(image, device=self.device)
        
        results = []
        
        for i, method_name in enumerate(progress.tqdm(methods, desc="Generating explanations")):
            try:
                explainer = self.explainers[method_name]
                
                # Generate explanation
                if method_name == 'RISE':
                    # RISE is slower, use fewer masks for demo
                    heatmap = explainer.explain(input_tensor, target_class, n_masks=500)
                else:
                    heatmap = explainer.explain(input_tensor, target_class)
                
                # Overlay on original image
                overlaid = overlay_heatmap(input_tensor, heatmap, alpha=0.5)
                
                # Convert to PIL Image
                overlaid_uint8 = (overlaid * 255).astype(np.uint8)
                results.append(Image.fromarray(overlaid_uint8))
            except Exception as e:
                print(f"Error with {method_name}: {e}")
                results.append(None)
        
        return results
    
    def evaluate_explanations(
        self,
        image: Image.Image,
        target_class: Optional[int],
        methods: List[str],
        ground_truth_mask: Optional[Image.Image] = None
    ) -> str:
        """Evaluate explanations with metrics."""
        if self.model is None:
            return "Please load a model first"
        
        if target_class is None:
            target_class = self.current_class
        
        input_tensor = prepare_input(image, device=self.device)
        
        # Initialize metrics
        di_metric = DeletionInsertion(self.model, self.device)
        faith_metric = FaithfulnessMetrics(self.model, self.device)
        
        results_text = f"**Evaluation Results (Target Class: {target_class})**\n\n"
        
        for method_name in methods:
            try:
                explainer = self.explainers[method_name]
                heatmap = explainer.explain(input_tensor, target_class)
                
                # Compute metrics
                del_auc = di_metric.deletion_score(input_tensor, heatmap, target_class)
                ins_auc = di_metric.insertion_score(input_tensor, heatmap, target_class)
                
                results_text += f"### {method_name}\n"
                results_text += f"- Deletion AUC: {del_auc:.3f} (lower is better)\n"
                results_text += f"- Insertion AUC: {ins_auc:.3f} (higher is better)\n"
                
                # Add plausibility metrics if ground truth provided
                if ground_truth_mask is not None:
                    gt_tensor = torch.from_numpy(
                        np.array(ground_truth_mask.convert('L'))
                    ).float() / 255.0
                    
                    plaus_metric = PlausibilityMetrics()
                    iou = plaus_metric.iou_score(heatmap, gt_tensor)
                    results_text += f"- IoU with ground truth: {iou:.3f}\n"
                
                results_text += "\n"
            except Exception as e:
                results_text += f"### {method_name}\n❌ Error: {str(e)}\n\n"
        
        return results_text


def create_interface():
    """Create the Gradio interface."""
    app = ExplainabilityApp()
    
    with gr.Blocks(title="Explainable AI Toolkit") as demo:
        gr.Markdown("""
        # 🔍 Explainable AI Toolkit
        ### Interactive Medical Imaging Explainability Comparison
        
        Compare different explainability methods for deep learning models in medical imaging.
        """)
        
        with gr.Tab("Setup & Prediction"):
            with gr.Row():
                with gr.Column():
                    model_dropdown = gr.Dropdown(
                        choices=['resnet50', 'resnet18', 'densenet121', 'efficientnet_b0'],
                        label="Select Model",
                        value='resnet50'
                    )
                    load_btn = gr.Button("Load Model", variant="primary")
                    model_status = gr.Textbox(label="Status", interactive=False)
                
                with gr.Column():
                    input_image = gr.Image(type="pil", label="Upload Image")
                    predict_btn = gr.Button("Get Prediction", variant="primary")
                    prediction_output = gr.Textbox(label="Top 5 Predictions", lines=5)
                    predicted_class = gr.Number(label="Predicted Class (for explanation)", precision=0)
            
            load_btn.click(
                fn=app.load_model_fn,
                inputs=[model_dropdown],
                outputs=[model_status]
            )
            
            predict_btn.click(
                fn=app.predict,
                inputs=[input_image],
                outputs=[prediction_output, predicted_class]
            )
        
        with gr.Tab("Generate Explanations"):
            gr.Markdown("### Select methods to compare")
            
            method_checkboxes = gr.CheckboxGroup(
                choices=['GradCAM', 'GradCAM++', 'Integrated Gradients', 'RISE', 'Occlusion'],
                value=['GradCAM', 'GradCAM++', 'Integrated Gradients'],
                label="Explainability Methods"
            )
            
            target_class_input = gr.Number(
                label="Target Class (leave empty for predicted class)",
                precision=0,
                value=None
            )
            
            generate_btn = gr.Button("Generate Explanations", variant="primary")
            
            with gr.Row():
                output1 = gr.Image(label="Method 1")
                output2 = gr.Image(label="Method 2")
                output3 = gr.Image(label="Method 3")
            
            with gr.Row():
                output4 = gr.Image(label="Method 4")
                output5 = gr.Image(label="Method 5")
            
            generate_btn.click(
                fn=app.generate_explanations,
                inputs=[input_image, target_class_input, method_checkboxes],
                outputs=[output1, output2, output3, output4, output5]
            )
        
        with gr.Tab("Quantitative Evaluation"):
            gr.Markdown("### Evaluate explanation faithfulness")
            
            eval_methods = gr.CheckboxGroup(
                choices=['GradCAM', 'GradCAM++', 'Integrated Gradients', 'RISE', 'Occlusion'],
                value=['GradCAM', 'GradCAM++'],
                label="Methods to Evaluate"
            )
            
            ground_truth = gr.Image(
                type="pil",
                label="Ground Truth Mask (optional, for plausibility metrics)"
            )
            
            eval_btn = gr.Button("Run Evaluation", variant="primary")
            eval_results = gr.Markdown(label="Evaluation Results")
            
            eval_btn.click(
                fn=app.evaluate_explanations,
                inputs=[input_image, target_class_input, eval_methods, ground_truth],
                outputs=[eval_results]
            )
        
        with gr.Tab("About"):
            gr.Markdown("""
            ## About This Tool
            
            This interactive tool allows you to compare various explainability methods for deep learning models,
            with a focus on medical imaging applications.
            
            ### Implemented Methods
            
            #### Gradient-based
            - **GradCAM**: Class Activation Mapping using gradients
            - **GradCAM++**: Improved weighted class activation mapping
            - **Integrated Gradients**: Path-based attribution method
            
            #### Perturbation-based
            - **RISE**: Randomized Input Sampling for Explanation
            - **Occlusion**: Sliding window occlusion sensitivity analysis
            
            ### Evaluation Metrics
            
            - **Deletion AUC**: Measures drop in confidence as important pixels are removed (lower is better)
            - **Insertion AUC**: Measures rise in confidence as important pixels are added (higher is better)
            - **IoU**: Intersection over Union with ground truth annotations
            
            ### Usage
            
            1. **Setup**: Load a pre-trained model
            2. **Prediction**: Upload an image and get model predictions
            3. **Explanation**: Select methods and generate visual explanations
            4. **Evaluation**: Quantitatively evaluate explanation quality
            
            ### Citation
            
            If you use this toolkit, please cite:
            ```
            @software{explainable_ai_toolkit,
              author = {Matthew Cockayne},
              title = {Explainable-AI: Comprehensive Medical Imaging Explainability Toolkit},
              year = {2025},
              url = {https://github.com/Matt-Cockayne/Explainable-AI}
            }
            ```
            """)
    
    return demo


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Launch Explainability Interface')
    parser.add_argument('--share', action='store_true', 
                        help='Create a public shareable link (for remote/headless environments)')
    parser.add_argument('--port', type=int, default=7860,
                        help='Port to run on (default: 7860)')
    parser.add_argument('--server', type=str, default="0.0.0.0",
                        help='Server address (default: 0.0.0.0)')
    args = parser.parse_args()
    
    demo = create_interface()
    
    # For headless/remote: use share=True to get a public URL
    # For local: use share=False
    demo.launch(
        share=args.share,
        server_name=args.server,
        server_port=args.port,
        show_error=True
    )
