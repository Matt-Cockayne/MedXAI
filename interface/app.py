"""
Interactive Gradio interface for MedMNIST explainability demonstrations.

Three pre-configured pages for:
1. DermaMNIST - Skin lesion classification
2. PneumoniaMNIST - Pneumonia detection  
3. ChestMNIST - Thoracic disease classification
"""

import gradio as gr
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import sys
from pathlib import Path
import torchvision.transforms as transforms

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from explainers import (
    GradCAM, GradCAMPlusPlus, IntegratedGradients, RISE
)
from metrics import DeletionInsertion, FaithfulnessMetrics
from utils import (
    load_model, get_medical_dataset, visualize_comparison,
    overlay_heatmap
)


class MedMNISTExplainabilityApp:
    """Interactive application for MedMNIST explainability demonstrations."""
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.datasets = {}
        self.models = {}
        self.explainers = {}
        self.data_dir = Path('./data')
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def load_dataset(self, dataset_name: str, num_classes: int, is_grayscale: bool = False):
        """Load MedMNIST dataset and initialize model."""
        try:
            # Import medmnist
            import medmnist
            
            # Custom transform for datasets
            if is_grayscale:
                transform = transforms.Compose([
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
            else:
                transform = None  # Use default
            
            # Load dataset
            dataset = get_medical_dataset(
                dataset_name,
                root=str(self.data_dir),
                split='test',
                transform=transform,
                download=True
            )
            
            self.datasets[dataset_name] = dataset
            
            # Load model
            model = load_model('resnet50', num_classes=num_classes, device=self.device)
            self.models[dataset_name] = model
            
            # Initialize explainers
            self.explainers[dataset_name] = {
                'GradCAM': GradCAM(model, 'layer4', self.device),
                'GradCAM++': GradCAMPlusPlus(model, 'layer4', self.device),
                'Integrated Gradients': IntegratedGradients(model, self.device),
                'RISE': RISE(model, self.device, n_masks=1000)
            }
            
            return f"Loaded {dataset_name}: {len(dataset)} test images, {num_classes} classes"
        except Exception as e:
            return f"Error loading {dataset_name}: {str(e)}"
    
    def load_sample(self, dataset_name: str, sample_idx: int, class_names: List[str]) -> Tuple:
        """Load a sample from dataset."""
        try:
            if dataset_name not in self.datasets:
                return None, "Dataset not loaded", None, None
            
            dataset = self.datasets[dataset_name]
            model = self.models[dataset_name]
            
            # Get sample
            image, label = dataset[sample_idx]
            
            # Handle multi-label (ChestMNIST)
            if isinstance(label, np.ndarray) and len(label.shape) > 0 and len(label) > 1:
                positive_labels = np.where(label == 1)[0]
                if len(positive_labels) > 0:
                    label = int(positive_labels[0])
                else:
                    label = 0
            else:
                # Convert numpy array to scalar properly
                label = int(label.item()) if isinstance(label, np.ndarray) else int(label)
            
            # Get original low-res image
            import medmnist
            from medmnist import INFO
            info = INFO[dataset_name]
            DataClass = getattr(medmnist, info['python_class'])
            original_dataset = DataClass(split='test', download=False, root=str(self.data_dir), transform=None)
            original_image = original_dataset[sample_idx][0]
            
            # Get prediction
            image_batch = image.unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = model(image_batch)
                probs = torch.nn.functional.softmax(output, dim=1)[0]
                pred_class = output.argmax(dim=1).item()
                confidence = probs[pred_class].item()
            
            true_label = class_names[label] if label < len(class_names) else f"Class {label}"
            pred_label = class_names[pred_class] if pred_class < len(class_names) else f"Class {pred_class}"
            
            info_text = f"Sample #{sample_idx}\nTrue: {true_label}\nPredicted: {pred_label} ({confidence:.2%})"
            
            return original_image, info_text, image_batch, label
            
        except Exception as e:
            return None, f"Error: {str(e)}", None, None
    
    def generate_explanations(
        self,
        dataset_name: str,
        image_batch: torch.Tensor,
        label: int,
        methods: List[str]
    ) -> List[Tuple[Image.Image, str]]:
        """Generate explanations and metrics."""
        try:
            if dataset_name not in self.explainers:
                return [(None, "Dataset not loaded")] * 4
            
            model = self.models[dataset_name]
            explainers = self.explainers[dataset_name]
            
            results = []
            di_metric = DeletionInsertion(model, self.device, n_steps=30)
            
            for method_name in methods[:4]:  # Max 4 methods
                try:
                    explainer = explainers[method_name]
                    
                    # Generate explanation
                    if method_name == 'RISE':
                        heatmap = explainer.explain(image_batch, label, n_masks=500)
                    else:
                        heatmap = explainer.explain(image_batch, label)
                    
                    # Ensure heatmap is detached and on CPU
                    if isinstance(heatmap, torch.Tensor):
                        heatmap = heatmap.detach().cpu()
                    
                    # Overlay on image (detach to avoid gradient issues)
                    overlaid = overlay_heatmap(image_batch.detach(), heatmap, alpha=0.5)
                    overlaid_uint8 = (overlaid * 255).astype(np.uint8)
                    img = Image.fromarray(overlaid_uint8)
                    
                    # Compute metrics
                    result = di_metric.evaluate(image_batch, heatmap, label)
                    metrics_text = f"{method_name}\nDel: {result['deletion_auc']:.3f}\nIns: {result['insertion_auc']:.3f}"
                    
                    results.append((img, metrics_text))
                except Exception as e:
                    results.append((None, f"{method_name}\nError: {str(e)}"))
            
            # Pad to 4 results
            while len(results) < 4:
                results.append((None, ""))
            
            return results
            
        except Exception as e:
            error_result = (None, f"Error: {str(e)}")
            return [error_result] * 4
    



def create_dataset_tab(app, dataset_name: str, num_classes: int, class_names: List[str], 
                       is_grayscale: bool = False, description: str = ""):
    """Create a tab for a specific MedMNIST dataset."""
    
    # State variables
    current_image_batch = gr.State(None)
    current_label = gr.State(None)
    
    with gr.Column():
        gr.Markdown(f"### {description}")
        
        # Load dataset button
        with gr.Row():
            load_btn = gr.Button(f"Load {dataset_name.upper()} Dataset", variant="primary", scale=2)
            status_text = gr.Textbox(label="Status", scale=3, interactive=False)
        
        load_btn.click(
            fn=lambda: app.load_dataset(dataset_name, num_classes, is_grayscale),
            outputs=[status_text]
        )
        
        # Sample selection
        with gr.Row():
            sample_slider = gr.Slider(
                minimum=0, maximum=1000, step=1, value=42,
                label="Select Sample Index"
            )
            load_sample_btn = gr.Button("Load Sample", variant="secondary")
        
        # Display sample and info
        with gr.Row():
            sample_image = gr.Image(label="Original Image (28x28)", type="pil", height=200)
            sample_info = gr.Textbox(label="Sample Information", lines=4)
        
        # Explainability methods
        gr.Markdown("### Generate Explanations")
        method_checkboxes = gr.CheckboxGroup(
            choices=['GradCAM', 'GradCAM++', 'Integrated Gradients', 'RISE'],
            value=['GradCAM', 'GradCAM++', 'Integrated Gradients', 'RISE'],
            label="Select Methods (max 4)"
        )
        
        explain_btn = gr.Button("Generate Explanations & Metrics", variant="primary")
        
        # Results display
        with gr.Row():
            with gr.Column():
                result1_img = gr.Image(label="Explanation 1", type="pil")
                result1_text = gr.Textbox(label="Metrics", lines=3)
            with gr.Column():
                result2_img = gr.Image(label="Explanation 2", type="pil")
                result2_text = gr.Textbox(label="Metrics", lines=3)
        
        with gr.Row():
            with gr.Column():
                result3_img = gr.Image(label="Explanation 3", type="pil")
                result3_text = gr.Textbox(label="Metrics", lines=3)
            with gr.Column():
                result4_img = gr.Image(label="Explanation 4", type="pil")
                result4_text = gr.Textbox(label="Metrics", lines=3)
        
        # Wire up callbacks
        def load_sample_wrapper(idx):
            img, info, batch, lbl = app.load_sample(dataset_name, int(idx), class_names)
            return img, info, batch, lbl
        
        load_sample_btn.click(
            fn=load_sample_wrapper,
            inputs=[sample_slider],
            outputs=[sample_image, sample_info, current_image_batch, current_label]
        )
        
        def explain_wrapper(batch, label, methods):
            if batch is None:
                empty = (None, "Load a sample first")
                return [empty] * 8
            results = app.generate_explanations(dataset_name, batch, label, methods)
            # Unpack tuples for outputs
            outputs = []
            for img, text in results:
                outputs.extend([img, text])
            return outputs
        
        explain_btn.click(
            fn=explain_wrapper,
            inputs=[current_image_batch, current_label, method_checkboxes],
            outputs=[
                result1_img, result1_text, result2_img, result2_text,
                result3_img, result3_text, result4_img, result4_text
            ]
        )


def create_interface():
    """Create the Gradio interface with three MedMNIST dataset tabs."""
    app = MedMNISTExplainabilityApp()
    
    with gr.Blocks(title="MedMNIST Explainability Toolkit") as demo:
        gr.Markdown("""
        # Explainable AI for Medical Imaging
        ### Interactive MedMNIST Explainability Demonstrations
        
        Explore three medical imaging datasets with comprehensive explainability analysis.
        Each tab demonstrates XAI methods on a different MedMNIST dataset.
        """)
        
        with gr.Tab("DermaMNIST - Skin Lesions"):
            create_dataset_tab(
                app,
                dataset_name='dermamnist',
                num_classes=7,
                class_names=[
                    'Actinic Keratoses', 'Basal Cell Carcinoma', 'Benign Keratosis',
                    'Dermatofibroma', 'Melanoma', 'Melanocytic Nevi', 'Vascular Lesions'
                ],
                is_grayscale=False,
                description="**7-class skin lesion classification from dermatoscopic images**"
            )
        
        with gr.Tab("PneumoniaMNIST - Chest X-rays"):
            create_dataset_tab(
                app,
                dataset_name='pneumoniamnist',
                num_classes=2,
                class_names=['Normal', 'Pneumonia'],
                is_grayscale=True,
                description="**Binary pneumonia detection from pediatric chest X-rays**"
            )
        
        with gr.Tab("ChestMNIST - Thoracic Diseases"):
            create_dataset_tab(
                app,
                dataset_name='chestmnist',
                num_classes=14,
                class_names=[
                    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
                    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
                    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
                    'Pleural Thickening', 'Hernia'
                ],
                is_grayscale=True,
                description="**14-class thoracic disease classification from NIH ChestX-ray14**"
            )
        
        with gr.Tab("About"):
            gr.Markdown("""
            ## About This Tool
            
            Interactive demonstration of explainability methods on MedMNIST datasets.
            
            ### Datasets
            
            - **DermaMNIST**: 7 skin lesion types from HAM10000 dataset
            - **PneumoniaMNIST**: Binary pneumonia detection
            - **ChestMNIST**: 14 thoracic diseases from NIH ChestX-ray14
            
            ### Explainability Methods
            
            - **GradCAM**: Gradient-weighted Class Activation Mapping
            - **GradCAM++**: Improved pixel-wise weighting
            - **Integrated Gradients**: Path-based attribution
            - **RISE**: Randomized Input Sampling for Explanation
            
            ### Metrics
            
            - **Deletion AUC**: Lower is better (explanation captures important features)
            - **Insertion AUC**: Higher is better (explanation is sufficient)
            
            ### Usage
            
            1. Click "Load Dataset" to download and initialize
            2. Select a sample using the slider
            3. Click "Load Sample" to view the image
            4. Choose explainability methods (up to 4)
            5. Click "Generate Explanations & Metrics"
            
            ### Citation
            
            ```bibtex
            @software{explainable_ai_toolkit,
              author = {Matthew Cockayne},
              title = {Explainable-AI: Medical Imaging Explainability Toolkit},
              year = {2025},
              url = {https://github.com/Matt-Cockayne/Explainable-AI}
            }
            ```
            
            ### References
            
            MedMNIST: Yang et al., "MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification", Scientific Data, 2023
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
