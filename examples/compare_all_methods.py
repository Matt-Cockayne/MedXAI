"""
Example script demonstrating comparison of all explainability methods.
"""

import torch
import matplotlib.pyplot as plt
from torchvision import models
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import explainability toolkit
from explainers import (
    GradCAM, GradCAMPlusPlus, IntegratedGradients,
    RISE, Occlusion
)
from metrics import DeletionInsertion, FaithfulnessMetrics, PlausibilityMetrics
from utils import (
    load_model, get_target_layer_name, load_image,
    visualize_comparison, plot_deletion_insertion_curves
)


def main():
    # Setup
    print("🚀 Initializing Explainable AI Toolkit Demo")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print("\nLoading model...")
    model_name = 'resnet50'
    model = load_model(model_name, pretrained=True, device=device)
    target_layer = get_target_layer_name(model, model_name)
    print(f"Model loaded: {model_name}")
    print(f"Target layer for GradCAM: {target_layer}")
    
    # Load image (you'll need to provide an image path)
    image_path = 'path/to/your/image.jpg'  # Update this!
    print(f"\n🖼️  Loading image: {image_path}")
    
    try:
        image = load_image(image_path)
    except FileNotFoundError:
        print("Image not found. Please update the image_path variable.")
        print("Using a random tensor for demonstration...")
        image = torch.randn(3, 224, 224)
    
    image = image.unsqueeze(0).to(device)
    
    # Get prediction
    print("\nGetting model prediction...")
    with torch.no_grad():
        output = model(image)
        probs = torch.nn.functional.softmax(output, dim=1)
        top5_probs, top5_indices = torch.topk(probs[0], 5)
    
    print("\nTop 5 Predictions:")
    for i, (prob, idx) in enumerate(zip(top5_probs, top5_indices)):
        print(f"  {i+1}. Class {idx}: {prob:.4f}")
    
    target_class = int(top5_indices[0])
    print(f"\n🎯 Using target class: {target_class}")
    
    # Initialize explainers
    print("\nInitializing explainers...")
    explainers = {
        'GradCAM': GradCAM(model, target_layer, device),
        'GradCAM++': GradCAMPlusPlus(model, target_layer, device),
        'Integrated Gradients': IntegratedGradients(model, device),
        'RISE': RISE(model, device, n_masks=2000),
        'Occlusion': Occlusion(model, device)
    }
    print(f"Initialized {len(explainers)} explainability methods")
    
    # Generate explanations
    print("\nGenerating explanations...")
    explanations = {}
    
    for method_name, explainer in explainers.items():
        print(f"  - {method_name}...", end=' ')
        try:
            explanation = explainer.explain(image, target_class)
            explanations[method_name] = explanation
            print("")
        except Exception as e:
            print(f"Error: {e}")
    
    # Visualize comparisons
    print("\nCreating visualizations...")
    fig = visualize_comparison(
        image,
        explanations,
        save_path='comparison.png'
    )
    print("Saved comparison to 'comparison.png'")
    plt.close(fig)
    
    # Evaluate with metrics
    print("\nComputing evaluation metrics...")
    
    # Deletion/Insertion curves
    di_metric = DeletionInsertion(model, device, n_steps=50)
    di_results = {}
    
    for method_name, heatmap in explanations.items():
        print(f"  - Evaluating {method_name}...", end=' ')
        try:
            result = di_metric.evaluate(image, heatmap, target_class)
            di_results[method_name] = result
            print(f"Del: {result['deletion_auc']:.3f}, Ins: {result['insertion_auc']:.3f}")
        except Exception as e:
            print(f"Error: {e}")
    
    # Plot deletion/insertion curves
    if di_results:
        fig = plot_deletion_insertion_curves(
            di_results,
            save_path='deletion_insertion_curves.png'
        )
        print("Saved curves to 'deletion_insertion_curves.png'")
        plt.close(fig)
    
    # Faithfulness metrics
    print("\nComputing faithfulness metrics...")
    faith_metric = FaithfulnessMetrics(model, device)
    
    for method_name, heatmap in list(explanations.items())[:3]:  # Evaluate first 3 methods
        print(f"\n  {method_name}:")
        try:
            metrics = faith_metric.evaluate_all(image, heatmap, target_class)
            for metric_name, value in metrics.items():
                print(f"    - {metric_name}: {value:.3f}")
        except Exception as e:
            print(f"    Error: {e}")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("\nGenerated files:")
    print("  - comparison.png")
    print("  - deletion_insertion_curves.png")
    
    # Summary
    print("\nSummary:")
    print("\nDeletion AUC (lower is better):")
    for method, result in sorted(di_results.items(), key=lambda x: x[1]['deletion_auc']):
        print(f"  {method}: {result['deletion_auc']:.3f}")
    
    print("\nInsertion AUC (higher is better):")
    for method, result in sorted(di_results.items(), key=lambda x: x[1]['insertion_auc'], reverse=True):
        print(f"  {method}: {result['insertion_auc']:.3f}")


if __name__ == '__main__':
    main()
