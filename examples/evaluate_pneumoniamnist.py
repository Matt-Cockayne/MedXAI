"""
Example: Evaluate explainability methods on Chest X-ray dataset (MedMNIST).

This script demonstrates:
- Loading PneumoniaMNIST or ChestMNIST from MedMNIST
- Generating explanations for chest X-ray pathology classification
- Quantitative evaluation
- Visualizing results

MedMNIST chest X-ray datasets:
- PneumoniaMNIST: Binary classification (pneumonia vs normal)
- ChestMNIST: Multi-class thoracic diseases
"""

import torch
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    get_medical_dataset, load_model, 
    visualize_comparison, plot_deletion_insertion_curves
)
from explainers import GradCAM, GradCAMPlusPlus, IntegratedGradients, RISE
from metrics import DeletionInsertion, FaithfulnessMetrics


def main():
    print("=" * 70)
    print("MedMNIST Chest X-ray Explainability Evaluation")
    print("=" * 70)
    
    # Check if medmnist is installed
    try:
        import medmnist
        print("medmnist package found")
    except ImportError:
        print("medmnist not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "medmnist"])
        print("medmnist installed")
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Choose dataset: 'pneumoniamnist' or 'chestmnist'
    dataset_name = 'pneumoniamnist'  # Binary: pneumonia vs normal
    # dataset_name = 'chestmnist'  # Multi-class: 14 thoracic diseases
    
    # Load dataset
    print(f"Loading {dataset_name.upper()} dataset...")
    
    # Ensure data directory exists
    data_dir = Path('./data')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Custom transform for grayscale chest X-rays
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    test_dataset = get_medical_dataset(
        dataset_name,
        root=str(data_dir),
        split='test',
        transform=transform,
        download=True
    )
    
    # Get number of classes
    if dataset_name == 'pneumoniamnist':
        num_classes = 2
        class_names = ['Normal', 'Pneumonia']
    else:  # chestmnist
        num_classes = 14
        class_names = [f'Disease_{i}' for i in range(num_classes)]
    
    print(f"Dataset loaded: {len(test_dataset)} test images")
    print(f"   Classes: {num_classes} ({', '.join(class_names[:3])}{'...' if num_classes > 3 else ''})")
    print(f"   Image size: 28x28 (will be resized to 224x224)\n")
    
    # Load model
    print("Loading ResNet50 model...")
    model = load_model('resnet50', num_classes=num_classes, device=device)
    print("Model loaded\n")
    
    # Initialize explainers
    print("Initializing explainability methods...")
    explainers = {
        'GradCAM': GradCAM(model, 'layer4', device),
        'GradCAM++': GradCAMPlusPlus(model, 'layer4', device),
        'Integrated Gradients': IntegratedGradients(model, device),
        'RISE': RISE(model, device, n_masks=1000)
    }
    print(f"{len(explainers)} methods initialized\n")
    
    # Select a sample
    sample_idx = 42  # You can change this
    image, label = test_dataset[sample_idx]
    label = int(label)  # Convert from numpy array to int
    image_batch = image.unsqueeze(0).to(device)
    
    # Load original image without transforms for visualization
    import medmnist
    from medmnist import INFO
    info = INFO[dataset_name]
    DataClass = getattr(medmnist, info['python_class'])
    original_dataset = DataClass(split='test', download=False, root=str(data_dir), transform=None)
    original_image = original_dataset[sample_idx][0]  # PIL Image
    
    # Get model prediction
    print(f"Analyzing sample {sample_idx}...")
    with torch.no_grad():
        output = model(image_batch)
        probs = torch.nn.functional.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()
    
    true_label_name = class_names[label] if label < len(class_names) else f"Class {label}"
    pred_label_name = class_names[pred_class] if pred_class < len(class_names) else f"Class {pred_class}"
    
    print(f"   True label: {true_label_name} ({label})")
    print(f"   Predicted: {pred_label_name} ({pred_class}) (confidence: {confidence:.3f})\n")
    
    # Generate explanations
    print("Generating explanations...")
    explanations = {}
    
    for name, explainer in explainers.items():
        print(f"   - {name}...", end=' ')
        try:
            explanation = explainer.explain(image_batch, target_class=label)
            explanations[name] = explanation
            print("")
        except Exception as e:
            print(f"Error: {e}")
    
    print()
    
    # Visualize comparison
    print("Creating visualizations...")
    output_dir = Path(f'./results/{dataset_name}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save original low-res image
    original_image.save(output_dir / f'sample_{sample_idx}_original.png')
    
    fig = visualize_comparison(
        image,
        explanations,
        original_image=original_image,
        save_path=output_dir / f'sample_{sample_idx}_comparison.png'
    )
    plt.close(fig)
    print(f"   Saved: {output_dir / f'sample_{sample_idx}_comparison.png'}")
    
    # Quantitative evaluation
    print("\nComputing evaluation metrics...")
    di_metric = DeletionInsertion(model, device, n_steps=50)
    results = {}
    
    for name, heatmap in explanations.items():
        print(f"   Evaluating {name}...", end=' ')
        try:
            result = di_metric.evaluate(image_batch, heatmap, label)
            results[name] = result
            print(f"Del: {result['deletion_auc']:.3f}, Ins: {result['insertion_auc']:.3f}")
        except Exception as e:
            print(f"Error: {e}")
    
    # Plot evaluation curves
    if results:
        print("\nCreating evaluation curves...")
        fig = plot_deletion_insertion_curves(
            results,
            save_path=output_dir / f'sample_{sample_idx}_curves.png'
        )
        plt.close(fig)
        print(f"   Saved: {output_dir / f'sample_{sample_idx}_curves.png'}")
    
    # Faithfulness metrics for top 2 methods
    print("\nComputing faithfulness metrics...")
    faith_metric = FaithfulnessMetrics(model, device)
    
    for name, heatmap in list(explanations.items())[:2]:
        print(f"\n   {name}:")
        try:
            metrics = faith_metric.evaluate_all(image_batch, heatmap, label)
            for metric_name, value in metrics.items():
                print(f"      - {metric_name}: {value:.3f}")
        except Exception as e:
            print(f"      Error: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Evaluation Complete!")
    print("=" * 70)
    
    print(f"\nResults Summary for: {true_label_name}")
    print(f"True label: {true_label_name} | Predicted: {pred_label_name}\n")
    
    print("Deletion AUC (lower is better):")
    for method, result in sorted(results.items(), key=lambda x: x[1]['deletion_auc']):
        print(f"   {method:25s}: {result['deletion_auc']:.4f}")
    
    print("\nInsertion AUC (higher is better):")
    for method, result in sorted(results.items(), key=lambda x: x[1]['insertion_auc'], reverse=True):
        print(f"   {method:25s}: {result['insertion_auc']:.4f}")
    
    print(f"\nOutput directory: {output_dir}")
    print("   - Method comparison visualization")
    print("   - Deletion/insertion curves")

if __name__ == '__main__':
    main()
