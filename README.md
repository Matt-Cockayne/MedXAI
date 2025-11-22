# Explainable-AI: Comprehensive Medical Imaging Explainability Toolkit

A PyTorch-based toolkit for comparing and evaluating explainability methods in medical image classification. This repository provides a unified interface for various explainability techniques with quantitative evaluation metrics.

## Features

### Explainability Methods

#### Gradient-based
- **GradCAM**: Class Activation Mapping using gradients
- **GradCAM++**: Improved weighted class activation mapping
- **Integrated Gradients**: Path-based attribution method

#### Perturbation-based
- **RISE**: Randomized Input Sampling for Explanation
- **Occlusion**: Sliding window occlusion sensitivity analysis

#### Attention-based
- **Raw Attention Maps**: Direct visualization of attention weights from Vision Transformers

#### Concept-based
- **CBM Attributions**: Concept Bottleneck Model concept importance

### Evaluation Metrics
- **Pointing Game**: Localization accuracy metric
- **Deletion/Insertion Curves**: Faithfulness evaluation
- **Faithfulness Assessment**: Correlation with model predictions
- **Plausibility Assessment**: Agreement with human annotations

### Visualization Tools
- Side-by-side method comparisons
- Quantitative metric dashboards
- Interactive exploration interface

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from explainers import GradCAM, GradCAMPlusPlus, IntegratedGradients
from metrics import pointing_game, deletion_insertion_curves
from utils.visualization import compare_methods
import torch
from torchvision import models

# Load your model
model = models.resnet50(pretrained=True)
model.eval()

# Initialize explainers
explainers = {
    'GradCAM': GradCAM(model, target_layer='layer4'),
    'GradCAM++': GradCAMPlusPlus(model, target_layer='layer4'),
    'IntegratedGradients': IntegratedGradients(model)
}

# Generate explanations
image = torch.randn(1, 3, 224, 224)
results = {}
for name, explainer in explainers.items():
    results[name] = explainer.explain(image, target_class=0)

# Compare methods visually
compare_methods(image, results, save_path='comparison.png')
```

## Directory Structure

```
Explainable-AI/
├── README.md
├── requirements.txt
├── setup.py
├── explainers/
│   ├── __init__.py
│   ├── base.py              # Base explainer class
│   ├── gradcam.py           # GradCAM implementation
│   ├── gradcam_plusplus.py  # GradCAM++ implementation
│   ├── integrated_gradients.py
│   ├── rise.py              # RISE implementation
│   ├── occlusion.py         # Occlusion analysis
│   ├── attention.py         # ViT attention extraction
│   └── cbm.py               # CBM concept attribution
├── metrics/
│   ├── __init__.py
│   ├── pointing_game.py
│   ├── deletion_insertion.py
│   ├── faithfulness.py
│   └── plausibility.py
├── utils/
│   ├── __init__.py
│   ├── visualization.py     # Plotting and comparison tools
│   ├── data_utils.py        # Dataset loading utilities
│   └── model_utils.py       # Model loading helpers
├── interface/
│   ├── app.py               # Interactive Gradio/Streamlit app
│   └── callbacks.py
├── notebooks/
│   ├── 01_basic_usage.ipynb
│   ├── 02_method_comparison.ipynb
│   ├── 03_quantitative_evaluation.ipynb
│   └── 04_medical_imaging_demo.ipynb
├── examples/
│   ├── compare_all_methods.py
│   ├── evaluate_faithfulness.py
│   └── medical_imaging_example.py
└── tests/
    ├── test_explainers.py
    ├── test_metrics.py
    └── test_visualization.py
```

## Usage Examples

### Method Comparison

```python
from explainers import get_all_explainers
from utils.visualization import visualize_comparison
from metrics import evaluate_all_metrics

# Load model and image
model = load_model('resnet50')
image, ground_truth = load_medical_image('path/to/image.png')

# Get all explainers
explainers = get_all_explainers(model)

# Generate explanations
explanations = {}
for name, explainer in explainers.items():
    explanations[name] = explainer.explain(image)

# Evaluate with all metrics
metrics_results = evaluate_all_metrics(
    model, image, explanations, ground_truth
)

# Visualize
visualize_comparison(image, explanations, metrics_results)
```

### Quantitative Evaluation

```python
from metrics import DeletionInsertion, PointingGame
from utils.data_utils import get_medical_dataset

# Load dataset
dataset = get_medical_dataset('ISIC2019')

# Initialize metrics
di_metric = DeletionInsertion(model)
pg_metric = PointingGame()

# Evaluate over dataset
results = {}
for image, mask, label in dataset:
    explanation = explainer.explain(image, target_class=label)
    
    results['deletion_auc'] = di_metric.deletion_score(image, explanation, label)
    results['insertion_auc'] = di_metric.insertion_score(image, explanation, label)
    results['pointing_acc'] = pg_metric.evaluate(explanation, mask)
```

### Interactive Interface

```bash
# Launch the interactive comparison tool
python interface/app.py --model resnet50 --dataset ISIC2019
```

## Supported Models

- ResNet family (ResNet18, ResNet50, ResNet101)
- Vision Transformers (ViT-B/16, ViT-L/16)
- EfficientNet family
- DenseNet family
- Custom PyTorch models

## Supported Datasets

- ISIC (Skin lesion classification)
- ChestX-ray14
- CheXpert
- Custom medical imaging datasets
- ImageNet (for general validation)

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{explainable_ai_toolkit,
  author = {Matthew Cockayne},
  title = {Explainable-AI: Comprehensive Medical Imaging Explainability Toolkit},
  year = {2025},
  url = {https://github.com/Matt-Cockayne/Explainable-AI}
}
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Related Work

This toolkit builds upon and extends the CAM-based methods from the Classification-to-Segmentation project, providing a comprehensive comparison framework for explainability research in medical imaging.
