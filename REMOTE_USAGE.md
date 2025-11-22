# Running in Remote/Headless Environments

This guide covers how to run the Explainable-AI toolkit in remote, headless, or HPC environments.

## Interactive Interface (Gradio)

### Option 1: Public Share Link (Easiest for Remote)

```bash
python interface/app.py --share
```

This creates a **temporary public URL** (valid for 72 hours) that you can access from anywhere:
```
Running on public URL: https://abc123.gradio.live
```

✅ **Best for:** Quick demos, temporary access, no firewall issues
❌ **Limitations:** URL expires after 72 hours, requires internet

### Option 2: SSH Port Forwarding (More Secure)

On your **local machine**, create an SSH tunnel:
```bash
ssh -L 7860:localhost:7860 user@remote-server
```

Then on the **remote server**:
```bash
python interface/app.py
```

Access via: `http://localhost:7860` on your local browser

✅ **Best for:** Secure access, persistent sessions
❌ **Requires:** SSH access, manual tunnel setup

### Option 3: Custom Port/Server

For specific network configurations:
```bash
# Bind to all interfaces on custom port
python interface/app.py --port 8080 --server 0.0.0.0

# With public sharing
python interface/app.py --share --port 8080
```

## Programmatic Usage (No GUI Required)

For **HPC clusters** or **batch processing**, use the Python API directly without the interface:

```python
import torch
from explainers import GradCAM, GradCAMPlusPlus, IntegratedGradients
from utils import load_model, load_image
from metrics import DeletionInsertion

# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = load_model('resnet50', pretrained=True, device=device)

# Load image
image = load_image('path/to/image.jpg').unsqueeze(0).to(device)

# Generate explanations
explainer = GradCAM(model, 'layer4', device)
heatmap = explainer.explain(image, target_class=0)

# Evaluate
evaluator = DeletionInsertion(model, device)
results = evaluator.evaluate(image, heatmap, target_class=0)

# Save results
torch.save({'heatmap': heatmap, 'metrics': results}, 'output.pt')
```

## Jupyter Notebooks in Remote Environments

### JupyterLab with Port Forwarding

On **remote server**:
```bash
conda activate XAI
jupyter lab --no-browser --port=8888
```

On **local machine**:
```bash
ssh -L 8888:localhost:8888 user@remote-server
```

Access: `http://localhost:8888`

### VS Code Remote

1. Install "Remote - SSH" extension
2. Connect to remote server
3. Open notebook directly in VS Code

## SLURM/HPC Batch Jobs

Example SLURM script:

```bash
#!/bin/bash
#SBATCH --job-name=xai_eval
#SBATCH --output=xai_%j.out
#SBATCH --error=xai_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

# Activate environment
conda activate XAI

# Run evaluation script
python examples/compare_all_methods.py

# Or batch process multiple images
python <<EOF
import torch
from pathlib import Path
from explainers import GradCAM
from utils import load_model, load_image

device = 'cuda'
model = load_model('resnet50', pretrained=True, device=device)
explainer = GradCAM(model, 'layer4', device)

# Process all images
for img_path in Path('data/images').glob('*.jpg'):
    image = load_image(str(img_path)).unsqueeze(0).to(device)
    heatmap = explainer.explain(image)
    torch.save(heatmap, f'results/{img_path.stem}_gradcam.pt')
EOF
```

## Docker Container (Fully Headless)

Create `Dockerfile`:
```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# For API mode (no GUI)
CMD ["python", "examples/compare_all_methods.py"]

# Or for Gradio with sharing
# CMD ["python", "interface/app.py", "--share"]
```

Build and run:
```bash
docker build -t explainable-ai .
docker run --gpus all explainable-ai
```

## Visualization in Headless Mode

When you can't display plots interactively:

```python
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from utils import compare_methods

# Generate visualization
fig = compare_methods(image, explanations, save_path='comparison.png')
plt.close(fig)  # Don't try to display

# Result saved to comparison.png
```

## Performance Considerations

### For HPC/Remote Servers:

1. **Use batch processing** instead of interactive interface
2. **Pre-download models** to avoid repeated downloads:
   ```python
   model = load_model('resnet50', pretrained=True)
   torch.save(model.state_dict(), 'resnet50_weights.pth')
   ```

3. **Optimize RISE** (slowest method):
   ```python
   # Use fewer masks for faster computation
   rise = RISE(model, device, n_masks=1000)  # Instead of 8000
   ```

4. **Enable mixed precision** for faster GPU computation:
   ```python
   from torch.cuda.amp import autocast
   
   with autocast():
       explanation = explainer.explain(image)
   ```

## Troubleshooting

### "Cannot connect to X server"
- Solution: Use `matplotlib.use('Agg')` for non-interactive backend
- Or: Use the programmatic API without visualization

### "Port already in use"
- Solution: `python interface/app.py --port 8080`

### "No module named 'tkinter'"
- This toolkit doesn't require tkinter
- If error appears, ensure you're using Agg backend for matplotlib

### Gradio share link not working
- Check firewall settings
- Ensure internet connectivity
- Try: `python interface/app.py --share --server 0.0.0.0`

## Recommended Setup for Remote Work

```bash
# 1. Create conda environment
conda create -n XAI python=3.10
conda activate XAI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Test installation (headless)
python -c "import torch; from explainers import GradCAM; print('✅ Setup successful')"

# 4. Run with sharing for remote access
python interface/app.py --share
```

## Security Notes

- **Public share links** are accessible to anyone with the URL
- For sensitive medical data, use **SSH tunneling** or **VPN**
- Consider using **authentication** for production deployments
- The toolkit processes data locally; no data is sent to external servers (except for Gradio's relay when using --share)
