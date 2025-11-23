# MedXAI Portfolio Integration Guide

## ✅ Completed Steps

1. **Created project page**: `matt-cockayne.github.io/_projects/medxai.md`
2. **Copied visualization images** to `matt-cockayne.github.io/assets/projects/medxai/`:
   - `gradcam_examples.png` - GradCAM visualizations
   - `lime_explanation.png` - LIME superpixel analysis
   - `shap_comparison.png` - SHAP method comparison
   - `dermamnist_comparison.png` - DermaMNIST multi-method comparison
   - `dermamnist_curves.png` - Deletion/Insertion curves
   - `pneumoniamnist_comparison.png` - PneumoniaMNIST results

## 📋 Next Steps to Deploy

### 1. Build and Preview Locally

```bash
cd ~/Documents/PhD/portfolio/matt-cockayne.github.io

# Build the site
bundle exec jekyll serve

# Access at: http://localhost:4000
```

### 2. Review the Project Page

Navigate to: `http://localhost:4000/projects.html`

The MedXAI project should now appear alongside DermFormer and Classification-to-Segmentation.

### 3. Verify Images Display Correctly

Check that all visualizations render properly on the project page. If images don't show:
- Verify file paths in `_projects/medxai.md`
- Check image files exist in `assets/projects/medxai/`
- Ensure Jekyll can access the assets directory

### 4. Customize Content (Optional)

Edit `_projects/medxai.md` to:
- Update the interactive demo URL once deployed
- Add more visualization examples
- Include additional tutorials or documentation links
- Adjust technical depth based on audience

### 5. Deploy to GitHub Pages

```bash
cd ~/Documents/PhD/portfolio/matt-cockayne.github.io

# Add new files
git add _projects/medxai.md
git add assets/projects/medxai/

# Commit changes
git commit -m "Add MedXAI project to portfolio"

# Push to GitHub
git push origin main
```

GitHub Pages will automatically rebuild your site (takes 1-2 minutes).

### 6. Verify Live Site

Visit your portfolio URL and confirm:
- MedXAI appears on the projects page
- All images load correctly
- Links work (GitHub repo, documentation)
- Mobile responsiveness looks good

## 🎨 Additional Enhancements

### Add an Interactive Demo

If you deploy the Gradio interface publicly (e.g., on Hugging Face Spaces):

1. Update the demo URL in `_projects/medxai.md`:
```yaml
links:
  - name: "Interactive Demo"
    url: "https://huggingface.co/spaces/YOUR_USERNAME/medxai"
```

2. Optionally embed the demo in the project page:
```html
<iframe src="YOUR_DEMO_URL" width="100%" height="800px"></iframe>
```

### Create a Featured Project Card

If you want MedXAI to appear prominently on your homepage, you can add a featured project section or update `index.md`.

### Add to CV/Resume

Don't forget to reference MedXAI in your CV under:
- **Projects** section
- **Skills** section (PyTorch, XAI, Medical Imaging)
- **Publications** (if you publish research using it)

## 📦 Project Structure Overview

```
matt-cockayne.github.io/
├── _projects/
│   ├── dermformer.md
│   ├── classification-to-segmentation.md
│   └── medxai.md                          # ← NEW
├── assets/
│   └── projects/
│       ├── dermformer/
│       └── medxai/                        # ← NEW
│           ├── gradcam_examples.png
│           ├── lime_explanation.png
│           ├── shap_comparison.png
│           ├── dermamnist_comparison.png
│           ├── dermamnist_curves.png
│           └── pneumoniamnist_comparison.png
└── projects.md                            # Lists all projects
```

## 🔗 Linking Strategy

The project page includes links to:
- ✅ GitHub repository (main codebase)
- ✅ Documentation (README)
- ⏳ Interactive demo (add URL when deployed)

Consider also linking:
- Tutorial notebooks (via GitHub)
- Published papers that use MedXAI
- Related projects (DermFormer integration)

## 🎯 Content Highlights

The project page emphasizes:
1. **Clinical Relevance**: Bridging AI and medical interpretability
2. **Technical Depth**: 6 XAI methods with evaluation metrics
3. **Educational Value**: Comprehensive tutorials
4. **Visual Evidence**: Multiple visualization examples
5. **Open Source**: MIT license, reproducible research

## 📊 SEO & Discoverability

The project metadata includes:
- Keywords: "Explainable AI", "Medical Imaging", "PyTorch"
- Technologies: Listed in YAML frontmatter
- Status: "Active Development"
- Links: GitHub, docs, demo

This helps with:
- Search engine discovery
- GitHub profile integration
- Academic citations
- Recruiter/collaborator interest

## 🚀 Future Updates

When you make significant updates to MedXAI:

1. Update the project page to reflect new features
2. Add new visualizations to the assets folder
3. Update the status or duration if needed
4. Reference any publications that cite or use MedXAI

## 💡 Tips

- **Keep it Updated**: Refresh the project page when you add major features
- **Add Metrics**: Include usage stats (GitHub stars, citations, etc.)
- **Show Impact**: Highlight any real-world deployments or collaborations
- **Link Projects**: Cross-reference with DermFormer for integrated XAI

---

**Questions?** The project page follows the same Jekyll structure as your other projects, so you can reference `dermformer.md` for formatting examples.
