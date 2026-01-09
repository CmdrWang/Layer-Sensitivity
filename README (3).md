# 🧠 LayerSensOrigin  
*This is a survey on the sources of sensitivity at different layers in pruning.*

![Status](https://img.shields.io/badge/Status-Under%20Development-orange)
![Type](https://img.shields.io/badge/Type-Survey-blueviolet)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Contributions](https://img.shields.io/badge/Contributions-Welcome-brightgreen)

**Quick links:** [Snapshot](#-snapshot) · [Overview](#-overview) · [Taxonomy](#-taxonomy) · [Papers](#-paper-list) · [Experiments](#-future-experiments) · [Layout](#-project-layout) · [Setup](#-setup) · [Contributing](#-contributing) · [Citation](#-citation) · [License](#-license) · [Acknowledgements](#-acknowledgements)

---

## ✦ Snapshot
- **Scope** — Why layers matter differently; how to **quantify** contribution & fragility; what each metric **assumes**.  
- **Taxonomy (by metric type)**  
  ① **Magnitude-based**（幅值类）— norms/magnitudes as *structural* proxies.  
  ② **Sensitivity-based**（灵敏度类）— perturbation/loss/curvature as *functional* signals.  
- **Audience** — pruning, robustness, interpretability, architecture design.  
- **Status** — curated reading list + scaffolding now; experiments & visual gallery next.

> North star: connect **signals**, **structure**, and **behavior** into one coherent account of layer sensitivity.

---

## 🔭 Overview
Layer sensitivity sits at the junction of **signals** (gradients, loss), **structure** (weights, layers, heads), and **behavior** (accuracy, robustness).  
This repository organizes methods and assumptions, highlights agreements & tensions, and prepares a minimal, fair **comparison harness**.

> 目标：以**度量类型**为主线，梳理“层敏感性”的来源假设与实证证据，并为后续复现实验与可视化预留位点。

---

## 🧬 Taxonomy
**Visual map**

~~~mermaid
flowchart LR
  A[Layer Sensitivity] --> B[Magnitude-based]
  A --> C[Sensitivity-based]
  B --> B1[L1/L2 Norm]
  B --> B2[Weight Magnitude]
  C --> C1[Grad Norm]
  C --> C2[Fisher Diagonal]
  C --> C3[Hessian Trace / Taylor]
~~~

**Matrix (by metric type)**

| Category | What it measures | Typical Metrics | Needs Data? | Cost | Typical Uses | Caveats |
|---|---|---|:--:|:--:|---|---|
| **Magnitude-based**（幅值类） | Parameter scale / structure proxies | L1/L2 norm, weight magnitude, filter energy | ❌ | ⭐ | pruning heuristics, hardware-aware compression | ignores loss landscape; brittle across tasks |
| **Sensitivity-based**（灵敏度类） | Loss/output response to perturbations | grad-norm, Fisher diag, Hessian trace, Taylor (1st) | ✅ | ⭐⭐–⭐⭐⭐ | pruning, robustness, saliency | data-dependent; costlier; noisy if undersampled |

> First-order Taylor saliency (per-layer) is often written as \( \Delta \mathcal{L} \approx g^\top \delta w \).

---

## 📚 Paper List


*Inclusion criteria: clear metric definition, reproducible setup, and insight into **why** sensitivity emerges.*



> Additions welcome — label each entry as **Magnitude / Sensitivity / Mixed**.

---

## 🧪  Experiments

## 🗂 Project Layout
~~~text
LayerSensOrigin/
├─ docs/                  # Notes, figures, references (papers.bib optional)
│  └─ figures/
├─ src/                   # Experimental scaffolding (WIP)
│  ├─ metrics/            # Magnitude & Sensitivity implementations
│  ├─ analysis/           # Perturbation and correlation routines
│  └─ visualization/      # Plot utilities
├─ scripts/               # CLI entry points (WIP)
├─ results/               # To-be-generated plots/tables
├─ requirements.txt
└─ README.md
~~~

---

## ⚙️ Setup
~~~bash
git clone https://github.com/CmdrWang/LayerSensOrigin.git
cd LayerSensOrigin
conda create -n layersens python=3.10 -y
conda activate layersens
pip install -r requirements.txt
~~~

---

## 🤝 Contributing
A precise, research-minded contribution helps everyone:

- **Papers** — add to the table with a one-liner on the **assumption**  
- **Summaries** — 5–10 lines: idea • assumption • caveat • setting  
- **Experiments** — small scripts comparing metrics on public models/datasets  
- **Visuals** — compact plots for honest, legible comparisons

**PR checklist**
- [ ] Labeled as **Magnitude / Sensitivity / Mixed**  
- [ ] Minimal commands are reproducible  
- [ ] Venue & year included; link to code if available

**Workflow**
~~~bash
git checkout -b feat/add-paper-<surname-YYYY>
git commit -m "docs(papers): add <Surname YYYY> with Sensitivity label"
git push -u origin feat/add-paper-<surname-YYYY>
~~~

**Style notes**
- Keep labels **Magnitude / Sensitivity / Mixed** consistent  
- Prefer simple, reproducible commands; avoid environment lock-in  
- Cite venues & years; link code when available

---

## 🧾 Citation
~~~bibtex
@misc{wang2025layersensorigin,
  title  = {LayerSensOrigin: A Survey on the Origin and Measurement of Layer Sensitivity in Deep Neural Networks},
  author = {Wang, ...},
  year   = {2025},
  url    = {https://github.com/CmdrWang/LayerSensOrigin}
}
~~~

---

## 📜 License
**MIT** — see `LICENSE`.

---

## 🙏 Acknowledgements
Built on pruning, sensitivity, and information-theoretic analyses.  
Thanks to open-source communities (PyTorch, HuggingFace, OpenReview) for tools that make careful comparisons feasible.

> Goal: connect **signals**, **structure**, and **behavior** into a coherent view of layer sensitivity.

---

### Appendix · Review Notes
- We emphasize **metric-type taxonomy**, not application theme.  
- Reports will show **absolute performance** *and* **ranking stability**.  
- Plots use consistent **sampling** (batches/seeds) and report **variance**.


## 📚 Paper List
# 剪枝论文列表 (Pruning Papers)

## Structured Pruning (结构化剪枝)
## Unstructured Pruning (非结构化剪枝)

<details open>
<summary><strong>Image</strong></summary>

| **Title & Authors** | **Areas** | **Tags** | **Links** | 
| --- | --- | --- | :---: | 
|  [![Publish](https://img.shields.io/badge/ACM_MM-2025-blue)]() [![Star](https://img.shields.io/github/stars/example/mm-prune-bench.svg?style=social&label=Star)](https://github.com/example/mm-prune-bench)<br>[Multimodal Pruning Benchmark](https://arxiv.org/abs/2510.00011)<br>G. Patel; H. Nguyen |  [![Area](https://img.shields.io/badge/Image-purple)]() [![Area](https://img.shields.io/badge/Video-purple)]() |  [![Type](https://img.shields.io/badge/structured-green)]() |  [Paper](https://arxiv.org/abs/2510.00011)<br> [GitHub](https://github.com/example/mm-prune-bench)<br> | 
|  [![Publish](https://img.shields.io/badge/ICCV-2025-blue)]() [![Star](https://img.shields.io/github/stars/example/sparse-vision.svg?style=social&label=Star)](https://github.com/example/sparse-vision)<br>[Sparse Vision at Scale](https://arxiv.org/abs/2509.01234)<br>A. Zhang; B. Lee |  [![Area](https://img.shields.io/badge/Image-purple)]() |  [![Type](https://img.shields.io/badge/structured-green)]() |  [Paper](https://arxiv.org/abs/2509.01234)<br> [GitHub](https://github.com/example/sparse-vision)<br> [Model](https://huggingface.co/example/sparse-vision)<br> | 
</details>

<details open>
<summary><strong>Video</strong></summary>

| **Title & Authors** | **Areas** | **Tags** | **Links** | 
| --- | --- | --- | :---: | 
|  [![Publish](https://img.shields.io/badge/ACM_MM-2025-blue)]() [![Star](https://img.shields.io/github/stars/example/mm-prune-bench.svg?style=social&label=Star)](https://github.com/example/mm-prune-bench)<br>[Multimodal Pruning Benchmark](https://arxiv.org/abs/2510.00011)<br>G. Patel; H. Nguyen |  [![Area](https://img.shields.io/badge/Image-purple)]() [![Area](https://img.shields.io/badge/Video-purple)]() |  [![Type](https://img.shields.io/badge/structured-green)]() |  [Paper](https://arxiv.org/abs/2510.00011)<br> [GitHub](https://github.com/example/mm-prune-bench)<br> | 
|  [![Publish](https://img.shields.io/badge/ICML-2025-blue)]() [![Star](https://img.shields.io/github/stars/example/video-prune.svg?style=social&label=Star)](https://github.com/example/video-prune)<br>[Magnitude-based Video Pruning](https://arxiv.org/abs/2507.07890)<br>E. Rossi; F. Chen |  [![Area](https://img.shields.io/badge/Video-purple)]() |  [![Type](https://img.shields.io/badge/unstructured-green)]() |  [Paper](https://arxiv.org/abs/2507.07890)<br> [GitHub](https://github.com/example/video-prune)<br> [Model](https://huggingface.co/example/video-prune)<br> | 
</details>

<details open>
<summary><strong>Audio</strong></summary>

| **Title & Authors** | **Areas** | **Tags** | **Links** | 
| --- | --- | --- | :---: | 
|  [![Publish](https://img.shields.io/badge/ACL-2025-blue)]() [![Star](https://img.shields.io/github/stars/example/sens-trans.svg?style=social&label=Star)](https://github.com/example/sens-trans)<br>[Sensitivity-Aware Transformers](https://arxiv.org/abs/2508.04567)<br>C. Wang; D. Kumar |  [![Area](https://img.shields.io/badge/Audio-purple)]() |  [![Type](https://img.shields.io/badge/semi--structured-green)]() |  [Paper](https://arxiv.org/abs/2508.04567)<br> [GitHub](https://github.com/example/sens-trans)<br> | 
</details>


<details open>
<summary><strong>ICCV 2025</strong></summary>

| **Title & Authors** | **Areas** | **Tags** | **Links** | 
| --- | --- | --- | :---: | 
|  [![Publish](https://img.shields.io/badge/ICCV-2025-blue)]() [![Star](https://img.shields.io/github/stars/example/sparse-vision.svg?style=social&label=Star)](https://github.com/example/sparse-vision)<br>[Sparse Vision at Scale](https://arxiv.org/abs/2509.01234)<br>A. Zhang; B. Lee |  [![Area](https://img.shields.io/badge/Image-purple)]() |  [![Type](https://img.shields.io/badge/structured-green)]() |  [Paper](https://arxiv.org/abs/2509.01234)<br> [GitHub](https://github.com/example/sparse-vision)<br> [Model](https://huggingface.co/example/sparse-vision)<br> | 
</details>

<details open>
<summary><strong>ACL 2025</strong></summary>

| **Title & Authors** | **Areas** | **Tags** | **Links** | 
| --- | --- | --- | :---: | 
|  [![Publish](https://img.shields.io/badge/ACL-2025-blue)]() [![Star](https://img.shields.io/github/stars/example/sens-trans.svg?style=social&label=Star)](https://github.com/example/sens-trans)<br>[Sensitivity-Aware Transformers](https://arxiv.org/abs/2508.04567)<br>C. Wang; D. Kumar |  [![Area](https://img.shields.io/badge/Audio-purple)]() |  [![Type](https://img.shields.io/badge/semi--structured-green)]() |  [Paper](https://arxiv.org/abs/2508.04567)<br> [GitHub](https://github.com/example/sens-trans)<br> | 
</details>

<details open>
<summary><strong>ICML 2025</strong></summary>

| **Title & Authors** | **Areas** | **Tags** | **Links** | 
| --- | --- | --- | :---: | 
|  [![Publish](https://img.shields.io/badge/ICML-2025-blue)]() [![Star](https://img.shields.io/github/stars/example/video-prune.svg?style=social&label=Star)](https://github.com/example/video-prune)<br>[Magnitude-based Video Pruning](https://arxiv.org/abs/2507.07890)<br>E. Rossi; F. Chen |  [![Area](https://img.shields.io/badge/Video-purple)]() |  [![Type](https://img.shields.io/badge/unstructured-green)]() |  [Paper](https://arxiv.org/abs/2507.07890)<br> [GitHub](https://github.com/example/video-prune)<br> [Model](https://huggingface.co/example/video-prune)<br> | 
</details>

<details open>
<summary><strong>ACM MM 2025</strong></summary>

| **Title & Authors** | **Areas** | **Tags** | **Links** | 
| --- | --- | --- | :---: | 
|  [![Publish](https://img.shields.io/badge/ACM_MM-2025-blue)]() [![Star](https://img.shields.io/github/stars/example/mm-prune-bench.svg?style=social&label=Star)](https://github.com/example/mm-prune-bench)<br>[Multimodal Pruning Benchmark](https://arxiv.org/abs/2510.00011)<br>G. Patel; H. Nguyen |  [![Area](https://img.shields.io/badge/Image-purple)]() [![Area](https://img.shields.io/badge/Video-purple)]() |  [![Type](https://img.shields.io/badge/structured-green)]() |  [Paper](https://arxiv.org/abs/2510.00011)<br> [GitHub](https://github.com/example/mm-prune-bench)<br> | 
</details>

