# Graph Deep Learning Project – TopoMasking

This repository contains the implementation and report for the **Graph Deep Learning group project** on **Linear Transformer Topological Masking (TopoMasking)**, based on the ICLR 2025 paper _“Linear Transformer Topological Masking with Graph Random Features”_.

---

## 📦 1. Clone the Repository

To get started, clone this repository locally:

```bash
git clone https://github.com/USI-Projects-Collection/GDL_Project.git
cd GDL_Project
```

---

## ⚙️ 2. Conda Environment Setup

We use **Conda** to manage dependencies and ensure reproducibility

### ▶️ Create the environment

```bash
conda env create -f environment.yaml
conda activate topo_masking
```

If the environment file changes (e.g., someone added new dependencies) [see Section 5](#👥-5-collaboration-workflow):

```bash
conda env update -f env/environment.yaml --prune
```

---

## 📄 3. LaTeX Report Compilation

Use a Makefile to build the final PDF.

#### 🔹 On macOS
```bash
brew install --cask mactex
```

#### 🔹 On Ubuntu / Debian
```bash
sudo apt install texlive-full
```

---

### 🧱 Build the PDF

From inside the `Template_GDL_Report/` folder:

```bash
make          # Compiles the LaTeX report
make clean    # Removes temporary LaTeX files
```

---

## 👥 4. Collaboration Workflow

When adding new dependencies:
1. Install the package in your Conda environment:
   ```bash
   conda install nome_pacchetto
   ```
2. Add the package manually to the dependencies section of `environment.yaml`:
   ```yaml
   dependencies:
     - pytorch
     - pyg
     - numpy
     - package_you_installed # newly added package
   ```

3. Commit and push the updated file so everyone can sync with:
   ```bash
   conda env update -f environment.yaml --prune
   ```

---

## 🧾 5. License & Credits

© 2025 — Università della Svizzera italiana (USI), Master in Artificial Intelligence.  
Developed by the Graph Deep Learning project group.
