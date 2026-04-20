# 🌌 AGN Morphology Pipeline : IA Summer Program 2024

### 🚀 TL;DR: The Software & Data Science Angle
If your primary interest is the code, this repository demonstrates applied **data engineering and computer vision**. 
* **The Problem:** Processing massive amounts of raw, noisy scientific image data into structured, analyzable metrics.
* **The Solution:** I built a fully automated Python pipeline that ingests raw FITS files, cleans corrupt or non-finite data, applies 8 different image segmentation algorithms (via `scikit-image`), calculates complex custom mathematical metrics from the resulting pixel arrays, and exports the data to CSVs with automated visualizations.
* **Key Skills:** Data Engineering pipelines, Computer Vision (Image Segmentation & Masking), Algorithm Implementation, Batch Processing, `Python`, `pandas`, `numpy`, `scikit-image`.

***

### 🔭 About the Internship Project
This repository contains the non-parametric image analysis pipeline I developed during my internship at the Instituto de Astrofísica e Ciências do Espaço for the **IA Summer Program 2024**. You can view the official list of program projects, including this one, on the [IA Summer Program 2024 Projects List](https://divulgacao.iastro.pt/pt/ia-summer-program-2024-projects-list/).

The overarching research goal was testing the 'merging paradigm' in AGN-host galaxies. The hypothesis explores whether the chaos that occurs during the process of galaxy mergers could alter the torus-like structure of the AGN, obscuring it and making it a Type II. The team analyzed a sample of 60 galaxies with [NeV]3426 emission with a redshift between 0.6 and 1.2. The data used and analyzed comes from the Hubble Space Telescope (HST) and James Webb Space Telescope (JWST).

While the team utilized both parametric and non-parametric methods to study these mergers, my specific responsibility was architecting and implementing the automated Non-Parametric Morphological Analysis Pipeline in Python.

### ⚙️ My Role and The Pipeline
Non-parametric methods use statistical measures to describe galaxy morphology without assuming any specific model. I built a robust automated system to process raw astronomical data (`.fits` files), isolate the galaxies, and compute key structural metrics.

**Core Responsibilities & Workflow:**
* **Data Ingestion & Cleaning:** Processing raw `FITS` files using `astropy`, handling non-finite pixels, and preparing the image arrays for analysis.
* **Advanced Image Segmentation:** Implementing multiple thresholding algorithms (Isodata, Li, Otsu, Triangle, Yen) via `scikit-image` and integrating `source-extractor` to isolate the galactic structure from the background space.
* **Morphological Mathematics:** Writing pure Python functions to calculate critical non-parametric indicators on the segmented pixels:
    * **Gini Coefficient:** Measures the inequality of pixel flux distribution.
    * **M20 Index:** Measures the second-order moment of the brightest 20% of the galaxy's flux.
    * **Filamentarity:** Evaluates the structural elongation based on the isophotal area.
* **Data Visualization & Export:** Automatically generating visual comparison grids for every thresholding method, exporting all metrics to structured CSV files, and rendering comparative bar charts with standard deviations using `pandas` and `matplotlib`.

### 📂 Repository Structure
* **`GGMF.py`:** The primary analysis engine. It loops through a directory of `.fits` files, applies 8 different segmentation methods, calculates the Gini, M20, and Filamentarity indices, and exports a visual diagnostic grid alongside a comprehensive `gini_m20_filamentarity_coefficients.csv`.
* **`HistoGenerator.py`:** A data visualization script that reads the exported CSV and generates comparative, error-bar-equipped histograms for the morphological metrics across different thresholding techniques.
* **`/Outputs/`:** Contains the generated analytical artifacts:
    * `*_thresholding_comparison.png`: A visual grid showing the original image, binary mask, cleaned mask, and segmented galaxy for every algorithm tested.
    * `*_combined_histograms.png`: Bar charts visualizing the Gini, M20, and Filamentarity results for an individual galaxy.
    * `gini_m20_filamentarity_coefficients.csv`: The final dataset compiling all metric calculations.
* **`It takes two to tango.pdf`:** The final team presentation detailing the comprehensive results of the summer program.

### 🛠️ Tech Stack
* **Languages:** Python
* **Astronomy Tools:** `astropy`, SExtractor (`source-extractor`)
* **Image Processing:** `scikit-image` (`skimage`), `scipy`
* **Data Science:** `numpy`, `pandas`, `matplotlib`