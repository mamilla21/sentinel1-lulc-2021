# Sentinel-1 Land Use and Land Cover Segmentation (2021)

**Master's project \| Deep learning · Remote sensing · Streamlit**

This project trains a **U-Net semantic segmentation model** to classify land use and land cover (LULC) from **Sentinel-1 synthetic aperture radar (SAR)** images and presents **12 monthly maps for 2021** in an interactive Streamlit dashboard. The prediction pipeline accepts a city or latitude/longitude, estimates the area covered by each land-cover class, and reports month-to-month and January-to-December differences in **percentage points**.

> **Important:** The model uses **only Sentinel-1 VV and VH radar channels as inference inputs**. ESA WorldCover 2021 provides **supervised training labels**, not an additional model input. Monthly outputs are model predictions, not independently verified physical land-cover changes.

## Features

- Sentinel-1 GRD (IW, descending orbit) 2021 radar composites, VV and VH channels  
- Radar preprocessing and 256 × 256 pixel training patches  
- ResNet18-encoder U-Net for 11-class pixel-wise segmentation  
- Baseline evaluation and fine-tuned model comparison  
- Monthly 2021 inference for selected locations  
- Interactive Streamlit dashboard with 12 predicted maps, percentage tables, monthly changes, total changes, plots, and downloads  
- Heidelberg as the principal 12-month demonstration case; additional locations supported by the dashboard

## Data and methodology

| Component | Description |
| :---- | :---- |
| Model inputs | Sentinel-1 GRD SAR: VV and VH |
| Supervision | ESA WorldCover 2021 v200 |
| Year | 2021 |
| Image patch | 256 × 256 pixels at 10 m resolution |
| Inference footprint | Approximately 2.56 × 2.56 km |
| Geographic projection | EPSG:3035 (dashboard processing) |
| Architecture | U-Net with ResNet18 encoder, two input channels, 11 output classes |
| Classes | Tree cover, Shrubland, Grassland, Cropland, Built-up, Bare/sparse vegetation, Snow/ice, Permanent water bodies, Herbaceous wetland, Mangroves, Moss/lichen |

Training data consist of matched Sentinel-1 radar patches and WorldCover label patches. The dataset also includes supplementary urban samples. Geographic splits were used to reduce spatial leakage between training and evaluation data.

**Preprocessing:** Sentinel-1 VV radar in dB was clipped to \[−30, 5\] and scaled to \[0, 1\]; VH was clipped to \[−35, 0\] and scaled to \[0, 1\]. Monthly inference uses radar composites for the selected 2021 month. Exact data preparation steps and saved execution outputs are documented in the notebooks.

## Repository organization

```
Sentinel1-LULC-Segmentation-2021/
├── notebooks/
│   ├── 01_Sentinel1_Data_Collection.ipynb
│   ├── 02_UNet_Model_Training.ipynb
│   ├── 03_Model_Evaluation.ipynb
│   └── 04_Monthly_LULC_Dashboard_2021.ipynb
├── app/
│   └── app_2021_full.py
├── results/
│   └── [selected exported 2021 plots, maps and CSVs]
├── README.md
├── requirements.txt
└── .gitignore
```

The fourth notebook listed above is the **final, executed Notebook 05**, renamed in the submission copy for a clean 01–04 sequence. The earlier exploratory Notebook 04 is kept separately as a backup. **Keep filenames consistent with what is actually uploaded.**

## Notebooks

1. **01 — Data collection:** Fetch and preprocess Sentinel-1 SAR and WorldCover training labels; construct training patches.  
2. **02 — Model training:** Train the ResNet18 U-Net and fine-tune using additional urban training examples.  
3. **03 — Model evaluation:** Inspect segmentation quality with accuracy, F1 and intersection-over-union (IoU) metrics.  
4. **04 — Monthly prediction and dashboard:** Use the saved fine-tuned model to generate 12 monthly LULC predictions and launch or interact with the Streamlit application.

## Evaluation summary

Results recorded during project experimentation:

| Evaluation | Result |
| :---- | ----: |
| Baseline pixel accuracy | 77.60% |
| Baseline mIoU (all reference-present classes) | 0.4270 |
| Baseline macro F1 | 0.5321 |
| Fine-tuned pixel accuracy | 78.63% |
| Fine-tuned mIoU (classes with ≥100 test pixels) | 0.5068 |
| Baseline mIoU on that same ≥100-pixel class subset | 0.4982 |

**Metric caveat:** The 0.4270 and 0.5068 mIoU figures are from *different class-inclusion protocols* and must not be directly compared. The appropriate like-for-like mIoU comparison is **0.4982 vs 0.5068**.

## Running the dashboard

### Google Colab (recommended)

1. Open the final dashboard notebook in Google Colab.  
2. Connect a runtime and mount the Google Drive containing the project files.  
3. Install dependencies from `requirements.txt`.  
4. Authenticate Google Earth Engine using your own Earth Engine-enabled Google Cloud project.  
5. Confirm the checkpoint `unet_sentinel1_finetune_best.pth` is available at the path expected by the app. Adjust the app's path configuration if you organize files differently.  
6. Execute the notebook's Streamlit launch cell and open the **new** Colab-provided dashboard URL.  
7. Select a city or enter coordinates, choose the 2021 start month if available, and click the button to generate monthly maps. Review the resulting plots and download tables.

The Colab dashboard URL is **temporary**; it stops working when the runtime disconnects. It is not a hosted permanent website.

### Local environment

Install Python dependencies and configure Google Earth Engine authentication and model paths first:

```sh
pip install -r requirements.txt
streamlit run app/app_2021_full.py
```

The script requires valid Earth Engine access and the downloaded trained weights. Colab-specific paths or code may need adaptation before local execution.

## Results and interpretation

The Heidelberg 2021 case study produced 12 predicted LULC maps with monthly class percentage tables. Results can also be downloaded from the dashboard for other supported locations when radar coverage and Earth Engine access are available.

**Limitations:** SAR-based classes can be confused; the model may overpredict built-up pixels, particularly in mixed urban scenes. Changes between predicted monthly percentages can reflect radar acquisition conditions or model uncertainty. These should **not** be presented as confirmed construction, deforestation, or other real land-use change without independent validation.

## Project resources — replace these placeholders before submission

| Deliverable | Link |
| :---- | :---- |
| Public dataset (training and urban patches, geographic split metadata) | [**https\://drive.google.com/drive/folders/1DbFdSLy43HouXBdLBTJBqH5rzDQm-UJy?usp=sharing**](https://drive.google.com/drive/folders/1DbFdSLy43HouXBdLBTJBqH5rzDQm-UJy?usp=sharing)&nbsp;&nbsp; |
| Model weights (`unet_sentinel1_finetune_best.pth`) | [**https\://drive.google.com/file/d/1An-uf-vOIE86nWj0z59S5c4DkwqjLvbh/view?usp=sharing**](https://drive.google.com/file/d/1An-uf-vOIE86nWj0z59S5c4DkwqjLvbh/view?usp=sharing)&nbsp; |
| Executed training/evaluation/prediction notebooks | **ADD\_GITHUB\_NOTEBOOKS\_LINK** |
| Streamlit prototype source code | **ADD\_GITHUB\_APP\_LINK** |

Test every public link in a private/incognito browser window. Do not upload credentials, access tokens, service account keys, or private Drive material to a public repository.

## References

- [Copernicus Sentinel-1 GRD, Google Earth Engine Data Catalog](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD)  
- [ESA WorldCover 2021, v200](https://doi.org/10.5281/zenodo.7254221)  
- Ronneberger, Fischer & Brox (2015). [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597).  
- He, Zhang, Ren & Sun (2016). [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).  
- Gorelick et al. (2017). [Google Earth Engine: Planetary-scale geospatial analysis for everyone](https://doi.org/10.1016/j.rse.2017.06.031).