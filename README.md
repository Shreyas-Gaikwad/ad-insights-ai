# Ad Insights AI

https://github.com/user-attachments/assets/6328324a-8ee3-4873-8b12-f6c1272b3844

An AI-powered **advertising intelligence and visual analytics system** that analyzes advertising creatives and predicts their underlying psychological attributes using computer vision and machine learning.

The project uses a **DINOv2 Vision Transformer (ViT-L/14)** to extract visual representations from advertising images, followed by six independently tuned Ridge regression models to predict the attributes associated with each creative.

The goal is to move beyond subjective ad analysis by providing structured, quantitative insights into how advertisements communicate brand identity, evoke emotions, and position products.

**Tech Stack:**  
Python · PyTorch · DINOv2 · Hugging Face Transformers · Scikit-learn · NumPy · Pandas · PyQt6

---

## Problem Statement

Advertising creatives communicate brand identity through visual elements such as composition, color, typography, product placement, background design, and human subjects.

However, evaluating these elements and the psychological impressions they create can be subjective, inconsistent, and difficult to scale across large collections of advertisements.

The goal of this project is to develop a computer vision-based prediction system that:

- Analyzes advertising images using visual representations.
- Predicts multiple psychological attributes for each advertisement.
- Converts visual analysis into interpretable numerical scores.
- Provides a consistent framework for comparing advertising creatives.
- Offers a desktop interface for convenient advertising analysis.

---

## Key Features

- **Vision Transformer-based feature extraction** using DINOv2 ViT-L/14.
- **Six independent attribute prediction models** trained using Ridge regression.
- **Manual dataset annotation** with partial assistance from a structured Google Form.
- **Independent hyperparameter tuning** for each attribute-specific regression model.
- **Continuous attribute predictions** displayed on a 1–100 scale.
- **Frozen feature extractor** that leverages visual representations without fine-tuning the backbone.
- **Regularized regression** to map high-dimensional visual embeddings to manually assigned attribute labels.
- **Desktop application** built with PyQt6 for interactive advertising analysis.
- **Reusable model artifacts** for generating predictions on new advertising creatives.

---

## Psychological Attribute Prediction

The system predicts six psychological attributes for each advertisement.

| Attribute | Description |
|---|---|
| Innovation | The extent to which an advertisement communicates novelty, originality of approach, or forward-thinking ideas. |
| Premium | The degree to which the creative conveys luxury, sophistication, exclusivity, or high-end positioning. |
| Minimalism | The extent to which the advertisement uses simplicity, visual restraint, clean composition, and limited clutter. |
| Excitement | The level of energy, dynamism, enthusiasm, or emotional intensity conveyed by the creative. |
| Social Orientation | The degree to which the advertisement emphasizes social relationships, family, community, or shared experiences. |
| Aspiration | The extent to which the advertisement communicates ambition, an ideal lifestyle, achievement, or personal growth. |

Each attribute is predicted independently, allowing an advertisement to receive different scores across multiple dimensions.

**Important:** These scores represent model predictions of the visual attributes learned from the labeled dataset. They should not be interpreted as objective measurements of consumer psychology or actual advertising effectiveness.

---

## Dataset Preparation

The dataset consists of advertising images paired with manually assigned numerical labels for six psychological attributes.

Each image is associated with an asset identifier, its image path, and its corresponding attribute scores.

The dataset preparation process includes:

- Mapping image files to their corresponding labels.
- Validating file paths and supported image formats.
- Handling different image dimensions through preprocessing.
- Checking missing labels and invalid image files.
- Preparing consistent training and evaluation datasets.

### Manual Data Labeling

The advertising dataset was **manually labeled** to assign numerical scores to the six psychological attributes: Innovation, Premium, Minimalism, Excitement, Social Orientation, and Aspiration.

A Google Form provided partial assistance in the labeling workflow by structuring the evaluation of advertising creatives and supporting a consistent annotation process.

**Annotation resource:** [Ad Attribute Labeling — Google Form](https://forms.gle/QV4oWX9LfTgdqN3s8)

- **Labeling approach:** Manual annotation of advertising images.
- **Annotation tool:** Google Forms, used for partial assistance with the labeling workflow.
- **Label format:** Numerical scores for each of the six psychological attributes.
- **Purpose:** Create a labeled dataset for supervised regression model training.

The resulting labels serve as the target values for training the six independent Ridge regression models. The DINOv2 feature extractor generates visual embeddings, while the regression models learn associations between those embeddings and the manually assigned attribute scores.

The Google Form assisted with the labeling workflow; the target labels were manually assigned rather than automatically generated by an AI model.

### Data Quality Considerations

The quality of supervised predictions depends on the consistency and reliability of the manually assigned labels. Subjective attributes may involve differences in interpretation, so the resulting scores should be understood as annotation-based judgments rather than universal psychological measurements.

---

## System Architecture

```text
              Advertising Image
                      |
                      v
           Image Preprocessing
                      |
                      v
               DINOv2 ViT-L/14
              (Frozen Backbone)
                      |
                      v
           Visual Feature Embedding
                      |
          +-----------+-----------+
          |           |           |
          v           v           v
       Ridge       Ridge       Ridge
       Model       Model       Model
     Innovation   Premium    Minimalism
          |           |           |
          +-----------+-----------+
                      |
          +-----------+-----------+
          |           |           |
          v           v           v
       Ridge       Ridge       Ridge
       Model       Model       Model
     Excitement    Social     Aspiration
                 Orientation
                      |
                      v
           Six Attribute Predictions
                      |
                      v
           Score Scaling (1–100)
                      |
                      v
             PyQt6 Desktop App
```

---

## Model Development Approach

### 1. Dataset Preparation and Annotation

Advertising images are paired with manually assigned scores for each psychological attribute.

The labeled dataset provides the ground-truth targets used during supervised learning. Image identifiers and file paths are used to associate each image with its corresponding labels.

Before training, the dataset should be checked for missing values, duplicate images, invalid paths, and inconsistent annotations.

### 2. Visual Feature Extraction

The project uses **DINOv2 ViT-L/14** as a frozen feature extractor.

Rather than training a deep neural network from scratch, the system leverages visual representations to extract embeddings from advertising images.

These embeddings encode useful visual information, including patterns related to image composition, objects, textures, and semantic content.

The extracted embeddings are used as input features for downstream regression models.

### 3. Attribute Prediction

Six separate Ridge regression models are trained to predict:

1. Innovation
2. Premium
3. Minimalism
4. Excitement
5. Social Orientation
6. Aspiration

Each model is independently tuned for its corresponding prediction target.

Ridge regression applies L2 regularization, helping control model complexity when learning from high-dimensional visual embeddings.

The prediction pipeline can be represented as:

```text
Advertising Image
        |
        v
DINOv2 Feature Extractor
        |
        v
Visual Embedding
        |
        +--> Innovation Model
        +--> Premium Model
        +--> Minimalism Model
        +--> Excitement Model
        +--> Social Orientation Model
        +--> Aspiration Model
        |
        v
Six Predicted Attribute Scores
```

### 4. Prediction Output

For a new advertising image, the system extracts its visual embedding and passes it through all six trained regressors.

The output is a set of six continuous attribute scores, presented on a 1–100 scale.

The scores provide a structured representation of the visual characteristics the models have learned to associate with each attribute.

Score scaling is a presentation step; it does not guarantee that predictions are calibrated probabilities or that a score of 80 corresponds to twice the psychological intensity of a score of 40.

---

## Model Evaluation

The regression models should be evaluated on held-out data that was not used during training or hyperparameter tuning.

Recommended evaluation metrics include:

| Metric | Purpose |
|---|---|
| MAE | Measures the average absolute prediction error. |
| RMSE | Penalizes larger prediction errors more heavily. |
| R² | Measures predictive performance relative to a mean-prediction baseline. |

Evaluation should be conducted separately for each of the six attributes because their labeling consistency, score distributions, and prediction difficulty may differ.

The evaluation process should also check for:

- Data leakage between training and test sets.
- Duplicate or near-duplicate creatives across dataset splits.
- Brand or campaign overlap between training and testing data.
- Performance differences across brands and advertising styles.
- Sensitivity to subjective or inconsistent annotations.

**Note:** Add the actual evaluation metrics once they have been measured on the held-out dataset. No performance figures are assumed in this README.

---

## Desktop Application

The project includes a desktop interface built with PyQt6 for analyzing advertising creatives.

The application is designed to load the saved model artifacts, accept an advertising image, and display the six predicted attribute scores.

### Intended Workflow

1. Launch the desktop application.
2. Select an advertising image.
3. Run the prediction pipeline.
4. View the six predicted psychological attribute scores.
5. Compare the predicted visual characteristics of different advertisements.

The interface provides a convenient way to inspect predictions without manually running the underlying feature extraction and regression code.

---

## Project Structure

```text
ad-insights-ai/
├── data/
│   └── apple
│   └── bmw
│   └── coke
│   └── mcdonalds
│   └── nike                      # Dataset containing images and videos
│
├── outputs/
│   └── dinov2_large              # Model artifact instructions
│       └── ...                   # Output Files and Saved Model
│
├── notebooks/
│   └── adInsightPrediction.ipynb
│
├── ad_insight_app.py
│
├── requirements.txt
└── README.md
```

*Adjust the structure above to match the actual files and directories in your repository.*

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Shreyas-Gaikwad/ad-insights-ai.git
cd ad-insights-ai
```

### 2. Create a Virtual Environment

Open PowerShell in the project directory and run:

```powershell
python -m venv .venv
```

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

Upgrade pip and install the application dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements_app.txt
```

---

## Usage

### 1. Verify the Model Bundle

Ensure that the trained model bundle is available at the expected path:

```text
outputs/
└── dinov2_large/
    └── six_attribute_ridge_models.joblib
```

The application uses the pretrained DINOv2 ViT-L/14 feature extractor and the six trained Ridge regression models. **The application does not train or fine-tune the models during execution.**

Make sure all required model artifacts are present before launching the application.

### 2. Launch the Application

From the project root directory, with the virtual environment activated, run:

```powershell
python ad_insight_app.py
```

### 3. Analyze Advertising Creatives

1. Launch the application.
2. Load an advertising image or supported video.
3. Run the prediction pipeline.
4. View the six predicted psychological attribute scores:
   - Innovation
   - Premium
   - Minimalism
   - Excitement
   - Social Orientation
   - Aspiration
5. Interpret the predictions on the displayed 1–100 scale.

The application provides an interface for analyzing advertising creatives using the saved model bundle without manually executing the training pipeline.

## Limitations

- Model in its current state does not fully generalize to videos due to the limited size of the training dataset.
- Predictions depend on the quality and consistency of the manually assigned labels.
- Visual embeddings may capture correlations in the training dataset rather than universally valid psychological signals.
- Different brands, industries, cultural contexts, and advertising styles may affect generalization.
- Predicted attribute scores do not directly establish consumer sentiment, purchase intent, or campaign effectiveness.
- Performance on unseen brands and advertising formats requires separate evaluation.
- Numerical scores may appear precise despite the subjective nature of the underlying attributes.

---

## Future Improvements

- Expand the manually labeled dataset across brands, industries, and advertising styles.
- Evaluate generalization on unseen brands and campaigns.
- Add systematic error analysis for each psychological attribute.
- Assess inter-annotator agreement and improve labeling guidelines.
- Explore improved calibration and more robust score interpretation.
- Extend the pipeline to video advertising through frame sampling and temporal aggregation.
- Develop comparative analytics across brands, campaigns, and creative variations.
- Validate predicted attributes against human ratings and real campaign outcome data.
- Explore additional visual and textual features for richer advertising analysis.

---

## Author

**Shreyas Gaikwad**

**Focus:** Applied Machine Learning · Computer Vision · Deep Learning · Advertising Intelligence · AI Applications
