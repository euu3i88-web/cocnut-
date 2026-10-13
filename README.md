# CocoMaturity AI — trained coconut image classifier

A Streamlit + OpenCV application that accepts an uploaded coconut photo or a camera capture and predicts one of the labeled dataset classes: `dry`, `green`, or `tender`.

## A trained model is included

This project includes `models/coconut_hog_svm.joblib`, trained in this environment using the coconut dataset supplied with the project. The model uses OpenCV HOG image features and a scikit-learn Support Vector Machine (SVM). It is a lightweight, CPU-friendly model and does not need an API key or GPU at runtime.

**Important distinction:** this included trained model is **not YOLOv8**. The supplied dataset is in YOLO annotation format, but network access was unavailable here to download the Ultralytics package and pretrained YOLOv8 starting weights. To give you an actual trained model now, I trained the included HOG + SVM classifier on coconut crops derived from the annotations. The app classifies the main image crop; it does not detect multiple coconuts or draw YOLO bounding boxes. The optional `train_yolov8.py` and Colab notebook are for training a YOLOv8 detector separately when you have Colab/network access.

## Evaluation

- Classes: `dry`, `green`, `tender`
- Training: 1,092 labeled training objects, with horizontal-flip augmentation (2,184 feature samples)
- Validation: 96 labeled objects; validation accuracy about 72.9%
- Held-out test: 44 labeled objects; test accuracy about **68.2%**

The test set is small, so this score is an initial evaluation rather than a guarantee of real-world accuracy. The model may perform differently on photos with other backgrounds, lighting, coconut varieties, or camera angles. A model score is not a calibrated probability of biological maturity.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud (free tier)

1. Download and extract `CocoMaturity_AI_Deploy_Code.zip`.
2. Upload its contents to a GitHub repository. Make sure the `models/coconut_hog_svm.joblib` file and `.streamlit/config.toml` are included.
3. In [Streamlit Community Cloud](https://share.streamlit.io/), create an app from your repository and choose `app.py` as the main file.
4. Deploy. No API key or secrets are needed.

You do not need to upload the `training_data/` directory for deployment. It is included only in the full project ZIP for training experiments. Keeping it out of the deployment repository reduces repository size.

## Included features

- Upload JPG, JPEG, PNG, or WEBP images
- Capture a photo using Streamlit camera input
- Predict dry / green / tender dataset class
- Show scores for all classes
- Export the assessment as CSV
- Session history and CSV export
- Lightweight CPU inference; no paid AI API

## Dataset source and attribution

Dataset: Coconut dataset version 5, provided via Roboflow by the `nit-calicut` workspace. The included metadata states CC BY 4.0. Review the original dataset page and license before redistribution or public demonstration: https://universe.roboflow.com/nit-calicut/coconut-veirf/dataset/5

## Limitations

A photo alone cannot directly measure internal water volume, kernel thickness, or exact harvest readiness. Treat this as an educational image-classification project, not as the sole basis for harvesting or commercial grading. For a stronger final-year project, collect more diverse labeled photos and evaluate them independently. The app currently assumes one prominent coconut in each image.
