import os, io, csv
from datetime import datetime
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(page_title="CocoMaturity AI", page_icon="🥥", layout="wide")
st.markdown("""
<style>
.main {background: linear-gradient(180deg,#f3fff6 0%,#ffffff 42%);}
.hero {padding:1.2rem 1.4rem;border-radius:18px;background:linear-gradient(110deg,#075b3a,#15945e);color:white;margin-bottom:1rem}
.hero h1 {color:white;margin:0}
.metric-card {padding:1rem;border:1px solid #dce9df;border-radius:14px;background:#fbfffc}
</style>
<div class="hero"><h1>🥥 CocoMaturity AI</h1><p>AI-assisted coconut class assessment · upload a photo or use your camera</p></div>
""", unsafe_allow_html=True)

MODEL_PATH = Path(__file__).parent / "models" / "coconut_hog_svm.joblib"
CLASS_GUIDE = {
    "tender": ("Tender coconut", "The model associates this image with the tender class in the training dataset."),
    "green": ("Green coconut", "The model associates this image with the green class in the training dataset."),
    "dry": ("Dry coconut", "The model associates this image with the dry class in the training dataset."),
}

@st.cache_resource(show_spinner=False)
def load_bundle(path_string, mtime):
    return joblib.load(path_string)

def build_hog():
    return cv2.HOGDescriptor(_winSize=(64,64), _blockSize=(16,16), _blockStride=(8,8), _cellSize=(8,8), _nbins=9)

HOG = build_hog()

def feature_for(image_bgr):
    resized = cv2.resize(image_bgr, (64,64), interpolation=cv2.INTER_AREA)
    return HOG.compute(resized).reshape(-1).astype(np.float32)

def image_views(pil_image):
    """Create a few plausible crops for one prominent coconut in the photo."""
    rgb = np.array(pil_image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    side = min(h, w)
    # Center square crop and a slightly tighter center crop; model was trained on annotated coconut crops.
    x0, y0 = (w-side)//2, (h-side)//2
    square = bgr[y0:y0+side, x0:x0+side]
    sh, sw = square.shape[:2]
    inset_x, inset_y = int(sw*.08), int(sh*.08)
    tighter = square[inset_y:sh-inset_y or sh, inset_x:sw-inset_x or sw]
    return [square, tighter if tighter.size else square]

def assess(pil_image):
    if not MODEL_PATH.exists():
        raise FileNotFoundError("The trained model file is missing from models/coconut_hog_svm.joblib")
    bundle = load_bundle(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
    model = bundle["model"]
    names = bundle.get("classes", ["dry", "green", "tender"])
    probs=[]
    for view in image_views(pil_image):
        probs.append(model.predict_proba(feature_for(view).reshape(1,-1))[0])
    mean_prob = np.mean(probs, axis=0)
    best = int(np.argmax(mean_prob))
    label = str(names[best])
    confidence = float(mean_prob[best])
    annotated = pil_image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    text = f"Prediction: {label.upper()}  |  {confidence:.1%} model score"
    draw.rectangle((8,8,min(annotated.width-8, 8+len(text)*8),45), fill=(8,85,53))
    draw.text((16,17), text, fill=(255,255,255))
    details = [{"Class": str(names[i]), "Model score": round(float(mean_prob[i]),4), "Percent": f"{mean_prob[i]*100:.1f}%"} for i in range(len(names))]
    return annotated, label, confidence, details, bundle

st.write("For best results, photograph one whole coconut clearly in good light. The included trained model is a lightweight HOG + Support Vector Machine (SVM) classifier trained on the labeled coconut dataset; it is not YOLOv8 and does not draw object-detection boxes.")
left, right = st.columns([1,1], gap="large")
with left:
    mode = st.radio("Image source", ["Upload photo", "Use camera"], horizontal=True)
    upload = st.file_uploader("Choose a coconut image", type=["jpg","jpeg","png","webp"]) if mode == "Upload photo" else st.camera_input("Capture coconut photo")
    run = st.button("Assess coconut", type="primary", use_container_width=True)
with right:
    st.subheader("Included features")
    st.markdown("- **Trained image classifier:** predicts `dry`, `green`, or `tender` from the supplied dataset classes.\n- **Score breakdown:** displays each class score.\n- **Camera and image upload:** use a phone or computer camera or upload an image.\n- **CSV export and session history:** save results for your project demo.\n- **No paid API key:** prediction runs locally inside the Streamlit app.")
    st.caption("A model score is not a calibrated probability of true biological maturity. Image-based estimates cannot directly measure internal water volume, kernel thickness, or exact harvest readiness.")

if upload is not None:
    try:
        image = Image.open(upload).convert("RGB")
        st.image(image, caption="Input image", use_container_width=True)
        if run:
            with st.spinner("Assessing the coconut image with the trained model..."):
                annotated, label, confidence, details, bundle = assess(image)
            result_time = datetime.now().isoformat(timespec="seconds")
            st.session_state["last_result"] = {"annotated":annotated, "label":label, "confidence":confidence, "details":details, "time":result_time, "validation_accuracy":bundle.get("validation",{}).get("accuracy"), "test_accuracy":bundle.get("test",{}).get("accuracy")}
            st.session_state.setdefault("history", []).append({"Time":result_time,"Predicted class":label,"Model score":round(confidence,4)})
    except Exception as e:
        st.error(f"Could not read or assess this image: {e}")

if "last_result" in st.session_state:
    r=st.session_state["last_result"]
    st.divider(); st.subheader("Maturity class assessment")
    c1,c2=st.columns([1.2,1])
    with c1:
        st.image(r["annotated"], caption="Model output", use_container_width=True)
    with c2:
        title, description = CLASS_GUIDE.get(r["label"], (r["label"], "Predicted dataset class."))
        st.metric("Predicted class", title)
        st.progress(max(0.0,min(1.0,float(r["confidence"]))))
        st.write(f"Top class score: **{r['confidence']:.1%}**")
        st.caption(description)
        st.dataframe(pd.DataFrame(r["details"]), use_container_width=True, hide_index=True)
        df=pd.DataFrame([{"Predicted class":r["label"],"Top class score":round(r["confidence"],4),"Time":r["time"]}])
        st.download_button("Download assessment CSV", df.to_csv(index=False).encode("utf-8"), "coconut_assessment.csv", "text/csv")
        if r.get("test_accuracy") is not None:
            st.caption(f"Training evaluation: held-out test accuracy {r['test_accuracy']:.1%} on the supplied test objects. Your results may differ on real-world photos.")
    st.info("Use this as an educational screening estimate only. Check the coconut physically before making harvesting, purchasing, or commercial grading decisions.")

if st.session_state.get("history"):
    with st.expander("Session history"):
        hist=pd.DataFrame(st.session_state["history"])
        st.dataframe(hist,use_container_width=True,hide_index=True)
        st.download_button("Download session history",hist.to_csv(index=False).encode(),"coconut_session_history.csv","text/csv")

with st.expander("Model information and limitations"):
    st.write("Model: OpenCV HOG features + scikit-learn SVM classifier. Training samples were cropped from the dataset's annotated coconut bounding boxes; horizontal flips were used for training augmentation. The supplied test split contains 44 annotated coconut objects.")
    st.write("Held-out test accuracy on the provided dataset was approximately 68.2% for object crops. This is a small test set and may not represent performance on photos taken in different lighting, backgrounds, varieties, or camera conditions. The current web app classifies the main image crop and does not locate multiple coconuts in one frame.")
