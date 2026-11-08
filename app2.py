
# Banana Ripeness AI Dashboard

# Streamlit Cloud entrypoint
from app.app import *

import ast
import gc
import json
import tempfile
from io import BytesIO
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps


# ============================================
# 1. PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="Banana Ripeness AI",
    page_icon="🍌",
    layout="wide"
)

project_directory = Path(__file__).resolve().parents[1]
results_directory = project_directory / "results"

class_display_names = {
    "freshripe": "Fresh Ripe (กล้วยสุกสด)",
    "freshunripe": "Fresh Unripe (กล้วยดิบสด)",
    "overripe": "Overripe (กล้วยสุกงอม)",
    "ripe": "Ripe (กล้วยสุก)",
    "rotten": "Rotten (กล้วยเน่าเสีย)",
    "unripe": "Unripe (กล้วยดิบ)"
}

class_names = list(class_display_names.keys())

# การจัดกลุ่มเพื่อวิเคราะห์ ไม่ใช่โมเดลใหม่
stage_mapping = {
    "freshunripe": "Unripe Stage (กลุ่มกล้วยดิบ)",
    "unripe": "Unripe Stage (กลุ่มกล้วยดิบ)",
    "freshripe": "Ripe Stage (กลุ่มกล้วยสุก)",
    "ripe": "Ripe Stage (กลุ่มกล้วยสุก)",
    "overripe": "Late Stage (กลุ่มสุกงอม/เน่าเสีย)",
    "rotten": "Late Stage (กลุ่มสุกงอม/เน่าเสีย)"
}


# ============================================
# 2. LOAD SVM SYSTEM
# ============================================

@st.cache_resource(show_spinner=False)
def load_svm_system():

    model_path = (
        project_directory / "models" / "svm_hsv_best.joblib"
    )

    notebook_path = (
        project_directory / "notebooks" / "03_SVM.ipynb"
    )

    svm_bundle = joblib.load(model_path)

    if list(svm_bundle["class_names"]) != class_names:
        raise ValueError("ลำดับคลาสไม่ตรงกับโมเดล")

    with open(notebook_path, encoding="utf-8") as file:
        notebook_data = json.load(file)

    config = svm_bundle["feature_config"]

    namespace = {
        "np": np,
        "cv2": cv2,
        "Image": Image,
        "ImageOps": ImageOps,
        "image_size": tuple(config["image_size"]),
        "padding_color": tuple(config["padding_color"]),
        "bins_per_channel": config["bins_per_channel"]
    }

    for cell in notebook_data["cells"]:

        if cell["cell_type"] != "code":
            continue

        source = "".join(cell["source"])

        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        for node in tree.body:

            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "extract_hsv_features"
            ):

                function_module = ast.Module(
                    body=[node],
                    type_ignores=[]
                )

                exec(
                    compile(
                        function_module,
                        str(notebook_path),
                        "exec"
                    ),
                    namespace
                )

                return (
                    svm_bundle,
                    namespace["extract_hsv_features"]
                )

    raise RuntimeError("ไม่พบฟังก์ชัน HSV")


# ============================================
# 3. SVM PREDICTION
# ============================================

def predict_svm(uploaded_image):

    svm_bundle, extract_hsv_features = load_svm_system()

    with tempfile.TemporaryDirectory() as temp_dir:

        suffix = Path(uploaded_image.name).suffix.lower()

        image_path = (
            Path(temp_dir) / f"banana{suffix}"
        )

        image_path.write_bytes(
            uploaded_image.getvalue()
        )

        features = np.asarray(
            extract_hsv_features(str(image_path)),
            dtype=np.float32
        ).reshape(1, -1)

    expected_features = (
        svm_bundle["scaler"].n_features_in_
    )

    if features.shape[1] != expected_features:
        raise ValueError("จำนวน HSV Features ไม่ถูกต้อง")

    scaled_features = (
        svm_bundle["scaler"].transform(features)
    )

    prediction = (
        svm_bundle["model"].predict(scaled_features)[0]
    )

    predicted_class = svm_bundle[
        "class_names"
    ][int(prediction)]

    return predicted_class


# ============================================
# 4. LOAD DEEP LEARNING MODELS
# ============================================

@st.cache_resource(
    max_entries=1,
    show_spinner=False
)
def load_deep_model(model_name):

    import tensorflow as tf

    tf.keras.backend.clear_session()
    gc.collect()

    if model_name == "ResNet50":

        backbone = tf.keras.applications.ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=(224, 224, 3),
            pooling="avg"
        )

        head_filename = (
            "resnet50_classifier_baseline.keras"
        )

    elif model_name == "EfficientNetV2B0":

        backbone = tf.keras.applications.EfficientNetV2B0(
            weights="imagenet",
            include_top=False,
            input_shape=(224, 224, 3),
            pooling="avg",
            include_preprocessing=True
        )

        head_filename = (
            "efficientnetv2b0_classifier_baseline.keras"
        )

    else:
        raise ValueError("ไม่รู้จักโมเดล")

    backbone.trainable = False

    classifier = tf.keras.models.load_model(
        project_directory / "models" / head_filename,
        compile=False
    )

    if (
        int(classifier.input_shape[-1])
        != int(backbone.output_shape[-1])
    ):
        raise ValueError("Feature dimensions ไม่ตรงกัน")

    if int(classifier.output_shape[-1]) != len(class_names):
        raise ValueError("จำนวนคลาสไม่ตรงกัน")

    return backbone, classifier


# ============================================
# 5. DEEP LEARNING PREDICTION
# ============================================

def predict_deep(uploaded_image, model_name):

    import tensorflow as tf

    backbone, classifier = load_deep_model(
        model_name
    )

    with Image.open(
        BytesIO(uploaded_image.getvalue())
    ) as opened_image:

        resized = ImageOps.pad(
            opened_image.convert("RGB"),
            (224, 224),
            method=Image.Resampling.LANCZOS,
            color=(128, 128, 128)
        )

        image_batch = np.asarray(
            resized,
            dtype=np.float32
        )[None, ...]

    if model_name == "ResNet50":

        image_batch = (
            tf.keras.applications.resnet50.preprocess_input(
                image_batch
            )
        )

    # EfficientNetV2B0 มี Preprocessing ภายในแล้ว
    features = backbone(
        image_batch,
        training=False
    )

    scores = classifier(
        features,
        training=False
    ).numpy()[0]

    predicted_index = int(np.argmax(scores))

    return class_names[predicted_index]


# ============================================
# 6. SIDEBAR NAVIGATION
# ============================================

st.sidebar.title("🍌 Banana Ripeness AI")

st.sidebar.caption(
    "ระบบจำแนกระดับความสุกของกล้วย"
)

selected_page = st.sidebar.radio(
    "เมนูหลัก",
    [
        "🍌 Prediction",
        "📊 Model Comparison",
        "🔍 Model Analysis"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Best Validation Model (6 Classes): SVM"
)

st.sidebar.caption(
    "Best External Model (3 Stages): ResNet50"
)


# ============================================
# PAGE 1: PREDICTION
# ============================================

if selected_page == "🍌 Prediction":

    st.title("🍌 Banana Ripeness Prediction")

    st.write(
        "อัปโหลดภาพกล้วยและเลือกโมเดลที่ต้องการใช้ทำนาย"
    )

    selected_model = st.selectbox(
        "เลือกโมเดลสำหรับทำนาย",
        [
            "ResNet50",
            "EfficientNetV2B0",
            "SVM"
        ]
    )

    uploaded_image = st.file_uploader(
        "เลือกภาพกล้วย",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:

        image_column, result_column = st.columns(2)

        with image_column:

            st.image(
                uploaded_image,
                caption="ภาพที่อัปโหลด",
                width=350
            )

        with result_column:

            st.subheader("ผลการทำนาย")

            try:

                with st.spinner(
                    f"กำลังทำนายด้วย {selected_model}..."
                ):

                    if selected_model == "SVM":

                        predicted_class = predict_svm(
                            uploaded_image
                        )

                    else:

                        predicted_class = predict_deep(
                            uploaded_image,
                            selected_model
                        )

                display_name = class_display_names[
                    predicted_class
                ]

                st.success(f"🍌 {display_name}")

                st.caption(
                    f"Model: {selected_model}"
                )

                st.write(
                    "**กลุ่มระดับความสุก:** "
                    + stage_mapping[predicted_class]
                )

                st.caption(
                    "ผล 3 กลุ่มมาจากการรวมคลาส "
                    "ไม่ใช่โมเดล 3 คลาสที่ฝึกใหม่"
                )

            except Exception as error:

                st.error("เกิดข้อผิดพลาดในการทำนาย")
                st.exception(error)

    with st.expander(
        "📖 รายละเอียดระดับความสุกทั้ง 6 คลาส"
    ):

        for display_name in class_display_names.values():
            st.write(display_name)

    st.info(
        "ผลการทำนายอาจคลาดเคลื่อนเมื่อภาพ "
        "มีแสง พื้นหลัง หรือรูปแบบต่างจากข้อมูลฝึก"
    )


# ============================================
# PAGE 2: MODEL COMPARISON
# ============================================

elif selected_page == "📊 Model Comparison":

    st.title("📊 Model Performance Comparison")

    st.write(
        "เปรียบเทียบผล Validation ของ 4 โมเดล "
        "และผล External Evaluation ของ 3 โมเดล"
    )

    comparison_path = (
        results_directory / "validation_model_comparison.csv"
    )

    if comparison_path.exists():

        comparison_results = pd.read_csv(
            comparison_path
        )

        comparison_results = comparison_results[
            ["Model", "Accuracy", "Macro_F1"]
        ]

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "Best Validation Model",
            "SVM"
        )

        metric2.metric(
            "Validation Macro-F1",
            "96.77%"
        )

        metric3.metric(
            "SVM Screened Test Accuracy",
            "95.17%"
        )

        st.subheader(
            "Screened Validation Set (801 Images)"
        )

        st.dataframe(
            comparison_results,
            hide_index=True,
            width="stretch"
        )

        st.bar_chart(
            comparison_results,
            x="Model",
            y=["Accuracy", "Macro_F1"],
            stack=False,
            horizontal=True,
            sort=False,
            height=350
        )

        st.caption(
            "คะแนนส่วนนี้มาจาก Screened Validation Set "
            "จำนวน 801 ภาพ ส่วน Test Set 393 ภาพ "
            "มีผลประเมินเฉพาะ SVM"
        )

    else:

        st.warning(
            "ไม่พบไฟล์ validation_model_comparison.csv"
        )

    # External Evaluation
    external_path = (
        results_directory / "external_3model_predictions.csv"
    )

    if external_path.exists():

        st.divider()

        st.subheader(
            "External Evaluation (30 Images)"
        )

        external_df = pd.read_csv(external_path)

        external_models = [
            "SVM",
            "ResNet50",
            "EfficientNetV2B0"
        ]

        required_columns = (
            {"true_label"} | set(external_models)
        )

        if required_columns.issubset(external_df.columns):

            from sklearn.metrics import (
                accuracy_score,
                f1_score
            )

            rows = []

            actual_stages = (
                external_df["true_label"].map(stage_mapping)
            )

            for model_name in external_models:

                actual_labels = external_df["true_label"]
                predicted_labels = external_df[model_name]

                predicted_stages = (
                    predicted_labels.map(stage_mapping)
                )

                accuracy_6 = accuracy_score(
                    actual_labels,
                    predicted_labels
                )

                f1_6 = f1_score(
                    actual_labels,
                    predicted_labels,
                    labels=class_names,
                    average="macro",
                    zero_division=0
                )

                accuracy_3 = accuracy_score(
                    actual_stages,
                    predicted_stages
                )

                f1_3 = f1_score(
                    actual_stages,
                    predicted_stages,
                    average="macro",
                    zero_division=0
                )

                rows.append({
                    "Model": model_name,
                    "Accuracy 6 Classes (%)": round(
                        accuracy_6 * 100, 2
                    ),
                    "Macro-F1 6 Classes (%)": round(
                        f1_6 * 100, 2
                    ),
                    "Accuracy 3 Stages (%)": round(
                        accuracy_3 * 100, 2
                    ),
                    "Macro-F1 3 Stages (%)": round(
                        f1_3 * 100, 2
                    )
                })

            external_comparison = pd.DataFrame(rows)

            st.dataframe(
                external_comparison,
                hide_index=True,
                width="stretch"
            )

            st.caption(
                "ผล External Evaluation เป็นการทดลองเบื้องต้น "
                "จากภาพ 30 ภาพที่กำหนด Label เอง "
                "บางคลาสมีความกำกวม และผล 3 กลุ่ม "
                "เป็นการวิเคราะห์เพิ่มเติมหลังเห็นผล 6 คลาส"
            )

        else:

            st.warning(
                "ไฟล์ External Predictions มีคอลัมน์ไม่ครบ"
            )

    else:

        st.caption(
            "ยังไม่พบไฟล์ External Evaluation"
        )


# ============================================
# PAGE 3: MODEL ANALYSIS
# ============================================

elif selected_page == "🔍 Model Analysis":

    st.title("🔍 Model Analysis")

    st.write(
        "วิเคราะห์ Confusion Matrix และ Learning Curves"
    )

    selected_model = st.selectbox(
        "เลือกโมเดล",
        [
            "SVM",
            "EfficientNetV2B0",
            "ResNet50",
            "Custom CNN (Tuned)"
        ]
    )

    analysis_files = {

        "SVM": {
            "confusion": "svm_final_test_confusion_matrix.png",
            "learning": None,
            "dataset": "Screened Test Set (393 ภาพ)"
        },

        "EfficientNetV2B0": {
            "confusion": "efficientnetv2b0_confusion_matrix.png",
            "learning": "efficientnetv2b0_learning_curves.png",
            "dataset": "Screened Validation Set (801 ภาพ)"
        },

        "ResNet50": {
            "confusion": "resnet50_confusion_matrix.png",
            "learning": "resnet50_learning_curves.png",
            "dataset": "Screened Validation Set (801 ภาพ)"
        },

        "Custom CNN (Tuned)": {
            "confusion": "custom_cnn_tuned_confusion_matrix.png",
            "learning": None,
            "dataset": "Screened Validation Set (801 ภาพ)"
        }
    }

    selected_files = analysis_files[selected_model]

    st.info(
        f"ชุดข้อมูล: {selected_files['dataset']}"
    )

    st.subheader("Confusion Matrix")

    confusion_path = (
        results_directory / selected_files["confusion"]
    )

    if confusion_path.exists():

        st.image(
            str(confusion_path),
            width=700
        )

    else:

        st.warning("ไม่พบ Confusion Matrix")

    learning_filename = selected_files["learning"]

    if learning_filename is not None:

        st.subheader("Learning Curves")

        learning_path = (
            results_directory / learning_filename
        )

        if learning_path.exists():

            st.image(
                str(learning_path),
                width=900
            )

        else:

            st.warning("ไม่พบ Learning Curves")

    else:

        st.caption(
            "ไม่มี Learning Curves สำหรับโมเดลนี้"
        )


# ============================================
# FOOTER: ORIGINAL SVM RESULTS
# ============================================

st.divider()

metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric(
    "Best Validation Model",
    "SVM"
)

metric2.metric(
    "SVM Validation Macro-F1",
    "96.77%"
)

metric3.metric(
    "SVM Test Accuracy",
    "95.17%"
)

metric4.metric(
    "SVM Test Macro-F1",
    "93.09%"
)
