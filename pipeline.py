# ============================================================
# Cell 1: Imports
# ============================================================

import cv2
import numpy as np
import pandas as pd


# ============================================================
# Cell 2: Task 1 - Image Acquisition & Preprocessing
# ============================================================
# Requirements:
# - Collect the required images
# - Convert images to grayscale
# - Remove noise
# - Adjust brightness/contrast
# ============================================================

def preprocess_image(img):
    """
    Task 1:
    Convert image to grayscale, adjust brightness/contrast,
    and remove noise.
    """

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adjust brightness and contrast
    alpha = 1.5   # contrast
    beta = 30     # brightness
    enhanced = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

    # MISSING: Extra noise removal was added to improve preprocessing
    denoised = cv2.medianBlur(enhanced, 5)

    # Smooth image to help segmentation
    preprocessed = cv2.GaussianBlur(denoised, (5, 5), 0)

    return gray, enhanced, preprocessed


# ============================================================
# Cell 3: Task 2 - Segmentation & Object Detection
# ============================================================
# Requirements:
# - Separate objects from the background using different methods
# - Detect and locate objects in the image
# - Prepare segmented images with clear objects
# ============================================================

def segment_image(preprocessed_gray):
    """
    Task 2:
    Segment image using Otsu and Adaptive Thresholding,
    then clean the mask using morphological operations.
    """

    average_brightness = np.mean(preprocessed_gray)

    # Method 1: Otsu Thresholding
    if average_brightness > 127:
        # Light background
        _, thresh_otsu = cv2.threshold(
            preprocessed_gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
    else:
        # Dark background
        _, thresh_otsu = cv2.threshold(
            preprocessed_gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    # Method 2: Adaptive Thresholding
    thresh_adaptive = cv2.adaptiveThreshold(
        preprocessed_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )

    # Prepare a clear mask using morphological operations
    kernel = np.ones((3, 3), np.uint8)

    clear_mask = cv2.morphologyEx(
        thresh_otsu,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    clear_mask = cv2.morphologyEx(
        clear_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    return thresh_otsu, thresh_adaptive, clear_mask


def detect_objects(clear_mask, original_img, min_area=100):
    """
    Task 2:
    Detect objects using contours and locate them using bounding boxes.
    """

    contours, _ = cv2.findContours(
        clear_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    valid_contours = []
    detected_img = original_img.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area >= min_area:
            valid_contours.append(cnt)

            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(
                detected_img,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

    return valid_contours, detected_img


# ============================================================
# Cell 4: Task 3 - Counting Algorithm & Validation
# ============================================================
# Requirements:
# - Implement the code to count objects
# - Handle overlapping objects
# - Test results on different images to ensure accuracy
# ============================================================

# MISSING: Basic object counting was missing in the original code
def count_objects_basic(contours):
    """
    Count objects using detected contours.
    """
    return len(contours)


# MISSING: Shape analysis was added to decide when Watershed is actually needed
def is_suspicious_overlap(cnt, median_area=None):
    """
    Decide whether a contour may contain overlapping or touching objects.

    The contour is considered suspicious if:
    - It is much larger than the median object area
    - It has low circularity
    - It has an unusual aspect ratio

    This prevents applying Watershed to every object.
    """

    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    if perimeter == 0 or area == 0:
        return False

    circularity = (4 * np.pi * area) / (perimeter ** 2)

    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / h if h != 0 else 0

    large_area = False
    if median_area is not None and median_area > 0:
        large_area = area > 1.45 * median_area

    low_circularity = circularity < 0.65
    unusual_aspect = aspect_ratio > 1.45 or aspect_ratio < 0.70

    if large_area and (low_circularity or unusual_aspect):
        return True

    return False


# MISSING: Selective Watershed was added to handle overlap without over-segmentation
def selective_watershed_for_overlapping_objects(original_img, clear_mask, basic_contours, min_area=100):
    """
    Hybrid counting method:

    1. Use basic contours as the default method.
    2. Apply Watershed only on suspicious contours.
    3. Keep normal contours unchanged.

    This balances between:
    - Basic Contours: stable for separated objects
    - Watershed: useful for touching or overlapping objects
    """

    if len(basic_contours) == 0:
        return 0, original_img.copy(), [], []

    areas = [
        cv2.contourArea(cnt)
        for cnt in basic_contours
        if cv2.contourArea(cnt) >= min_area
    ]

    median_area = np.median(areas) if len(areas) > 0 else None

    final_contours = []
    suspicious_contours = []
    hybrid_result = original_img.copy()

    for cnt in basic_contours:

        # If the contour does not look like overlap, keep it as it is
        if not is_suspicious_overlap(cnt, median_area):
            final_contours.append(cnt)

            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(
                hybrid_result,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )
            continue

        # If contour is suspicious, apply Watershed locally
        suspicious_contours.append(cnt)

        local_mask = np.zeros(clear_mask.shape, dtype="uint8")
        cv2.drawContours(local_mask, [cnt], -1, 255, -1)

        x, y, w, h = cv2.boundingRect(cnt)

        pad = 10
        x1 = max(x - pad, 0)
        y1 = max(y - pad, 0)
        x2 = min(x + w + pad, clear_mask.shape[1])
        y2 = min(y + h + pad, clear_mask.shape[0])

        roi_mask = local_mask[y1:y2, x1:x2]
        roi_img = original_img[y1:y2, x1:x2].copy()

        kernel = np.ones((3, 3), np.uint8)

        opening = cv2.morphologyEx(
            roi_mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1
        )

        sure_bg = cv2.dilate(
            opening,
            kernel,
            iterations=2
        )

        dist_transform = cv2.distanceTransform(
            opening,
            cv2.DIST_L2,
            5
        )

        if dist_transform.max() == 0:
            final_contours.append(cnt)

            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(
                hybrid_result,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )
            continue

        # Higher threshold reduces over-segmentation
        _, sure_fg = cv2.threshold(
            dist_transform,
            0.60 * dist_transform.max(),
            255,
            0
        )

        sure_fg = np.uint8(sure_fg)

        unknown = cv2.subtract(sure_bg, sure_fg)

        _, markers = cv2.connectedComponents(sure_fg)

        markers = markers + 1
        markers[unknown == 255] = 0

        markers = cv2.watershed(roi_img, markers)

        local_split_contours = []

        for marker_id in np.unique(markers):
            if marker_id <= 1:
                continue

            object_mask = np.zeros(roi_mask.shape, dtype="uint8")
            object_mask[markers == marker_id] = 255

            cnts, _ = cv2.findContours(
                object_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            for split_cnt in cnts:
                split_area = cv2.contourArea(split_cnt)

                if split_area >= min_area:
                    # Convert ROI contour coordinates back to full image coordinates
                    split_cnt[:, :, 0] += x1
                    split_cnt[:, :, 1] += y1

                    local_split_contours.append(split_cnt)

        # Safeguard:
        # Accept Watershed only if it creates a reasonable split.
        # If it creates too many parts, reject it and keep the original contour.
        if 2 <= len(local_split_contours) <= 4:
            for split_cnt in local_split_contours:
                final_contours.append(split_cnt)

                sx, sy, sw, sh = cv2.boundingRect(split_cnt)
                cv2.rectangle(
                    hybrid_result,
                    (sx, sy),
                    (sx + sw, sy + sh),
                    (255, 0, 0),
                    2
                )
        else:
            final_contours.append(cnt)

            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(
                hybrid_result,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

    final_count = len(final_contours)

    return final_count, hybrid_result, final_contours, suspicious_contours


# MISSING: Validation was missing in the original code
def validate_count(predicted_count, true_count=None):
    """
    Validate counting result if the true count is known.
    """

    if true_count is None:
        return None

    if true_count <= 0:
        return None

    error = abs(predicted_count - true_count)
    accuracy = max(0, (1 - error / true_count) * 100)

    validation_result = {
        "True Count": true_count,
        "Predicted Count": predicted_count,
        "Error": error,
        "Accuracy (%)": round(accuracy, 2)
    }

    return validation_result


# ============================================================
# Cell 5: Task 4 - Feature Extraction
# ============================================================
# Requirements:
# - Calculate area and perimeter of each object
# - Extract object shape: aspect ratio and circularity
# - Organize features into structured data
# ============================================================

# MISSING: The original code used 'features' but did not create it
# MISSING: Full feature extraction was missing in the original code
def extract_features(contours):
    """
    Extract area, perimeter, aspect ratio, circularity,
    and centroid for each detected object.
    """

    features = []

    for i, cnt in enumerate(contours, start=1):
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        x, y, w, h = cv2.boundingRect(cnt)

        aspect_ratio = w / h if h != 0 else 0

        if perimeter != 0:
            circularity = (4 * np.pi * area) / (perimeter ** 2)
        else:
            circularity = 0

        M = cv2.moments(cnt)

        if M["m00"] != 0:
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])
        else:
            centroid_x = x + w // 2
            centroid_y = y + h // 2

        features.append({
            "Object ID": i,
            "Area (px)": round(area, 2),
            "Perimeter (px)": round(perimeter, 2),
            "Width": w,
            "Height": h,
            "Aspect Ratio": round(aspect_ratio, 3),
            "Circularity": round(circularity, 3),
            "Centroid X": centroid_x,
            "Centroid Y": centroid_y
        })

    features_df = pd.DataFrame(features)

    return features_df


# MISSING: Object ID visualization was added to make feature extraction clearer
def draw_object_ids(original_img, contours):
    """
    Draw object IDs on the final detected objects.
    """

    output_img = original_img.copy()

    for i, cnt in enumerate(contours, start=1):
        M = cv2.moments(cnt)

        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            x, y, w, h = cv2.boundingRect(cnt)
            cx = x + w // 2
            cy = y + h // 2

        cv2.putText(
            output_img,
            str(i),
            (cx - 10, cy + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    return output_img


# MISSING: Feature visualization charts were added to summarize extracted features
def show_feature_charts(features_df):
    """
    Display area and circularity charts.
    """

    if features_df is None or len(features_df) == 0:
        print("No features to display.")
        return

    obj_ids = features_df["Object ID"].tolist()
    areas = features_df["Area (px)"].tolist()
    circularities = features_df["Circularity"].tolist()

    plt.figure(figsize=(14, 4))

    plt.subplot(1, 2, 1)
    plt.bar(obj_ids, areas)
    plt.xlabel("Object ID")
    plt.ylabel("Area")
    plt.title("Area per Object")
    plt.xticks(obj_ids)

    plt.subplot(1, 2, 2)
    plt.bar(obj_ids, circularities)
    plt.axhline(y=1.0, linestyle="--", label="Perfect Circle")
    plt.xlabel("Object ID")
    plt.ylabel("Circularity")
    plt.title("Circularity per Object")
    plt.xticks(obj_ids)
    plt.legend()

    plt.suptitle("Feature Summary", fontweight="bold")
    plt.tight_layout()
    plt.show()


# ============================================================
# Cell 6: Run Tasks 1 to 4
# ============================================================

def run_tasks_1_to_4(img, true_count=None, min_area=100):
    """
    Run Tasks 1 to 4 in the correct order:

    1. Image Acquisition & Preprocessing
    2. Segmentation & Object Detection
    3. Counting Algorithm & Validation
    4. Feature Extraction
    """

    # ------------------------------------------------------------
    # Task 1
    # ------------------------------------------------------------

    gray, enhanced, preprocessed = preprocess_image(img)

    # ------------------------------------------------------------
    # Task 2
    # ------------------------------------------------------------

    thresh_otsu, thresh_adaptive, clear_mask = segment_image(preprocessed)

    basic_contours, detected_img = detect_objects(
        clear_mask,
        img,
        min_area=min_area
    )

    # ------------------------------------------------------------
    # Task 3
    # ------------------------------------------------------------

    basic_count = count_objects_basic(basic_contours)

    final_count, hybrid_result, final_contours, suspicious_contours = selective_watershed_for_overlapping_objects(
        original_img=img,
        clear_mask=clear_mask,
        basic_contours=basic_contours,
        min_area=min_area
    )

    validation_result = validate_count(
        predicted_count=final_count,
        true_count=true_count
    )

    # ------------------------------------------------------------
    # Task 4
    # ------------------------------------------------------------

    features_df = extract_features(final_contours)

    object_ids_img = draw_object_ids(img, final_contours)

    return {
        "original": img,
        "gray": gray,
        "enhanced": enhanced,
        "preprocessed": preprocessed,
        "otsu_threshold": thresh_otsu,
        "adaptive_threshold": thresh_adaptive,
        "clear_mask": clear_mask,
        "basic_contours": basic_contours,
        "final_contours": final_contours,
        "suspicious_contours": suspicious_contours,
        "basic_count": basic_count,
        "final_count": final_count,
        "features_df": features_df,
        "validation_result": validation_result,
        "detected_img": detected_img,
        "hybrid_result": hybrid_result,
        "object_ids_img": object_ids_img,
    }
