import os
from ultralytics import YOLO
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Define your checkpoints with full paths
checkpoints = {
    'cp1': 'datasets/shopping-trolley-5/runs/detect/pretrainedv8n/weights/bestpretrainedv8n.pt',
    'cp2': 'datasets/shopping-trolley-5/runs/detect/pretrainedv8n/weights/lastpretrainedv8n.pt',
    'cp3': 'datasets/shopping-trolley-5/runs/detect/unpretrainedv8n/weights/bestpretrainedv8n.pt',
    'cp4': 'datasets/shopping-trolley-5/runs/detect/unpretrainedv8n/weights/lastunpretrainedv8n.pt'
}

# Directory containing test images
test_images_dir = 'datasets/shopping-trolley-5/test/images'
test_images = [os.path.join(test_images_dir, img) for img in os.listdir(test_images_dir) if img.endswith('.jpg')]

# Define the directory for saving predictions
predictions_dir = 'datasets/shopping-trolley-5/test/predictions'
os.makedirs(predictions_dir, exist_ok=True)  # Create the directory if it does not exist

# Directory containing labels in some format, adjust the reading mechanism according to your label format
label_dir = 'datasets/shopping-trolley-5/test/labels'

# Function to load labels, this needs to be adapted to your label format
def load_labels(image_path):
    label_path = os.path.splitext(image_path.replace('images', 'labels'))[0] + '.txt'
    try:
        # Attempt to load labels from the file
        labels = np.loadtxt(label_path, delimiter=' ')
        if labels.size == 0:
            print(f"No label data found in {label_path}, skipping.")
            return None
        # Ensure labels are two-dimensional even if there is only one label
        if labels.ndim == 1:
            labels = np.expand_dims(labels, axis=0)
    except IOError:
        print(f"Label file not found for {image_path}, skipping.")
        return None
    except ValueError:
        print(f"Could not read label data correctly in {label_path}, skipping.")
        return None
    return labels[:, 0]  # Assuming first column is class

# Analyze each checkpoint
for name, cp_path in checkpoints.items():
    model = YOLO(cp_path)  # Load model with checkpoint

    all_true_classes = []
    all_pred_classes = []
    all_pred_probs = []

    for img_path in test_images:
        true_labels = load_labels(img_path)
        if true_labels is None:
            continue  # Skip this image if labels are not loaded correctly

        results = model(img_path)
        prediction_details = []
        detected_objects = 0  # Track the number of detected objects

        for result in results:
            if result.boxes and result.boxes.xyxy.size:
                detected_objects += len(result.boxes.cls)
                for cls, conf in zip(result.boxes.cls, result.boxes.conf):
                    if len(all_true_classes) > len(all_pred_probs):  # Only add predictions if there are labels to match them
                        all_pred_probs.append(conf)
                        all_pred_classes.append(cls)

        if prediction_details:
            with open(os.path.join(predictions_dir, os.path.basename(img_path).replace('.jpg', '_pred.txt')), 'w') as f:
                f.write('\n'.join(prediction_details))

        # Match or note mismatch
        if detected_objects != len(true_labels):
            print(f"Warning: Number of detections ({detected_objects}) does not match number of labels ({len(true_labels)}) for image: {img_path}")

    # Ensure all_pred_probs is not longer than all_true_classes
    all_pred_probs = all_pred_probs[:len(all_true_classes)]

    # Calculate metrics
    if all_true_classes and all_pred_classes:
        fpr, tpr, _ = roc_curve(all_true_classes, all_pred_probs)
        roc_auc = auc(fpr, tpr)
        accuracy = accuracy_score(all_true_classes, all_pred_classes)
        precision, recall, fscore, _ = precision_recall_fscore_support(all_true_classes, all_pred_classes, average='weighted')
        cm = confusion_matrix(all_true_classes, all_pred_classes)

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver operating characteristic for {name}')
        plt.legend(loc="lower right")
        plt.show()

        # Output the metrics
        print(f'Checkpoint: {name}')
        print(f'Accuracy: {accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F-Score: {fscore}')
        print('Confusion Matrix:')
        print(cm)
    else:
        print("No valid data to process for metrics calculation.")
