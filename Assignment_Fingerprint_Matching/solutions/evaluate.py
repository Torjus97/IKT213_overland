import os
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matchers import orb_match_fingerprints, sift_match_fingerprints


def evaluate_dataset(dataset_root, results_folder, method="orb"):
    os.makedirs(results_folder, exist_ok=True)
    y_true, y_pred = [], []
    threshold = 20

    print(f"\n--- Evaluating {method.upper()} pipeline ---\n")

    # Check if dataset_root itself contains images (single dataset case)
    image_files_in_root = [f for f in os.listdir(dataset_root)
                           if f.lower().endswith(('.png', '.jpg', '.tif'))]

    if len(image_files_in_root) >= 2:
        # This is a single dataset folder with images directly inside
        folders_to_process = [(os.path.basename(dataset_root), dataset_root)]
    else:
        # This contains multiple subfolders
        folders_to_process = []
        for folder in sorted(os.listdir(dataset_root)):
            folder_path = os.path.join(dataset_root, folder)
            if os.path.isdir(folder_path):
                folders_to_process.append((folder, folder_path))

    # Process each folder
    for folder_name, folder_path in folders_to_process:
        image_files = [f for f in os.listdir(folder_path)
                       if f.lower().endswith(('.png', '.jpg', '.tif'))]
        if len(image_files) != 2:
            print(f"Skipping {folder_name} — expected 2 images, found {len(image_files)}")
            continue

        img1_path = os.path.join(folder_path, image_files[0])
        img2_path = os.path.join(folder_path, image_files[1])

        if method == "orb":
            match_count, match_img = orb_match_fingerprints(img1_path, img2_path)
        else:
            match_count, match_img = sift_match_fingerprints(img1_path, img2_path)

        # True label: assume Data1 = same, Data2 = different, Data3 = UiA (same content)
        if "data1" in folder_name.lower():
            actual = 1  # fingerprints match
        elif "data2" in folder_name.lower():
            actual = 0  # fingerprints do not match
        elif "data3" in folder_name.lower():
            actual = 1  # university images, matching
        else:
            actual = 0
        y_true.append(actual)

        pred = 1 if match_count > threshold else 0
        y_pred.append(pred)

        result = "matched" if pred == 1 else "unmatched"
        print(f"{folder_name}: {match_count} good matches → {result.upper()}")

        # Save match image
        if match_img is not None:
            save_path = os.path.join(results_folder, f"{folder_name}_{method}_{result}.png")
            cv2.imwrite(save_path, match_img)
            print(f"Saved: {save_path}")

            # Display match image
            # plt.figure(figsize=(12, 6))
            plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
            plt.title(f"{folder_name}: {match_count} matches - {result.upper()}")
            plt.axis('off')
            plt.tight_layout()
            plt.show()

    # Only generate confusion matrix if multiple samples
    if len(y_true) > 1:
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Different", "Same"])
        plt.figure(figsize=(8, 6))
        disp.plot(cmap="Blues", values_format="d")
        plt.title(f"{method.upper()} Confusion Matrix")

        # Save confusion matrix
        cm_save_path = os.path.join(results_folder, f"{method}_confusion_matrix.png")
        plt.savefig(cm_save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved confusion matrix: {cm_save_path}")
        plt.show()
    else:
        print(f"\nOnly {len(y_true)} sample(s) processed - skipping confusion matrix (need multiple samples)")
        print(
            f"Results: {y_true[0] if y_true else 'No data'} (actual) vs {y_pred[0] if y_pred else 'No data'} (predicted)")


def compare_pipelines(dataset_root):
    import time
    import tracemalloc

    results = {}

    for method in ["orb", "sift"]:
        tracemalloc.start()
        start = time.time()

        evaluate_dataset(dataset_root, f"saved_pictures/{method}", method)

        elapsed = time.time() - start
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        results[method] = {
            'time': elapsed,
            'memory': peak / 1024 / 1024  # Convert to MB
        }

        print(f"\n{method.upper()} completed in {elapsed:.2f} seconds.")

    # Print comparison
    print("\n" + "=" * 60)
    print("RESOURCE COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<25} {'ORB':<15} {'SIFT':<15}")
    print("-" * 60)
    print(f"{'Execution Time (s)':<25} {results['orb']['time']:<15.2f} {results['sift']['time']:<15.2f}")
    print(f"{'Peak Memory (MB)':<25} {results['orb']['memory']:<15.2f} {results['sift']['memory']:<15.2f}")
    print("=" * 60)