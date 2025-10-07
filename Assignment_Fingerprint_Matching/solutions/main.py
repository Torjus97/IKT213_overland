import cv2
from matchers import orb_match_fingerprints, sift_match_fingerprints
from evaluate import evaluate_dataset, compare_pipelines


def select_dataset():
    """Simple dataset selector."""
    print("\nSelect dataset:")
    print("1 - Data1")
    print("2 - Data2")
    print("3 - Data3")
    choice = input("Enter choice (1-3): ")

    if choice == "1":
        return "datasets/Data1"
    elif choice == "2":
        return "datasets/Data2"
    elif choice == "3":
        return "datasets/Data3"
    else:
        print("Invalid choice, using all datasets")
        return "datasets"


def orb_sift_matching():
    dataset_path = select_dataset()
    results_folder = "saved_pictures/orb"
    evaluate_dataset(dataset_path, results_folder, method="orb")


def sift_flann_matching():
    dataset_path = select_dataset()
    results_folder = "saved_pictures/sift"
    evaluate_dataset(dataset_path, results_folder, method="sift")


def pipeline_comparison():
    dataset_path = "datasets"
    compare_pipelines(dataset_path)


def main():
    operations = {
        "1": orb_sift_matching,
        "2": sift_flann_matching,
        "3": pipeline_comparison
    }

    while True:
        print("\nChoose an operation:")
        print("1 - ORB + BFMatcher (fingerprints)")
        print("2 - SIFT + FLANN (fingerprints)")
        print("3 - Compare both pipelines (speed + accuracy)")
        print("0 - Exit")

        choice = input("Enter your choice: ")

        if choice == "0":
            break
        elif choice in operations:
            operations[choice]()
        else:
            print("Invalid choice, try again.")


if __name__ == "__main__":
    main()