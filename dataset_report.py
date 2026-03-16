import os

DATASET_DIR = "dataset"

def get_image_counts(dataset_dir):
    """Scans the dataset directory and returns a dictionary of class counts."""
    class_counts = {}
    if not os.path.exists(dataset_dir):
        print(f"Error: Directory '{dataset_dir}' not found.")
        return class_counts

    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_dir):
            # Count files that look like images (basic check)
            images = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            class_counts[class_name] = len(images)
    
    return class_counts

def analyze_dataset():
    """Analyzes the dataset and prints a report."""
    print("--- Dataset Diagnostic Report ---")
    
    counts = get_image_counts(DATASET_DIR)
    
    if not counts:
        return

    # Print table header
    print(f"\n{'Class':<15} {'Images':>10}")
    print("-" * 26)
    
    total_images = 0
    max_count = 0
    min_count = float('inf')
    
    for class_name, count in sorted(counts.items()):
        print(f"{class_name:<15} {count:>10}")
        total_images += count
        if count > max_count:
            max_count = count
        if count < min_count:
            min_count = count

    print("-" * 26)
    print(f"{'Total':<15} {total_images:>10}\n")
    
    # Check for imbalance (> 3x difference between largest and smallest classes)
    if min_count > 0 and max_count > 3 * min_count:
        print("Warning: Dataset imbalance detected. Consider balancing the classes.\n")
    elif min_count == 0:
        print("Warning: One or more classes have zero images.\n")
    else:
        print("Status: Dataset appears reasonably balanced.\n")
        
    # Print balance suggestions
    print("Recommended balance:")
    print("- 400–500 images per class for best results.")
    print("- Ensure all classes have roughly the same number of images to prevent model bias.")

if __name__ == "__main__":
    analyze_dataset()
