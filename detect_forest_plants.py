import os
from inference_engine import predict_plant

if __name__ == "__main__":

    image_path = input("Enter image path: ")

    predictions, result_image_path, err = predict_plant(image_path)

    if err:
        print(f"Error: {err}")
    elif not predictions:
        print("\nNo medicinal plant could be confidently identified.")
    else:
        for i, result in enumerate(predictions, 1):
            print(f"\n--- Detection {i} ---")
            print(f"Plant:       {result['plant']}")
            print(f"Confidence:  {result['confidence']}%")
            print(f"Box:         {result['box']}")
            print(f"Uses:        {result['uses']}")
            print(f"Precautions: {result['precautions']}")

        if result_image_path:
            print(f"\nAnnotated image saved to: {result_image_path}")