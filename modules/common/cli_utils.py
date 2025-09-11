from typing import Tuple

def prompt_for_mdpi_configuration(capture_rate: float, image_height_cm: float) -> Tuple[float, float]:
    """
    Prompts the user to accept or modify MDPI configuration values.

    Args:
        capture_rate: The default capture rate.
        image_height_cm: The default image height in cm.

    Returns:
        A tuple containing the final capture rate and image height.
    """
    print("\n--- MDPI Configuration ---")
    print(f"Default values:")
    print(f"  1) CAPTURE_RATE: {capture_rate} Hz")
    print(f"  2) IMAGE_HEIGHT_CM: {image_height_cm} cm")

    while True:
        choice = input("Accept these values? (Y/N): ").strip().upper()
        if choice in ['Y', 'N']:
            break
        print("Invalid input. Please enter 'Y' or 'N'.")

    final_capture_rate = capture_rate
    final_image_height_cm = image_height_cm

    if choice == 'N':
        while True:
            try:
                new_capture_rate_str = input(f"Enter new CAPTURE_RATE (current: {capture_rate}): ").strip()
                if new_capture_rate_str:
                    final_capture_rate = float(new_capture_rate_str)
                break
            except ValueError:
                print("Invalid input. Please enter a number.")

        while True:
            try:
                new_image_height_str = input(f"Enter new IMAGE_HEIGHT_CM (current: {image_height_cm}): ").strip()
                if new_image_height_str:
                    final_image_height_cm = float(new_image_height_str)
                break
            except ValueError:
                print("Invalid input. Please enter a number.")
    print("--------------------------\n")
    
    return final_capture_rate, final_image_height_cm
