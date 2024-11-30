import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm


# Constants
SUITS = ['R', 'B', 'G', 'Y']
NUMBERS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'd', 'r', 's']


def get_card():
    """
    Randomly selects a card from the dataset based on suits and numbers.

    Returns:
        tuple: RGB image of the card, suit, and number.
    """
    suit = np.random.choice(SUITS)
    num = np.random.choice(NUMBERS)
    img_num = np.random.choice([1, 2, 3, 4, 5])
    img_path = f'dataset/{suit}{num}-{img_num}.jpg'
    # print(img_path)

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    h, w = img.shape[:2]
    if h < w:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img, suit, num


def fill_card(img):
    """
    Fills gaps in a binary image by applying morphological operations.

    Args:
        img (numpy.ndarray): Input binary image.

    Returns:
        numpy.ndarray: Image with filled gaps.
    """
    ksize = np.random.choice([3, 5, 7])
    kernel = np.ones((ksize, ksize), np.uint8)
    iterations = 15
    img = cv2.dilate(img, kernel, iterations=iterations)
    img = cv2.erode(img, kernel, iterations=iterations)

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_image = img.copy()
    for contour in contours:
        cv2.fillPoly(filled_image, [contour], (255, 255, 255))
    return filled_image


def get_cropped_images(img, points):
    """
    Crops the image using perspective transformation based on provided points.

    Args:
        img (numpy.ndarray): Input image.
        points (list): Points defining the corners of the region of interest.

    Returns:
        tuple: Two transformed images with different orientations.
    """
    width, height = 300, 400
    if points is None:
        return np.zeros((height, width)), np.zeros((height, width))

    corners = np.array(points, dtype=np.float32)
    output_corners1 = np.array([[0, 0], [0, height], [width, 0], [width, height]], dtype=np.float32)
    output_corners2 = np.array([[0, 0], [0, height], [width, height], [width, 0]], dtype=np.float32)

    matrix1 = cv2.getPerspectiveTransform(corners, output_corners1)
    matrix2 = cv2.getPerspectiveTransform(corners, output_corners2)

    output_image1 = cv2.warpPerspective(img, matrix1, (width, height))
    output_image2 = cv2.warpPerspective(img, matrix2, (width, height))
    return output_image1, output_image2


def get_zoomed_image(img):
    """
    Crops the central region of an image.

    Args:
        img (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Cropped image.
    """
    return img[50:350, 25:275]


def skeletonize_number(img):
    """
    Creates a binary skeletonized image of the largest contour in the input image.

    Args:
        img (numpy.ndarray): Input grayscale image.

    Returns:
        numpy.ndarray: Binary skeletonized image.
    """
    edges = cv2.Canny(img.astype(np.uint8), 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(img.shape[:2])

    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    mask = np.zeros_like(img)
    cv2.drawContours(mask, [approx], -1, (255, 255, 255), -1)
    return mask


def get_skeletonized_image(output_image1, output_image2):
    """
    Skeletonizes one of the two provided images based on content.

    Args:
        output_image1, output_image2 (numpy.ndarray): Input images.

    Returns:
        numpy.ndarray: Skeletonized image or an empty array if unsuccessful.
    """
    for img in [output_image1, output_image2]:
        zoomed_img = get_zoomed_image(img)
        skeletonized = skeletonize_number(zoomed_img)
        percentage_white = np.sum(skeletonized == 255) / np.prod(skeletonized.shape) * 100
        if percentage_white > 2:
            return skeletonized.astype(np.uint8)
    return np.zeros(skeletonized.shape[:2])


def get_templates():
    """
    Loads and processes template images for template matching.

    Returns:
        dict: Dictionary of processed templates.
    """
    templates = {}
    for template_file in os.listdir('template'):
        if not template_file.endswith('.png'):
            continue
        key = template_file.split('_')[0]
        template = cv2.imread(f'template/{template_file}', cv2.IMREAD_GRAYSCALE)
        template = cv2.resize(template, (250, 300))
        _, binary_template = cv2.threshold(template, 127, 255, cv2.THRESH_BINARY)
        templates[key] = binary_template
    return templates


def match_template(img):
    """
    Matches the input image against a set of templates.

    Args:
        img (numpy.ndarray): Input binary image.

    Returns:
        dict: Match scores for each template.
    """
    templates = get_templates()
    results = {key[0]: [] for key in templates.keys()}

    if np.all(img == 0):
        return results

    for flip_code in [0, 1, -1, None]:  # No flip, horizontal flip, vertical flip
        flipped_img = cv2.flip(img, flip_code) if flip_code is not None else img
        for key, template in templates.items():
            res = cv2.matchTemplate(flipped_img, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            results[key[0]].append(max_val)

    return results


def predict_digit(image):
    """
    Predicts the digit from the input grayscale card image.

    Args:
        image (numpy.ndarray): Grayscale image of a card.

    Returns:
        list: Top 3 predicted digits, ranked by match score.
    """
    blurred = cv2.GaussianBlur(image, (5, 5), 3)
    edges = cv2.Canny(blurred, 50, 150)

    # Fill gaps in the edges
    filled_card = edges.copy()
    for _ in range(5):
        filled_card = fill_card(filled_card)

    # Extract contours and find a quadrilateral contour
    contours, _ = cv2.findContours(filled_card, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    points = None
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            points = approx.reshape((4, 2))
            break

    # If no quadrilateral contour is found, return an empty prediction
    if points is None:
        return []

    output_image1, output_image2 = get_cropped_images(image, points)
    skeletonized_image = get_skeletonized_image(output_image1, output_image2)

    if np.all(skeletonized_image == 0):
        return []

    # Perform template matching
    results = match_template(skeletonized_image)

    # Sort and extract top predictions
    scores = {key: max(vals) if vals else 0 for key, vals in results.items()}
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [digit[0] for digit in sorted_scores if digit[1] > 0][:3]


def visualize_digits_accuracy(data):
    """
    Visualizes the accuracy of digit predictions using a stacked bar chart.

    Args:
        data (dict): Accuracy data in the form {card: accuracy}.
    """
    results = {suit: [] for suit in SUITS}
    for suit in SUITS:
        for num in NUMBERS:
            results[suit].append(data.get(f'{suit}{num}', 0))

    x = NUMBERS
    y_values = np.array([results[suit] for suit in SUITS]) / 4

    plt.figure(figsize=(10, 6))
    cumulative = np.zeros(len(x))
    color_map = {'R': 'red', 'B': 'blue', 'G': 'green', 'Y': '#FFD700'}  # Gold for Yellow

    for i, suit in enumerate(SUITS):
        plt.bar(x, y_values[i], bottom=cumulative, label=f"{suit} suit", color=color_map[suit])
        cumulative += y_values[i]

    overall_accuracy = sum(data.values()) / len(data)
    plt.xticks(x)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.xlabel("Card Numbers")
    plt.title("Prediction Accuracy")
    plt.axhline(overall_accuracy, color='black', linestyle='--', label=f'Overall Accuracy ({overall_accuracy:.2f})')
    plt.tight_layout()
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.savefig("accuracy.png")
    plt.show()


def find_color(image):
    """
    Detects the dominant color in the input image.

    Args:
        image (numpy.ndarray): RGB image of a card.

    Returns:
        str: Detected color ('R', 'G', 'B', 'Y').
    """
    color_ranges = {
        'R': [(100, 0, 0), (255, 50, 50)],
        'G': [(0, 100, 0), (50, 255, 50)],
        'B': [(0, 0, 100), (50, 50, 255)],
        'Y': [(100, 100, 0), (255, 255, 50)]
    }

    color_percentages = {}
    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(image, np.array(lower), np.array(upper))
        color_percentages[color] = np.sum(mask == 255) / np.prod(mask.shape) * 100

    return max(color_percentages, key=color_percentages.get)


def evaluate(iterations):
    """
    Evaluates the accuracy of digit and color predictions over multiple iterations.

    Args:
        iterations (int): Number of iterations to run the evaluation.

    Returns:
        dict: Accuracy data for each card in the form {card: accuracy}.
    """
    correct = {f'{suit}{num}': 0 for suit in SUITS for num in NUMBERS}
    total = {f'{suit}{num}': 0 for suit in SUITS for num in NUMBERS}
    color_detection = {suit: 0 for suit in SUITS}
    until_now = 0

    with tqdm(total=iterations, desc="Evaluating") as pbar:
        while until_now < iterations:
            card_image, actual_suit, actual_num = get_card()
            gray_image = cv2.cvtColor(card_image, cv2.COLOR_RGB2GRAY)
            predicted_color = find_color(card_image)
            predicted_digits = predict_digit(gray_image)

            # Color prediction evaluation
            if predicted_color == actual_suit:
                color_detection[actual_suit] += 1

            # Digit prediction evaluation
            if actual_num in predicted_digits:
                correct[f'{actual_suit}{actual_num}'] += 1
            if predicted_digits:
                total[f'{actual_suit}{actual_num}'] += 1
                until_now += 1
                pbar.update(1)
            # else:
            #     print("No prediction found for:", actual_suit, actual_num)

    # Calculate accuracy
    accuracy = {
        card: correct[card] / total[card] if total[card] > 0 else 0
        for card in correct
    }

    # Normalize color detection scores
    color_totals = {suit: sum(total[f'{suit}{num}'] for num in NUMBERS) for suit in SUITS}
    for suit in SUITS:
        if color_totals[suit] > 0:
            color_detection[suit] = min(color_detection[suit], color_totals[suit]) / color_totals[suit]

    print("Color Detection Accuracy:", color_detection)
    print("Total Correct Predictions:", sum(correct.values()))
    print("Total Predictions:", sum(total.values()))

    return accuracy


acc = evaluate(iterations = 2000)
print(acc)
visualize_digits_accuracy(acc)
