import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import os

suits = [
    'R',
      'B',
        'G', 
      'Y',
         ]
numbers = [
    '0', '1', 
    '2',
      '3', '4', 
           '5', 
           '6', 
           '7', 
           '8',
             '9',
           #    'd', 'r', 's'
           ]


def get_card():
    suit = np.random.choice(suits)
    if suit == 'W':
        num = np.random.choice(['1', '2'])
    else:
        num = np.random.choice(numbers)
    # num = '6'
    # suit = 'R'
    # img_num = 4
    img_num = np.random.choice([1, 2, 3, 4, 5])

    if suit == 'Y':
        img_num = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])

    # img_num = 3
    img_path = f'dataset/{suit}{num}-{img_num}.jpg'
    # print(f'Getting card: {img_path}')
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    if h < w:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), suit, num


def fill_card(img):
    ksize = np.random.choice([3, 5, 7])
    kernel = np.ones((ksize, ksize), np.uint8)
    iter_ = 15
    img = cv2.dilate(img, kernel, iterations=iter_)
    img = cv2.erode(img, kernel, iterations=iter_)

    contours, _ = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filled_image = img.copy()
    i = 0
    for contour in contours:
        cv2.fillPoly(filled_image, [contour], (255, 255, 255))
        i += 1
    return filled_image

# Crop the imsge to the four corners


def get_cropped_images(img, points):
    # Define the dimensions of the output image
    width = 300
    height = 400

    if points is None:
        return np.zeros((height, width)), np.zeros((height, width))

    corners = np.array([points[0], points[1], points[2],
                       points[3]], dtype=np.float32)

    # Define the four corners of the output image
    output_corners1 = np.array([[0, 0], [0, height], [width, 0], [
                               width, height]], dtype=np.float32)
    output_corners2 = np.array(
        [[0, 0], [0, height], [width, height], [width, 0]], dtype=np.float32)

    # Compute the perspective transform matrix
    matrix1 = cv2.getPerspectiveTransform(corners, output_corners1)
    matrix2 = cv2.getPerspectiveTransform(corners, output_corners2)

    # Apply the perspective transform
    output_image1 = cv2.warpPerspective(img, matrix1, (width, height))
    output_image2 = cv2.warpPerspective(img, matrix2, (width, height))

    return output_image1, output_image2


def get_zoomed_image(img):
    return img[50:350, 25:275]


def skeletonize_number(img):
    # Apply edge detection
    edges = cv2.Canny(img.astype(np.uint8), 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assume the largest contour is the card
    if not contours:
        return np.zeros(img.shape[:2])
    largest_contour = max(contours, key=cv2.contourArea)
    # second_largest_contour = contour_sorted[1]
    # largest_contour = max(contours, key=cv2.contourArea)

    # Approximate the contour to a quadrilateral
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    mask = np.zeros_like(img)
    cv2.drawContours(mask, [approx], -1, (255, 255, 255), -1)

    return mask


def get_skeletonized_image(output_image1, output_image2):
    for img in [output_image1, output_image2]:
        img = get_zoomed_image(img)
        skeletonized = skeletonize_number(img)
        # Print the percentage of white pixels in the image
        percentage_white = np.sum(skeletonized == 255) / np.prod(skeletonized.shape) * 100
        plt.imshow(skeletonized, cmap='gray')
        if percentage_white > 2:
            img = skeletonized.astype(np.uint8)
            return img
    return np.zeros(skeletonized.shape[:2])


def get_templates():
    templates = {}
    for template in os.listdir('template'):
        if not template.endswith('.png'):
            continue
        i = template.split('_')[0]

        template = cv2.imread(f'template/{template}', cv2.IMREAD_GRAYSCALE)
        template = cv2.resize(template, (250, 300))

        # Binary thresholding
        _, template = cv2.threshold(template, 127, 255, cv2.THRESH_BINARY)
        templates[i] = template
    return templates


def match_template(img):
    templates = get_templates()

    results = {}
    for i, template in templates.items():
        i = i[0]
        results[i] = []

    if np.all(img == 0):
        return results

    for i in range(4):
        img_copy = img.copy()
        hflip = i % 2
        vflip = i//2
        if hflip:
            img_copy = cv2.flip(img_copy, 1)
        if vflip:
            img_copy = cv2.flip(img_copy, 0)

        for i, template in templates.items():
            i = i[0]
            res = cv2.matchTemplate(img_copy, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            results[i].append(max_val)
            if i == '6':
                results['9'].append(max_val)
            if i == '9':
                results['6'].append(max_val)
        return results


def predict_digit(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 3)
    edges = cv2.Canny(blurred, 50, 150)

    filled_card = edges.copy()
    for i in range(5):
        filled_card = fill_card(filled_card)

    contours, _ = cv2.findContours(
        filled_card, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    points = None
    # Approximate the contour to a polygon
    for contour in contours:
        new_img = image.copy()
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:  # If the contour has 4 vertices, it might be the card
            # print("Corner points:")
            for ind, point in enumerate(approx):
                cv2.circle(new_img, tuple(point[0]), 10, (255, 0, 0), -1)
                cv2.putText(new_img, f"{ind}", tuple(
                    point[0]), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 2)
                # print(f"Point {ind}: {point[0]}")
            points = approx.reshape((4, 2))
            points = np.array(sorted(points, key=lambda x: x[0]))
            # print("Edge equations:")
            for i in range(4):
                # Get endpoints of each side
                x1, y1 = points[i]
                x2, y2 = points[(i+1) % 4]

                # Equation of the line: Ax + By + C = 0
                A = y2 - y1
                B = x1 - x2
                C = x2 * y1 - x1 * y2

                cv2.line(new_img, tuple(points[i]), tuple(
                    points[(i+1) % 4]), (0, 255, 0), 2)

                # print(f"Line {i+1}: {A}x + {B}y + {C} = 0")
            plt.imshow(new_img)
            break

    output_image1, output_image2 = get_cropped_images(image, points)

    skeletonized_image = get_skeletonized_image(output_image1, output_image2)
    # if np.any(skeletonized_image != 0):
    #     plt.clf()
    #     plt.imsave(f"7_{time.time()}.png", skeletonized_image, cmap='gray')
    #     print("Saved")
    plt.imshow(skeletonized_image, cmap='gray')
    # plt.show()

    results = match_template(skeletonized_image)

    # Print the number with the highest match score
    number = {}
    for i, scores in results.items():
        number[i] = max(scores) if scores else 0

    # Sort the numbers by match score
    number = sorted(number.items(), key=lambda x: x[1], reverse=True)
    if number[0][1] == 0:
        return []
    return [digit[0] for digit in number[:3]]


def evaluate(iter):
    correct = {}
    total = {}
    for suit in suits:
        for num in numbers:
            correct[f'{suit}{num}'] = 0
            total[f'{suit}{num}'] = 0
    for i in range(iter):
        image, suit, num = get_card()
        preds = predict_digit(image)
        print(preds)
        if num in preds:
            correct[f'{suit}{num}'] += 1
        if preds != []:
            total[f'{suit}{num}'] += 1
        
    print(np.sum(list(correct.values())))
    print(np.sum(list(total.values())))

    acc = {}
    for suit in suits:
        for num in numbers:
            acc[f'{suit}{num}'] = correct[f'{suit}{num}'] / total[f'{suit}{num}'] if total[f'{suit}{num}'] != 0 else 0
    print(acc)

    plt.clf()
    results = {"R": [], "B": [], "G": [], "Y": []}
    for suit in suits:
        for num in numbers:
            results[suit].append(acc[f'{suit}{num}'])
    # Plot a stacked bar chart
    # x = np.arange(len(numbers))
    # width = 0.35
    # fig, ax = plt.subplots()
    # for i, suit in enumerate(suits):
    #     ax.bar(x + i * width, results[suit], width, label=suit)
    # ax.set_ylabel('Accuracy')
    # ax.set_title('Accuracy by number and color')
    # ax.set_xticks(x + width)
    # ax.set_xticklabels(numbers)
    # ax.legend()
    # plt.show()


    print(results)

    # Assuming that the keys "R", "B", "G", "Y" are suits and the inner lists represent numerical values
    categories = list(results.keys())
    values = np.array([results[cat] for cat in categories])

    # Create an array for the x-axis (based on the length of one of the lists)
    x = np.arange(len(values[0]))

    # Plotting stacked bars
    fig, ax = plt.subplots(figsize=(10, 6))

    # Cumulative heights for stacking
    cumulative = np.zeros(len(x))

    colormap = {
        'R': 'red',
        'B': 'blue',
        'G': 'green',
        'Y': '#FFD700'  # Gold
    }
    # Loop through each category and add to the plot
    for i, cat in enumerate(categories):
        ax.bar(x, values[i], bottom=cumulative, label=cat, color=colormap[cat])
        cumulative += values[i]  # Update cumulative heights

    # Customization
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}' for i in range(len(x))])  # Replace with actual labels if needed
    ax.set_ylabel("Values")
    ax.set_xlabel("Numbers")
    ax.set_title("Stacked Bar Plot")
    ax.legend(title="Suits")

    plt.show()


evaluate(iter = 200)
