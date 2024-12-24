import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import os
from statistics import mode  # Import mode function from statistics module
import pandas as pd

# Path to the video file
video_path = "/Users/govinda-dashugolienart/Library/CloudStorage/GoogleDrive-govinda.lienart@three-monkeys.org/My Drive/TMWC - Govinda /TMWC - Govinda /Data Science/Courses/Erasmus/AI project/Videos/video 2 calibration.mov"

# Directory to save the histogram and speed graph
output_dir = "/Users/govinda-dashugolienart/Library/CloudStorage/GoogleDrive-govinda.lienart@three-monkeys.org/My Drive/TMWC - Govinda /TMWC - Govinda /Data Science/Environments/Pycharm/PycharmProjects/Erasmus/fish tank project/outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Open the video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

# Calibration step
ret, frame = cap.read()
if not ret:
    print("Error: Cannot read video frame for calibration.")
    cap.release()
    exit()

print("Calibration Step:")
print("Select a reference object of known size (e.g., a ruler) for calibration.")
roi = cv2.selectROI("Calibration - Select Reference", frame, False, False)
cv2.destroyAllWindows()

reference_width_pixels = roi[2]  # Width of the selected ROI
try:
    reference_width_cm = float(input("Enter the real-world width of the selected object in cm: "))
    CM_TO_PIXELS_RATIO = reference_width_pixels / reference_width_cm
except ValueError:
    print("Invalid input. Calibration failed.")
    cap.release()
    exit()

# Ask the user for grid size in centimeters
try:
    grid_cell_size_cm = float(input("Enter the size of each grid cell in centimeters: "))
    grid_cell_size_px = int(grid_cell_size_cm * CM_TO_PIXELS_RATIO)
except ValueError:
    print("Invalid input. Please enter a numeric value for the grid size.")
    cap.release()
    exit()

# Constants
MIN_OBJECT_SIZE_PX = 200  # Minimum object size in pixels

# Get video frame dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Total vertical depth in cm
total_depth_cm = frame_height / CM_TO_PIXELS_RATIO

# Calculate total width of the fish tank in cm
total_width_cm = frame_width / CM_TO_PIXELS_RATIO

# Display the total depth and width of the fish tank
print(f"Total depth of the fish tank: {frame_height} pixels")
print(f"Total depth of the fish tank: {total_depth_cm:.2f} cm")
print(f"Total width of the fish tank: {frame_width} pixels")
print(f"Total width of the fish tank: {total_width_cm:.2f} cm")

# Grid configuration
GRID_COLS = frame_width // grid_cell_size_px
GRID_ROWS = frame_height // grid_cell_size_px

# Initialize background subtractor
fg_bg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

# Tracking variables
total_crossings = 0
total_distance_pixels = 0
previous_position = None
previous_grid = None
smoothed_rect = None  # Initialize smoothed rectangle

# List to store vertical positions of the fish's center
vertical_positions = []

# Speed smoothing variables
speed_smoothing_window = 5  # Number of frames for smoothing
speeds = []
smoothed_speeds = []
instantaneous_speeds = []  # Store the instantaneous speeds

print("Press 'q' or 'Esc' to exit.")


# Function to draw grid lines and display grid size in cm
def draw_grid(frame):
    for x in range(0, GRID_COLS + 1):
        cv2.line(frame, (x * grid_cell_size_px, 0), (x * grid_cell_size_px, frame_height), (200, 200, 200), 1)
        if x < GRID_COLS:
            cv2.putText(frame, f"{grid_cell_size_cm} cm", (x * grid_cell_size_px + 5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1)

    for y in range(0, GRID_ROWS + 1):
        cv2.line(frame, (0, y * grid_cell_size_px), (frame_width, y * grid_cell_size_px), (200, 200, 200), 1)
        if y < GRID_ROWS:
            cv2.putText(frame, f"{grid_cell_size_cm} cm", (5, y * grid_cell_size_px + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1)


# Smooth the speeds using a simple moving average
def smooth_speeds(speeds, window_size):
    if len(speeds) < window_size:
        return speeds
    smoothed = []
    for i in range(len(speeds)):
        if i < window_size - 1:
            smoothed.append(speeds[i])
        else:
            smoothed.append(np.mean(speeds[i - window_size + 1:i + 1]))
    return smoothed


# Process video frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video")
        break

    draw_grid(frame)

    mask = fg_bg.apply(frame)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # Smaller kernel
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)  # Fewer iterations
    _, mask = cv2.threshold(mask, 244, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > MIN_OBJECT_SIZE_PX]

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)

        if smoothed_rect is None:
            smoothed_rect = rect
        else:
            smoothed_rect = (
                (
                    0.5 * smoothed_rect[0][0] + 0.5 * rect[0][0],
                    0.5 * smoothed_rect[0][1] + 0.5 * rect[0][1]
                ),
                (
                    0.5 * smoothed_rect[1][0] + 0.5 * rect[1][0],
                    0.5 * smoothed_rect[1][1] + 0.5 * rect[1][1]
                ),
                0.5 * smoothed_rect[2] + 0.5 * rect[2]
            )

        box = cv2.boxPoints(smoothed_rect)
        box = np.int0(box)
        cv2.drawContours(frame, [box], 0, (255, 0, 0), 2)

        object_center = (int(smoothed_rect[0][0]), int(smoothed_rect[0][1]))
        cv2.circle(frame, object_center, 5, (0, 255, 0), -1)

        # Save the vertical position (y-coordinate)
        vertical_positions.append(object_center[1])

        if previous_position is not None:
            distance = math.sqrt((object_center[0] - previous_position[0]) ** 2 +
                                 (object_center[1] - previous_position[1]) ** 2)
            total_distance_pixels += distance
            speed_cm_per_sec = (distance / CM_TO_PIXELS_RATIO) / (1 / cap.get(cv2.CAP_PROP_FPS))  # Speed in cm/sec
            speeds.append(speed_cm_per_sec)
            instantaneous_speeds.append(speed_cm_per_sec)  # Store instantaneous speed

        previous_position = object_center

        grid_x = object_center[0] // grid_cell_size_px
        grid_y = object_center[1] // grid_cell_size_px
        current_grid = (grid_y, grid_x)

        if previous_grid != current_grid:
            total_crossings += 1
            previous_grid = current_grid
    else:
        previous_position = None

    # Smooth speeds and update the smoothed list
    smoothed_speeds = smooth_speeds(speeds, speed_smoothing_window)

    # Display metrics on the frame
    cv2.putText(frame, f"Crossings: {total_crossings}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Pixels Moved: {int(total_distance_pixels)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255), 2)
    cv2.putText(frame, f"Speed: {smoothed_speeds[-1]:.2f} cm/sec" if smoothed_speeds else "Speed: 0.00 cm/sec",
                (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Depth: {vertical_positions[-1] / CM_TO_PIXELS_RATIO:.2f} cm" if vertical_positions else "Depth: 0.00 cm",
                (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Object Tracking", frame)

    key = cv2.waitKey(20) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Convert total distance to centimeters using calibration
total_distance_cm = total_distance_pixels / CM_TO_PIXELS_RATIO
average_speed = np.mean(smoothed_speeds) if smoothed_speeds else 0
max_speed = np.max(smoothed_speeds) if smoothed_speeds else 0

# Average and most frequent depth
if vertical_positions:
    vertical_positions_cm = [y / CM_TO_PIXELS_RATIO for y in vertical_positions]
    average_depth = np.mean(vertical_positions_cm)
    most_frequent_depth = mode(vertical_positions_cm)

    print(f"\nAverage depth of the fish: {average_depth:.2f} cm")
    print(f"Most frequent depth of the fish: {most_frequent_depth:.2f} cm")
else:
    print("No vertical positions recorded.")
    vertical_positions_cm = []

# Print speed metrics
print(f"Total distance traveled: {total_distance_cm:.2f} cm")
print(f"Average speed: {average_speed:.2f} cm/sec")
print(f"Maximum speed: {max_speed:.2f} cm/sec")

# Save instantaneous speed graph
if instantaneous_speeds:
    plt.figure(figsize=(10, 6))
    plt.plot(instantaneous_speeds, color='blue', label="Instantaneous Speed")
    plt.title("Instantaneous Speed of Fish")
    plt.xlabel("Frame Number")
    plt.ylabel("Speed (cm/sec)")
    plt.grid(True)
    plt.legend()

    # Save the graph as a PNG file
    speed_graph_path = os.path.join(output_dir, "instantaneous_speed_graph.png")
    plt.savefig(speed_graph_path)
    plt.close()
    print(f"Instantaneous speed graph saved to {speed_graph_path}")
else:
    print("No speed data recorded.")

# Create a table with metrics
data = {
    "Metric": [
        "Total depth of the fish tank",
        "Total length of the fish tank",
        "Grid cell size",
        "Average depth of the fish",
        "Most frequent depth of the fish",
        "Total distance traveled",
        "Average speed",
        "Maximum speed"
    ],
    "Pixels": [
        frame_height,
        frame_width,
        grid_cell_size_px,
        np.mean(vertical_positions) if vertical_positions else 0,
        mode(vertical_positions) if vertical_positions else 0,
        total_distance_pixels,
        np.mean(smoothed_speeds) if smoothed_speeds else 0,
        np.max(smoothed_speeds) if smoothed_speeds else 0
    ],
    "Centimeters": [
        total_depth_cm,
        total_width_cm,  # Now using the calculated total width in cm
        grid_cell_size_cm,
        np.mean(vertical_positions) / CM_TO_PIXELS_RATIO if vertical_positions else 0,
        mode(vertical_positions) / CM_TO_PIXELS_RATIO if vertical_positions else 0,
        total_distance_cm,
        np.mean(smoothed_speeds) / CM_TO_PIXELS_RATIO if smoothed_speeds else 0,
        np.max(smoothed_speeds) / CM_TO_PIXELS_RATIO if smoothed_speeds else 0
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Save the table as a CSV file
table_path = "/Users/govinda-dashugolienart/Library/CloudStorage/GoogleDrive-govinda.lienart@three-monkeys.org/My Drive/TMWC - Govinda /TMWC - Govinda /Data Science/Environments/Pycharm/PycharmProjects/Erasmus/fish tank project/outputs/fish_metrics_table.csv"
df.to_csv(table_path, index=False)

import ace_tools as tools; tools.display_dataframe_to_user(name="Fish Metrics Table", dataframe=df)

table_path