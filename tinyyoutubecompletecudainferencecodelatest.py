import cv2
import numpy as np
import os
import time
import datetime
import csv
from fpdf import FPDF
import webbrowser

# Load YOLOv3-tiny
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Set GPU as backend and target for OpenCV DNN module
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Set your desired shop name
shop_name = "Your Shop Name"

# Set the time interval for generating PDF report (in seconds)
report_interval = 600  # 10 minutes

# Initialize variables
start_time = time.time()
crop_counter = {}
pdf = FPDF()

def detect_objects(image):
    height, width, channels = image.shape

    # Perform object detection
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialize lists for detected objects and their bounding boxes
    class_ids = []
    confidences = []
    boxes = []

    # Iterate over each detection
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter out weak detections
            if confidence > 0.5:
                # Scale the bounding box coordinates
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calculate the top-left corner coordinates of the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Add the object information to the lists
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Apply non-maximum suppression to eliminate overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Initialize lists for final detected objects and their classes
    detected_objects = []
    detected_classes = []

    # Check if there are any detections
    if len(indices) > 0:
        for i in indices.flatten():
            # Get the class label and confidence of the current detection
            class_id = class_ids[i]
            class_name = classes[class_id]
            confidence = confidences[i]

            # Get the crop image and save it
            crop_image = image[boxes[i][1]:boxes[i][1] + boxes[i][3], boxes[i][0]:boxes[i][0] + boxes[i][2]]
            save_image(class_name, crop_image)

            # Add the detected object to the list
            detected_objects.append(crop_image)
            detected_classes.append(class_name)

    return detected_objects, detected_classes


def save_image(class_name, image):
    # Check if the class has already reached the image limit
    if class_name not in crop_counter:
        crop_counter[class_name] = 0
    elif crop_counter[class_name] >= 10:
        return

    # Create a folder for the class if it doesn't exist
    class_folder = os.path.join("cropped_images", class_name)
    if not os.path.exists(class_folder):
        os.makedirs(class_folder)

    # Generate unique image name with class, ID, date, time, and shop
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    image_name = f"{class_name}_{crop_counter[class_name]}_{timestamp}_{shop_name}.jpg"
    image_path = os.path.join(class_folder, image_name)

    # Save the cropped image
    cv2.imwrite(image_path, image)

    # Increment the counter for the class
    crop_counter[class_name] += 1


def search_product_online(query):
    # Add your code to search for the product online and retrieve the links and prices
    # You can use libraries like BeautifulSoup or Selenium for web scraping


def compare_prices(detected_classes):
    # Initialize the CSV file
    with open("price_comparison.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Class", "Product Name", "Link", "Price"])

        # Compare prices for each detected class
        for class_name in detected_classes:
            if class_name not in crop_counter:
                continue

            class_folder = os.path.join("cropped_images", class_name)
            images = os.listdir(class_folder)
            if len(images) < 10:
                continue

            # Select one image for price comparison
            image_path = os.path.join(class_folder, images[0])
            product_name = class_name  # Update with actual product name

            # Search for the product online
            links, prices = search_product_online(product_name)

            # Write the results to the CSV file
            for i in range(min(10, len(links))):
                writer.writerow([class_name, product_name, links[i], prices[i]])


def generate_pdf_report():
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Product Detection Report", ln=True, align="C")
    pdf.ln(10)

    # Add the detected objects and their links to the PDF
    class_folders = os.listdir("cropped_images")
    for class_folder in class_folders:
        pdf.set_font("Arial", size=10)
        pdf.cell(200, 10, txt=class_folder, ln=True)
        pdf.ln(5)

        images = os.listdir(os.path.join("cropped_images", class_folder))
        links, prices = search_product_online(class_folder)

        for i in range(min(len(images), len(links))):
            image_path = os.path.join("cropped_images", class_folder, images[i])
            pdf.image(image_path, x=10, y=pdf.get_y() + 5, w=40)
            pdf.set_xy(60, pdf.get_y() + 5)
            pdf.cell(200, 10, txt=links[i], ln=True)
            pdf.ln(5)

    # Save the PDF report
    report_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf.output(f"product_detection_report_{report_name}.pdf")


# Main code
cap = cv2.VideoCapture("youtube_video_link")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Perform object detection on the frame
    detected_objects, detected_classes = detect_objects(frame)

    # Compare prices for the detected classes
    compare_prices(detected_classes)

    # Check if it's time to generate a PDF report
    current_time = time.time()
    elapsed_time = current_time - start_time

    if elapsed_time >= report_interval:
        generate_pdf_report()
        start_time = current_time

    # Display the frame with detected objects
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()



