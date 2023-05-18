import cv2
import numpy as np
import torch
from torchvision import transforms
import time
import os
from PIL import Image
from io import BytesIO
import requests
from bs4 import BeautifulSoup
import csv
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from pdf2image import convert_from_path

# Configuration Paths
config_path = "yolov3.cfg"
weights_path = "yolov3.weights"
tiny_config_path = "yolov3-tiny.cfg"
tiny_weights_path = "yolov3-tiny.weights"
class_names_path = "coco.names"

# Constants
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
IMG_SIZE = 416
CROP_IMAGES_PER_CLASS = 100
SHOP_NAME = "Your Shop Name"
YOUTUBE_URL = "Your YouTube URL"

# Load YOLO model
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Load classes
with open(class_names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def detect_objects(frame):
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (IMG_SIZE, IMG_SIZE), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    height, width, _ = frame.shape

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > CONF_THRESHOLD:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
        detected_objects = []
        for i in indices:
            i = i[0]
            box = boxes[i]
            x, y, w, h = box
            class_id = class_ids[i]
            label = classes[class_id]
            detected_objects.append((label, x, y, x + w, y + h))

        # Crop and save images
        for label, x1, y1, x2, y2 in detected_objects:
            class_output_dir = os.path.join(output_dir, label)
            os.makedirs(class_output_dir, exist_ok=True)
            class_files = os.listdir(class_output_dir)
            if len(class_files) >= CROP_IMAGES_PER_CLASS:
                continue  # Skip if already cropped required number of images for the class

            cropped_image = frame[y1:y2, x1:x2]
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            image_name = f"{label}_{timestamp}.jpg"
            image_path = os.path.join(class_output_dir, image_name)
            cv2.imwrite(image_path, cropped_image)

            print(f"Cropped image saved: {image_path}")

        # Search for detected classes without ID on e-commerce platforms
        for label, _, _, _, _ in detected_objects:
            if label not in classes:
                search_query = f"{label} {SHOP_NAME}"
                search_results = search_ecommerce_platforms(search_query)

                # Compare prices and generate CSV file
                csv_data = []
                for result in search_results:
                    title, link, price = result
                    csv_data.append([label, title, link, price])

                csv_file = "price_comparison.csv"
                write_to_csv(csv_file, csv_data)

        # Display the detected objects and links on the video stream
        display_objects(frame, detected_objects)

        # Save the video frame with the detected objects and links as a PDF
        pdf_file = "detection_report.pdf"
        save_as_pdf(pdf_file, frame, detected_objects)

        # Wait for 10 minutes before processing the next frame
        time.sleep(600)

# Function to search for a given query on e-commerce platforms
def search_ecommerce_platforms(query):
    search_results = []

    # Flipkart
    flipkart_results = search_flipkart(query)
    search_results.extend(flipkart_results)

    # Google
    google_results = search_google(query)
    search_results.extend(google_results)

    # Myntra
    myntra_results = search_myntra(query)
    search_results.extend(myntra_results)

    # Amazon
    amazon_results = search_amazon(query)
    search_results.extend(amazon_results)

    return search_results

# Function to search for a given query on Flipkart
def search_flipkart(query):
    # Perform the search and scrape the results
    # Replace this code with your own Flipkart search and scraping logic
    # Here's an example of how to use requests and BeautifulSoup for scraping
    for result in results:
            title = result.find("a", {"class": "IRpwTa"}).text.strip()
            link = result.find("a", {"class": "IRpwTa"})["href"]
            price = result.find("div", {"class": "_30jeq3 _1_WHN1"}).text.strip()
            search_results.append((title, link, price))

    return search_results

# Function to search for a given query on Google
def search_google(query):
    # Perform the search and scrape the results
    # Replace this code with your own Google search and scraping logic
    # Here's an example of how to use requests and BeautifulSoup for scraping

    url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    results = soup.find_all("div", {"class": "g"})
    search_results = []
    for result in results:
        title = result.find("h3").text.strip()
        link = result.find("a")["href"]
        search_results.append((title, link, None))

    return search_results

# Function to search for a given query on Myntra
def search_myntra(query):
    # Perform the search and scrape the results
    # Replace this code with your own Myntra search and scraping logic
    # Here's an example of how to use requests and BeautifulSoup for scraping

    url = f"https://www.myntra.com/{query.replace(' ', '-')}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    results = soup.find_all("li", {"class": "product-base"})
    search_results = []
    for result in results:
        title = result.find("a", {"class": "product-brand"}).text.strip()
        link = result.find("a", {"class": "product-brand"})["href"]
        price = result.find("div", {"class": "product-price"}).text.strip()
        search_results.append((title, link, price))

    return search_results

# Function to search for a given query on Amazon
def search_amazon(query):
    # Perform the search and scrape the results
    # Replace this code with your own Amazon search and scraping logic
    # Here's an example of how to use requests and BeautifulSoup for scraping

    url = f"https://www.amazon.com/s?k={query.replace(' ', '+')}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    results = soup.find_all("div", {"data-component-type": "s-search-result"})
    search_results = []
    for result in results:
        title = result.find("h2").text.strip()
        link = result.find("a")["href"]
        price = result.find("span", {"class": "a-offscreen"}).text.strip()
        search_results.append((title, link, price))

    return search_results

# Function to write CSV data to a file
def write_to_csv(csv_file, data):
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Class", "Title", "Link", "Price"])
        writer.writerows(data)

    print(f"CSV file saved: {csv_file}")

# Function to display the detected objects and links on the video stream
def display_objects(frame, detected_objects):
    # Display the objects and links on the frame
    for label, x1, y1, x2, y2 in detected_objects:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with objects and links
    cv2.imshow("Object Detection", frame)
    cv2.waitKey(1)

# Function to save the video frame with the detected objects and links as a PDF
def save_as_pdf(pdf_file, frame, detected_objects):
    # Create a canvas for the PDF
    pdf_canvas = canvas.Canvas(pdf_file, pagesize=letter)
    pdf_canvas.setFont("Helvetica", 12)

    # Draw the frame on the PDF
    img_temp_file = "temp.jpg"
    cv2.imwrite(img_temp_file, frame)
    pdf_canvas.drawImage(img_temp_file, 50, 500, width=500, height=375)

    # Draw the detected objects and links on the PDF
    y_position = 450
    for label, _, _, _, _ in detected_objects:
        pdf_canvas.drawString(50, y_position, f"Detected Object: {label}")
        y_position -= 20

        # Search the object on Flipkart
        flipkart_results = search_flipkart(label)
        if flipkart_results:
            title, link, price = flipkart_results[0]
            pdf_canvas.drawString(50, y_position, f"Flipkart Link: {link}")
            pdf_canvas.drawString(350, y_position, f"Price: {price}")
            y_position -= 20

        # Search the object on Google
        google_results = search_google(label)
        if google_results:
            title, link, _ = google_results[0]
            pdf_canvas.drawString(50, y_position, f"Google Link: {link}")
            y_position -= 20

        # Search the object on Myntra
        myntra_results = search_myntra(label)
        if myntra_results:
            title, link, price = myntra_results[0]
            pdf_canvas.drawString(50, y_position, f"Myntra Link: {link}")
            pdf_canvas.drawString(350, y_position, f"Price: {price}")
            y_position -= 20

        # Search the object on Amazon
        amazon_results = search_amazon(label)
        if amazon_results:
            title, link, price = amazon_results[0]
            pdf_canvas.drawString(50, y_position, f"Amazon Link: {link}")
            pdf_canvas.drawString(350, y_position, f"Price: {price}")
            y_position -= 20

        y_position -= 10

    pdf_canvas.save()
    print(f"PDF report saved: {pdf_file}")

# Capture live video stream from YouTube
# Replace YOUTUBE_URL with your desired YouTube video URL
cap = cv2.VideoCapture(YOUTUBE_URL)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Perform object detection on the frame
    detected_objects = detect_objects(frame)

    # Specify the output directory for cropped images
    output_dir = "cropped_images"

        # Crop and save images based on their classes
    crop_and_save_images(frame, detected_objects, output_dir)

    # Search for detected classes without ID on e-commerce platforms
    search_and_compare_prices(detected_objects)

    # Display the detected objects and links on the video stream
    display_objects(frame, detected_objects)

    # Save the video frame with the detected objects and links as a PDF
    save_as_pdf("detection_report.pdf", frame, detected_objects)

    # Wait for 10 minutes before processing the next frame
    time.sleep(600)

cap.release()
cv2.destroyAllWindows()

    
