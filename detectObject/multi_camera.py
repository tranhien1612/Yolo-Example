from queue import Queue
from threading import Thread
import cv2
from ultralytics import YOLO

def detection_task(source, model, output_queue):
    video = cv2.VideoCapture(source)
    while True:
        ret, frame = video.read()
        if not ret:
            break
        results = model.predict(frame)
        if 'bicycle' in results.names:
            output_queue.put((source, 'Bicycle detected'))

# Setup
output_queue = Queue()
model = YOLO('yolov8n.pt')

threads = []
sources = ['stream1.mp4', 'stream2.mp4', ...]  # Your stream sources

for source in sources:
    thread = Thread(target=detection_task, args=(source, model, output_queue))
    thread.start()
    threads.append(thread)

# Collect results
while any(thread.is_alive() for thread in threads):
    while not output_queue.empty():
        source, message = output_queue.get()
        print(f'Result from {source}: {message}')

for thread in threads:
    thread.join()
