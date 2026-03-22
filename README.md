# YOLO Detection Platform

This is the code for a Python server, hosted in the Raspberry that detects whether scrolling is occuring or not using footage from a camera. It uses three YOLO models: one pretrained model (yolo12n.pt) to detect phones, a general model to detect faces and hands (general_model.pt), and a custom model to detect hands (hand_model.pt). The general and hand models have copies that have been converted to .onxx format. The server features a web UI for testing purposes. 
