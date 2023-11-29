from fastapi import FastAPI, Form, UploadFile, File, Request, Response
import requests, json
import cv2
import numpy as np
import subprocess
from keras.models import load_model
from starlette.responses import JSONResponse
from fastapi.responses import FileResponse, StreamingResponse
from typing import List
import os
from pydantic import BaseModel
import shutil
from fastapi.middleware.cors import CORSMiddleware
from dropbox import Dropbox
import traceback
from pathlib import Path
from PIL import Image, ImageDraw
import torch
from ultralytics import YOLO
import datetime


app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ownerModel = YOLO('carDetection.pt')
personModel = torch.hub.load('ultralytics/yolov5', 'custom', path='human_weights.pt')
PMClasses = personModel.names

# class ImageInfo(BaseModel):
#     uri: str
#     name: str
#     type: str

# class ImageProcessRequest(BaseModel):
#     user_name: str
#     images: List[ImageInfo]

@app.get("/")
async def index():
    return {"message": "FastAPI Video Streaming"}

def preprocess_image(img, img_size):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = np.expand_dims(img, axis=0)
    return img

def predict_car(image):
    img_size = 32
    model_path = 'car_detection_model.h5'  # Path to your model
    model = load_model(model_path)
    img = preprocess_image(image, img_size)
    prediction = model.predict(img)
    class_idx = np.argmax(prediction[0])
    class_label = 'car' if class_idx == 0 else 'not a car'
    confidence = prediction[0][class_idx] * 100
    return class_label, confidence

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    nparr = np.fromstring(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    class_label, confidence = predict_car(img)
    return JSONResponse(content={'class_label': class_label, 'confidence': confidence})

def delete_folder_contents(folder_path):
    try:
        for root, dirs, files in os.walk(folder_path, topdown=False):
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                shutil.rmtree(dir_path)
        print("All subfolders and files deleted.")
    except Exception as e:
        print(f"An error occurred: {e}")

def delete_subfolders(folder_path):
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for folder in dirs:
            folder_path = os.path.join(root, folder)
            try:
                shutil.rmtree(folder_path)
                print(f"Deleted folder and its contents: {folder_path}")
            except OSError as e:
                print(f"Error deleting folder {folder_path}: {e}")

ACCESS_TOKEN='sl.BkmvZBp1SOqlMpC1qR_qAA4mu8FzWG7szEL3y-2gyA_x9LTdXlxpGRIgMBzaCC6W-UGuWB87tG3KXjYieEYtdwp-iAGLWZY3ygPl68lL_3M586BPNPzHUa4Y4N7iYJj8JwwSYrla5rbT0gAW032jNW0'
                
dbx = Dropbox(ACCESS_TOKEN)

@app.post('/run_yolov5')
async def run_yolov5(file: UploadFile = File(...)):
    try:
        source_video_directory = os.path.join(os.getcwd(), 'app','yolov5','source_videos')
        print("Source_video_directory: ", source_video_directory)
        source_video_path = os.path.join(source_video_directory, file.filename)
        print("Source Video Path:", source_video_path)
        video_content = file.file.read()
        print("Video Content Length:", len(video_content))
        with open(source_video_path, 'wb') as f:
            print(f)
            f.write(video_content)

        print("Video File Written:", source_video_path)
        folder_to_clean = os.path.join(os.getcwd(), 'app','yolov5','runs', 'detect')
        print("FOLDER TO CLEAN:", folder_to_clean)
        delete_subfolders(folder_to_clean)
        original_directory=os.getcwd()
        new_directory = os.path.join(original_directory, 'app','yolov5')
        os.chdir(new_directory)
        print("ORIGINAL:", os.getcwd())
        command = (
            "python detect.py"
            f" --source source_videos\\video.mp4"
            " --weights best.pt"
            " --img-size 640 --conf 0.4 --device cpu"
        )
        print("YOLOv5 Command:", command)
        subprocess.run(command, shell=True)
        
        output_video_directory = os.path.join(os.getcwd(), 'runs','detect', 'exp')
        output_video_path = os.path.join(output_video_directory, 'video.mp4')
        print("OUTPUT: ", output_video_path)
        
        if os.path.exists(output_video_path):
            video_filename = "video.mp4"
            with open(output_video_path, "rb") as video_file:
                video_data = video_file.read()
            # Specify the destination path on Dropbox (just the filename since you're uploading to the root)
            dropbox_video_path = f"/{video_filename}"
            try:
                dbx.files_delete_v2(dropbox_video_path)
            except Exception as delete_error:
                print("Error deleting existing file:", delete_error)
                pass

            response = dbx.files_upload(video_data, dropbox_video_path)
            shared_link = dbx.sharing_create_shared_link(dropbox_video_path)
            public_link = shared_link.url.replace("?dl=0", "?dl=1")

            return {"message": "File uploaded and link generated", "link": public_link}
        else:
            return JSONResponse(content={'error': 'Video not found'}, status_code=404)
        
    except Exception as e:
        traceback_str = traceback.format_exc()
        print("Error:", traceback_str)
        return JSONResponse(content={'error': str(e)}, status_code=500)
    finally:
        os.chdir(original_directory)
    
@app.post("/face_dataset")
async def face_recognition(name: str = Form(...), images: List[UploadFile] = []):
    try:
        if ' ' in name:
            name = name.replace(' ', '_')
        if not images:
            return JSONResponse(content={"message": "No images provided"}, status_code=400)
        training_directory = os.path.join(os.getcwd(), 'app','Face_recognition')
        original_directory = os.getcwd()
        os.chdir(training_directory)
        aligned_images_path = os.path.join(os.getcwd(), 'aligned_img')
        training_images = os.path.join(os.getcwd(), 'train_img', name)
        delete_folder_contents(aligned_images_path)
        if Path(training_images).is_dir():
            return JSONResponse(content={"message": "Your model is already trained"}, status_code=400)
        
        # Create a directory to save uploaded images
        os.makedirs(training_images, exist_ok=True)
        print("START WRITING IMAGES")
        for image in images:
            print(image.filename)
            image_path = os.path.join(training_images, image.filename)
            with open(image_path, "wb") as f:
                f.write(image.file.read())
        command = (
            "python data_preprocess.py"
        )
        print("Data Preprocessing Command:", command)
        subprocess.run(command, shell=True)
        
        print("PREPROCESSING DONE")
        
        print("TRAINING START")
        command = (
            "python train_main.py"
            f" {name}"
        )
        print("Training Command:", command)
        subprocess.run(command, shell=True)
        return {"message": f"Training completed for {name}"}
        
    except Exception as e:
        traceback_str = traceback.format_exc()
        print("Error:", traceback_str)
        return JSONResponse(content={'error': str(e)}, status_code=500)
    finally:
        os.chdir(original_directory)
        
@app.post("/face_recognition")
async def face_detection(name: str = Form(...), file: UploadFile = File(...)):
    try:
        if ' ' in name:
            name = name.replace(' ', '_')
        if not file:
            return JSONResponse(content={"message": "No Video provided"}, status_code=400)
        training_directory = os.path.join(os.getcwd(), 'app','Face_recognition')
        original_directory = os.getcwd()
        os.chdir(training_directory)
        # source_video_directory = os.path.join(os.getcwd(), 'app','yolov5','source_videos')
        # print("Source_video_directory: ", source_video_directory)
        source_video_path = os.path.join(os.getcwd(), file.filename)
        print("Source Video Path:", source_video_path)
        video_content = file.file.read()
        print("Video Content Length:", len(video_content))
        with open(source_video_path, 'wb') as f:
            print(f)
            f.write(video_content)

        command = (
            "python face_recognition.py"
            f" {source_video_path} {name}"
        )
        print("Recognition Command:", command)
        subprocess.run(command, shell=True)
        output_video_path = os.path.join(os.getcwd(), 'output.mp4')
        print("OUTPUT: ", output_video_path)
        
        if os.path.exists(output_video_path):
            video_filename = "output.mp4"
            with open(output_video_path, "rb") as video_file:
                video_data = video_file.read()
            # Specify the destination path on Dropbox (just the filename since you're uploading to the root)
            dropbox_video_path = f"/{video_filename}"
            try:
                # Check if the file exists
                dbx.files_get_metadata(dropbox_video_path)

                # If the file exists, delete it
                dbx.files_delete_v2(dropbox_video_path)
                print("Existing file deleted.")
            except dbx.exceptions.ApiError as not_found_error:
                if not_found_error.error.is_path() and not_found_error.error.get_path().is_not_found():
                    print("File not found.")
                else:
                    print("Error checking/deleting file:", not_found_error)
            except Exception as e:
                print("An error occurred:", e)

            response = dbx.files_upload(video_data, dropbox_video_path)
            shared_link = dbx.sharing_create_shared_link(dropbox_video_path)
            public_link = shared_link.url.replace("?dl=0", "?dl=1")

            return {"message": "File uploaded and link generated", "link": public_link}
        else:
            return JSONResponse(content={'error': 'Video not found'}, status_code=404)
        
    except Exception as e:
        traceback_str = traceback.format_exc()
        print("Error:", traceback_str)
        return JSONResponse(content={'error': str(e)}, status_code=500)
    finally:
        os.chdir(original_directory)

def is_folder_empty(folder_path):
    return not any(os.listdir(folder_path))

@app.post("/car_dataset")
async def car_dataset(name: str = Form(...), file: UploadFile = File(...)):
    try:
        if ' ' in name:
            name = name.replace(' ', '_')
        
        weights_path = 'for_annotation.pt'
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
        print("Model loaded for annotations")

        source_video_directory = os.path.join(os.getcwd(), 'Video_for_Dataset')
        print("Source_video_directory: ", source_video_directory)
        source_video_path = os.path.join(source_video_directory, file.filename)
        print("Source Video Path:", source_video_path)
        video_content = file.file.read()
        print("Video Content Length:", len(video_content))
        with open(source_video_path, 'wb') as f:
            print(f)
            f.write(video_content)

        input_folder = './new_car_images/'
        output_folder = './annotated_images/'
        os.makedirs(output_folder, exist_ok=True)

        if not is_folder_empty(input_folder):
            delete_folder_contents(input_folder)
        
        if not is_folder_empty(output_folder):
            delete_folder_contents(output_folder)

        cap = cv2.VideoCapture(source_video_path)

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            frame_filename = os.path.join(input_folder, f"{frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)
            print("File", frame_filename,"Writed")

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

        print("ALL IMAGES SAVED")
        
        for img_file in os.listdir(input_folder):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img_path = os.path.join(input_folder, img_file)
                img = Image.open(img_path)

                results = model(img)
                
                detected_objects = results.xyxy[0].cpu().numpy()
                
                
                annotation_file_path = os.path.join(output_folder, os.path.splitext(img_file)[0] + '.txt')
                with open(annotation_file_path, 'w') as f:
                    for obj in detected_objects:
                        class_id = int(obj[5])
                        x_center = (obj[0] + obj[2]) / 2 / img.width
                        y_center = (obj[1] + obj[3]) / 2 / img.height
                        width = (obj[2] - obj[0]) / img.width
                        height = (obj[3] - obj[1]) / img.height
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                
                
                annotated_img = img.copy()
                draw = ImageDraw.Draw(annotated_img)
                for obj in detected_objects:
                    class_id = int(obj[5])
                    bbox = [obj[0], obj[1], obj[2], obj[3]]
                    draw.rectangle(bbox, outline='red', width=3)
                    draw.text((bbox[0], bbox[1]), f"Class {class_id}", fill='red')
                
                annotated_img.save(os.path.join(output_folder, os.path.splitext(img_file)[0] + '_annotated.jpg'))
        print("All images processed, annotations saved in YOLOv5 format, and annotated images saved.")

        label_files = [file for file in os.listdir(output_folder) if file.endswith('.txt')]
        for label_file in label_files:
            label_file_path = os.path.join(output_folder, label_file)
            image_file_path = os.path.join(input_folder, label_file.replace('.txt', '.jpg'))
            
            if os.path.getsize(label_file_path) == 0:
                os.remove(label_file_path)
                if os.path.exists(image_file_path):
                    os.remove(image_file_path)
                    
                print(f"Deleted empty file: {label_file}")
        dataset_directory = os.path.join(os.getcwd(), 'app', 'YOLOv8-DeepSORT-Object-Tracking','ultralytics', 'yolo','v8','detect',name + '_dataset')
        print(dataset_directory)
        if os.path.exists(dataset_directory):
            shutil.rmtree(dataset_directory)
        os.makedirs(dataset_directory, exist_ok=True)
        print("DatasetDirectory made")

        train_dir = os.path.join(dataset_directory, 'train')
        valid_dir = os.path.join(dataset_directory, 'valid')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(valid_dir, exist_ok=True)
        print("Train and valid folders made")

        train_images_dir = os.path.join(train_dir, 'images')
        valid_images_dir = os.path.join(valid_dir, 'images')
        train_labels_dir = os.path.join(train_dir, 'labels')
        valid_labels_dir = os.path.join(valid_dir, 'labels')
        os.makedirs(train_images_dir, exist_ok=True)
        os.makedirs(valid_images_dir, exist_ok=True)
        os.makedirs(train_labels_dir, exist_ok=True)
        os.makedirs(valid_labels_dir, exist_ok=True)
        print("Images and labels folders made")

        image_files = os.listdir('new_car_images')
        label_files = [file for file in os.listdir('annotated_images') if file.endswith('.txt')]

        total_images = len(image_files)
        train_split = int(0.9 * total_images)
        print("Training images ", train_split)
        
        for i in range(total_images):
            if i < train_split:
                shutil.move(os.path.join('new_car_images', image_files[i]), train_images_dir)
                shutil.move(os.path.join('annotated_images', label_files[i]), train_labels_dir)
            else:
                shutil.move(os.path.join('new_car_images', image_files[i]), valid_images_dir)
                shutil.move(os.path.join('annotated_images', label_files[i]), valid_labels_dir)

        data_yaml_file = f"names:\n- {name}_car\n- {name}_car\nnc: 2\n\ntest: ../test/images\ntrain: {name}_dataset/train/images\nval: {name}_dataset/valid/images\n"
        with open(os.path.join(dataset_directory, 'data.yaml'), 'w') as f:
            f.write(data_yaml_file)
        original_directory = os.getcwd()
        print(original_directory)
        folder_to_delete = os.path.join(os.getcwd() ,'app' ,'YOLOv8-DeepSORT-Object-Tracking' ,'runs' ,'detect')
        delete_subfolders(folder_to_delete)
        yolo_directory = os.path.join(original_directory, 'app', 'YOLOv8-DeepSORT-Object-Tracking','ultralytics', 'yolo','v8','detect')
        os.chdir(yolo_directory)
    
        # command = (
        #         f"python train.py"
        #         f" model=yolo8l.pt data=./{name}_dataset/data.yaml"
        #         " epochs 30 imgsz=640"
        #     )
        # print("YOLOv8 Command:", command)
        # subprocess.run(command, shell=True)
        # os.chdir(original_directory)
        # source_path = os.path.join(os.getcwd(), 'app' ,'YOLOv8-DeepSORT-Object-Tracking' ,'runs' ,'detect', 'train', 'weights', 'best.pt')  
        # destination_path = os.path.join(original_directory, name + '.pt')

        # shutil.move(source_path, destination_path) 
        # print("Weights File Moved")
        return JSONResponse(content={'message': 'Dataset Collected and Model Training Done'}, status_code=200)

    except Exception as e:
        traceback_str = traceback.format_exc()
        print("Error:", traceback_str)
        return JSONResponse(content={'error': str(e)}, status_code=500)
    finally:
        os.chdir(original_directory)

@app.post("/pervious_motion_detection")
async def upload_frame(name: str = Form(...), file: UploadFile = File(...)):
    try:
        if ' ' in name:
            name = name.replace(' ', '_')
        original_directory = os.getcwd()
        source_video_directory = os.path.join(os.getcwd(), 'app','Face_recognition','source_videos')
        print("Source_video_directory: ", source_video_directory)
        source_video_path = os.path.join(source_video_directory, file.filename)
        print("Source Video Path:", source_video_path)
        video_content = file.file.read()
        print("Video Content Length:", len(video_content))
        with open(source_video_path, 'wb') as f:
            print(f)
            f.write(video_content)
        cap = cv2.VideoCapture(source_video_path)

        # ownerModel = YOLO('carDetection.pt')
        # personModel = torch.hub.load('ultralytics/yolov5', 'custom', path='human_weights.pt')
        # PMClasses = personModel.names

        current_time = None
        recording = False
        out = None
        frame_count = 0

        output_path = None
        alerts = []
        motions = []

        starting_time = datetime.datetime.now().strftime('%H:%M:%S')
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            ownerCarDetection = ownerModel(frame)
            ownerCar = ownerCarDetection[0]
            person = personModel(frame)

            motion_detected = False
            alert_detected = False

            for box, det in zip(ownerCar.boxes, person.pred[0]):
                #owner car detection results
                car_name = ownerCar.names[box.cls[0].item()]
                cords = box.xyxy[0].tolist()
                cords = [round(x) for x in cords]
                conf = round(box.conf[0].item(), 2)

                #person detection results
                class_id = int(det[-1])
                class_name = PMClasses[class_id]
                confidence = det[-2]

                if conf > 0.50 and car_name =='Saim_car':
                    x, y, width, height = map(int, cords[:4])
                    print("X",x,"Y",y,"WIDTH",width,"HEIGHT",height)
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)
                    cv2.putText(frame, f'{car_name}: {conf:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # person detection coordinates
                if confidence > 0.60 and class_name == 'Person':
                    x_cord, y_cord, w, h = map(int, det[:4])
                    print("W", w)
                    cv2.rectangle(frame, (x_cord, y_cord), (x_cord + w, y_cord + h), (0, 0, 255), 2)
                    cv2.putText(frame, f'{class_name}: {confidence:.2f}', (x_cord, y_cord - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    center_car = np.array([(x + w) / 2, (y + height) / 2])
                    print(center_car)
                    center_person = np.array([(x_cord + w) / 2, (y_cord + h) / 2])
                    print(center_person)
                    distance = np.linalg.norm(center_car - center_person)
                    print(distance)
                    if int(distance) < 450:
                        frame_image = f'{current_time}_{frame_count}.jpg'
                        frame_image_path = os.path.join(os.getcwd(), 'app', 'Face_recognition', 'frames', frame_image)
                        frame_count+=1
                        cv2.imwrite(frame_image_path, frame)
                        recognition_path = os.path.join(os.getcwd(), 'app', 'Face_recognition')
                        
                        os.chdir(recognition_path)
                        command = (
                            "python face_recognition.py"
                            f" {frame_image_path} {name}"
                        )
                        print("Recognition Command:", command)
                        subprocess.run(command, shell=True)
                        # status_file_path = os.path.join(os.getcwd(), 'app', 'Face_recognition', 'owner_status.txt')
                        with open("owner_status.txt", "r") as f:
                            lines = f.readlines()
                            if lines:
                                owner_status_str = lines[-1].strip()  # Get the last line and remove any leading/trailing whitespace
                            else:
                                owner_status_str = "False"
                            owner = bool(owner_status_str.strip())
                        if owner == False:
                            alert_detected = True
                        else:
                            motion_detected =True
                        
                        # os.remove("owner_status.txt")
                    else:
                        motion_detected = True
            print(alert_detected)
            print(motion_detected)
            # Get current timestamp
            timestamp = datetime.datetime.now().strftime('%H:%M:%S')

            # Save alerts to the alerts list
            if alert_detected:
                alerts.append(f'ALERT THEFT DETECTION at time {timestamp}')
            elif motion_detected:
                alerts.append(f'MOTION DETECTION at time {timestamp}')

            # Save motion detection or alert video
            if motion_detected or alert_detected:
                if not recording:
                    print("STARTED WRITING")
                    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    output_path = f'recordings/{current_time}.mp4'
                    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (frame.shape[1], frame.shape[0]))
                    recording = True

                out.write(frame)
            print("MOTION",motion_detected)
            print("ALERT",alert_detected)
            print("RECORDING", recording)
            # Stop recording if no motion or alert detected
            if not motion_detected and not alert_detected and recording:
                print("VIDEO RELEASED")
                out.release()
                recording = False

            # Display frame with rectangles and text
            cv2.imshow('Motion Detection', frame)
            os.chdir(original_directory)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Save alerts to a text file
        if alerts:
            with open(f'recordings/{current_time}.txt', 'w') as f:
                for alert in alerts:
                    f.write(alert + '\n')

        if motions:
            with open(f'recordings/{current_time}.txt', 'w') as f:
                for motion in motions:
                    f.write(motion + '\n')

        cap.release()
        cv2.destroyAllWindows()
        return JSONResponse(content={'message': "Motion Detection Record Saved"}, status_code=200)
    except Exception as e:
        traceback_str = traceback.format_exc()
        print("Error:", traceback_str)
        return JSONResponse(content={'error': str(e)}, status_code=500)
    finally:
        os.chdir(original_directory)

@app.post("/motion_detection")
async def upload_frame(name: str = Form(...), file: UploadFile = File(...)):
    try:
        if ' ' in name:
            name = name.replace(' ', '_')
        original_directory = os.getcwd()
        source_video_directory = os.path.join(os.getcwd(), 'app','Face_recognition','source_videos')
        print("Source_video_directory: ", source_video_directory)
        source_video_path = os.path.join(source_video_directory, file.filename)
        print("Source Video Path:", source_video_path)
        video_content = file.file.read()
        print("Video Content Length:", len(video_content))
        with open(source_video_path, 'wb') as f:
            print(f)
            f.write(video_content)
        file_to_execute = os.path.join(os.getcwd(), 'app',  'Face_recognition')
        os.chdir(file_to_execute)

        command = (
            f"python face_recognition.py"
            f" {source_video_path} {name}"
        )
        print("Recognition Command:", command)
        subprocess.run(command, shell=True)
    except Exception as e:
        traceback_str = traceback.format_exc()
        print("Error:", traceback_str)
        return JSONResponse(content={'error': str(e)}, status_code=500)
    finally:
        os.chdir(original_directory)
