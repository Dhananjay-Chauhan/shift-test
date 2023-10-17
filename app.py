from flask import Flask, render_template, Response, request,redirect,url_for,jsonify
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread
import mediapipe as mp

camera_active = False
cap=None
global capture,rec_frame, grey, switch, neg, face, rec, out 
capture=0
grey=0
neg=0
face=0
switch=1
rec=0

#make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass

#Load pretrained face detection model    
# net = cv2.dnn.readNetFromCaffe('./saved_model/deploy.prototxt.txt', './saved_model/res10_300x300_ssd_iter_140000.caffemodel')

#instatiate flask app  
app = Flask(__name__, template_folder='./templates')



def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 
 

def gen_frames(cap):  # generate frame by frame from camera
    # global out, capture,rec_frame
    mp_drawing = mp.solutions.drawing_utils # Drawing helpers
    mp_holistic = mp.solutions.holistic # Mediapipe Solutions
    # mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    if camera_active ==True:
        counter = 0 
        stage = None
        # Initiate holistic model
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            # cap = cv2.VideoCapture(0)
            while cap.isOpened():
                ret, frame = cap.read()
                
                # Recolor Feed
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False        
                
                # Make Detections
                results = holistic.process(image)
                # print(results.face_landmarks)
                
                # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
                
                # Recolor image back to BGR for rendering
                image.flags.writeable = True   
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Get coordinates
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    

                    # Calculate angle
                    angle = calculate_angle(shoulder, elbow, wrist)
                    
                    # Visualize angle
                    cv2.putText(image, str(angle), 
                                tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )
                    
                    # Curl counter logic
                    if angle > 160:
                        stage = "down"
                    if angle < 30 and stage =='down':
                        stage="up"
                        counter +=1
                        print(counter)
                            
                except:
                    pass
                
                # Render curl counter
                # Setup status box
                cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
                
                # Rep data
                cv2.putText(image, 'REPS', (15,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter), 
                            (10,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                
                # Stage data
                cv2.putText(image, 'STAGE', (65,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, stage, 
                            (60,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                
                
                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                        )               
                
                cv2.imshow('Mediapipe Feed', image)
                
                                
                # cv2.imshow('Make your Pose', image)
                
                if cv2.waitKey(30) & camera_active==False:
                    cap.release()
                    break
                # if cv2.waitKey(10) & 0xFF == ord('q'):
                #     break

            # if camera_active==False:
            cap.release()
            cv2.destroyAllWindows()
            # redirect(url_for('index'))
    else:
        print('colosed')
    

@app.route('/')
def index():
    return render_template('index.html')
    
# @app.route('/help')
# def help():
#    return Response(redirect(url_for('index')))
@app.route('/restart', methods=['POST'])
def restart_server():
    os.execl(sys.executable, sys.executable, *sys.argv)

@app.route('/video_feed')
def video_feed():
    
    return Response(gen_frames(cap), mimetype='multipart/x-mixed-replace; boundary=frame')
   



@app.route('/start_camera' )
def start_camera():
    global camera_active
    global cap
    if not camera_active:
        # Add code here to start the camera
        camera_active = True
        cap = cv2.VideoCapture(0)
        return redirect(url_for('video_feed'))
        
    else:
        return {'success': False, 'message': 'Camera is already active.'}

@app.route('/stop_camera')
def stop_camera():
    global camera_active
    global cap
    if camera_active:
        # Add code here to stop the camera
        camera_active = False
        cap.release()
        cap=None
        # cv2.destroyAllWindows()
        return redirect(url_for('video_feed'))
    else:
        return {'success': False, 'message': 'Camera is not active.'}


if __name__ == '__main__':
    app.run(debug=True)

    
