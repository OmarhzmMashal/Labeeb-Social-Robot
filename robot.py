import warnings
warnings.filterwarnings("ignore")
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
folder = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, folder)
from pathlib import Path
import glob
import time
import threading
from threading import Thread
import re
import json
import socket
#WEB SCRAPPING 
import requests
import re
from bs4 import BeautifulSoup
#SPEECH DETECTION
from vosk import Model, KaldiRecognizer
import pyaudio
import pyttsx3
#FRAME MANIPLATION
from imutils.video import VideoStream
import imutils
import cv2
#FACE DETECTION
from fer import FER
import face_recognition
import pyodbc
#TEXT DETECTION
import pytesseract
from pytesseract import Output
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
#IMAGE PROCESSING
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
#MASK DETECTION
from mask_detection import mask_detection
#OBJECT DETECTION
from object_detection import object_detection

class omda:   
    visual_memory = {"people": {"name": "", "emotion": "", "mask": ""}, 
                     "objects": {}}
    
    vocal_memory = {"speech": ""}
    
    commands = {"live":0, "eye": ""}
    
    objects_dict={}


    def __init___(self):
        pass
                              
    def input_visual(self):        
        #RETRIEVE KNOWN PEOPLE
        known_face_encodings=[]
        known_face_names=[]
        images_paths = [f for f in glob.glob(folder+"/known_people/"+'*.jpg')]
        for i in range(len(images_paths)):
            known_face_encodings.append(face_recognition.face_encodings(
                face_recognition.load_image_file(images_paths[i]))[0])
            images_paths[i] = images_paths[i].replace(".jpg","")
            known_face_names.append(Path(images_paths[i]).name)

        #MASK AND EMOTION PREP
        detector=FER()
        prototxtPath =folder+ "/mask_detection/deploy.prototxt"
        weightsPath =folder+ "/mask_detection/res10_300x300_ssd_iter_140000.caffemodel"
        faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
        maskNet = load_model(folder+"/mask_detection//mask_detector.model")  
        
        #CAMERA
        video_capture = cv2.VideoCapture(0)
        
        while True:
            if self.commands["live"] == 1:
                #CAPTURE FRAME
                _, frame = video_capture.read()  
                
                #FACE RECOGNITION
                small_frame = imutils.resize(frame, width=400)
                rgb_small_frame = small_frame[:, :, ::-1]
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                name = "unknown"
                face_names = []   
                face_distances=[]
                best_match_index=[]

                for face_encoding in face_encodings:
                    face_distances= face_recognition.face_distance(
                    known_face_encodings, face_encoding)
                    best_match_index.append(np.argmin(face_distances))
     
                
                for i in range(len(face_locations)):
                    if face_distances[best_match_index[i]] < 0.6:
                        name = known_face_names[best_match_index[i]]
                    else: 
                        name = "unkown"
                    face_names.append(name)
                            
                print(face_names)            
                self.visual_memory["people"]["name"]  = face_names
            
                #EMOTION DETECTION
                try:   
                   emotion, score = detector.top_emotion(frame)
                   self.visual_memory["people"]["emotion"] = emotion
                except:
                    emotion="unknown"
                    
                    
                #MASK
                preds=[]
                locs=[]
                try:
                    (locs, preds) = mask_detection.detect_and_predict_mask(small_frame, faceNet, maskNet)
                except:
                    pass

                
                if len(preds) != 0:                    
                    mask_info=[]
                    for i in range(len(preds)):
                        info = locs[i] + tuple(preds[i])
                        mask_info.append(info)

                    for x,y,w,h,mask,nomask in mask_info:    
                        if mask > nomask: 
                            color=(0,255,0)
                            mask = "Mask"
                        else: 
                            color=(0,0,255)
                            mask = "No Mask"
                        self.visual_memory["people"]["mask"] = mask
                        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
                        p1 = [x-30 ,y+ h]
                        p2 = [x-30, y+ h + 90]
                        p3 =  [x+ w  +30, y+ h + 90]
                        p4 = [x+ w + 30, y + h]
                        points = np.array([p1,p2 ,p3 ,p4],dtype = np.int32)
                        cv2.fillPoly(frame, [points], color)
                        cv2.putText(frame, 'Name: ' + str(name) ,(x - 30 , y + h + 20) , cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255),1,1)                                
                        cv2.putText(frame, 'Emo: ' + str(emotion) ,(x - 30 , y + h + 50) , cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255),1,1)                                
                        cv2.putText(frame, str(mask) ,(x-30 , y + h + 80) , cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255),1,1)
                    

                #OBJECT DETECTION
                try:                    
                    detections, labels = object_detection.detect_object(frame)                   
                    for i in np.arange(0, detections.shape[2]):            
                	#Extracting the confidence of predictions
                        confidence = detections[0, 0, i, 2]            
                        #Filtering out weak predictions
                        if confidence > 0.95:
                            idx = int(detections[0, 0, i, 1])
                            #print(idx)
                            self.objects_dict[labels[idx]] = 1
                            self.visual_memory["objects"] = self.objects_dict 
                            #Extracting the index of the labels from the detection
                            #Computing the (x,y) - coordinates of the bounding box        
                            #idx = int(detections[0, 0, i, 1])
                            #Extracting bounding box coordinates
                            #box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                            #(startX, startY, endX, endY) = box.astype("int")
                            #Drawing the prediction and bounding box
                            #label = "{}: {:.2f}%".format(labels[idx], confidence * 100)
                            #cv2.rectangle(frame, (startX, startY), (endX, endY), (0,0,0), 2)
                            #y = startY - 15 if startY - 15 > 15 else startY + 15
                            #cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,0), 2)
                except:
                    pass
                
                if self.commands["eye"] == "read":
                    speech=""
                    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    threshold_img = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                    #Detect words
                    custom_config = r'--oem 3 --psm 6'
                    # now feeding image to tesseract
                    details = pytesseract.image_to_data(threshold_img, output_type=Output.DICT, config=custom_config, lang='eng')
                    total_boxes = len(details['text'])
                    #Draw boxes around words
                    for s in range(total_boxes):
                        if int(details['conf'][s]) > 90:  
                           (x, y, w, h) = (details['left'][s], details['top'][s], details['width'][s],  details['height'][s])
                           cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)        
                    #Extract words
                    parse_text = ''
                    word_list = ''
                    last_word = '' 
                    for word in details['text']:
                         if word!='':
                             word_list += ' ' + word
                             last_word = word
                         if (last_word!='' and word == '') or (word==details['text'][-1]):
                             if int(details['conf'][s]) > 90:  
                                 parse_text += ' ' + word_list 
                                 word_list = ''
                    print(speech)
                    speech = re.sub(r"[-()\"#/@:<>{}\\`*'+=~—§“|._!‘?,¥;»]", "", parse_text)
                    self.respond(speech, "IsThink")
                    self.commands["eye"] = ""
                
                #SHOW FRAME
                cv2.imshow("frame", frame)
                if cv2.waitKey(20) == 27: # exit on ESC
                    break 
            else:
                break
        cv2.destroyWindow("frame")    
                    
    
    
    def input_vocal(self):  
         #INIT VOSK
         model = Model(folder+"/speech_detection")
         rec = KaldiRecognizer(model, 16000)
         p = pyaudio.PyAudio()
         stream = p.open(format=pyaudio.paInt16, channels=1,
                 rate=16000, input=True, frames_per_buffer=8000)
         stream.start_stream()
         while True:
             if self.commands["live"] == 1:
                 data = stream.read(8000)    
                 if len(data) == 0:
                     break
                 if rec.AcceptWaveform(data): 
                     KIRA_RESULT = rec.Result()
                     KIRA_PARSED_JSON = json.loads(KIRA_RESULT)
                     speech = KIRA_PARSED_JSON['text']
                     #STORE DATA WHEN OMDA HEARS STH
                     if speech !="":
                         print("USER: " + str(speech))
                         self.vocal_memory["speech"] = speech
                         self.processor()
             else:
                break

    def respond(self, signal, anim):     
        if signal !=0:
            self.commands["eye"] = ""
            #INIT NETWORK
            UDP_IP_1 = "192.168.137.18"
            UDP_IP_2 = "127.0.0.1"
            UDP_PORT = 5065
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            
            #SEND ANIMATION AND START LIPSYNC
            sock.sendto((anim).encode(), (UDP_IP_2, UDP_PORT))
            #WAIT WHILE THE ANIMATION IS TRIGGERED
            time.sleep(1)
            #STOP ANIMATION
            sock.sendto(("stop1").encode(), (UDP_IP_2, UDP_PORT))
            
            #SPEAK
            engine = pyttsx3.init() 
            engine.setProperty('rate', 170)
            engine.say(signal)
            engine.runAndWait()
            engine.stop()
            
            #STOP LIPSYNC
            sock.sendto(("stop2").encode(), (UDP_IP_2, UDP_PORT))
            
    
    def processor(self): 
        #CREATE CHATBOT OBJECT
        from dnlp import custom_chatbot
        c = custom_chatbot.chatbot()
        
        
        #CREATE DATABASE OBJECT
        conn = pyodbc.connect('Driver={Microsoft Access Driver (*.mdb, *.accdb)};DBQ='+folder+'/Expert_System.accdb;')
        cursor = conn.cursor()

        #GET SENSES MEMORY
        name = self.visual_memory["people"]["name"]
        emotion = self.visual_memory["people"]["emotion"]
        objects = self.visual_memory["objects"]
        speech = self.vocal_memory["speech"]
        checkmask = self.visual_memory["people"]["mask"]
        
        #CHECK IF FACE IS KNOWN
        if self.visual_memory["people"]["name"] != "unknown": isKnown = 1
        else: isKnown = 0
        
        #WEB SCRAPPING
        question_part='what is '
        dot='.'
        whatisidx = speech.find('what is')
        if speech.find('what is') != -1:
            key_word=speech[len(question_part)+ whatisidx:]
            wiki='https://en.wikipedia.org/wiki/'
            wiki=wiki+key_word
            page = requests.get(wiki)
            soup = BeautifulSoup(page.content, 'html.parser')
            sections =soup.find_all('p')
            first_section=''
            for i in sections :
                if(i.get_text()!='\n'):
                    first_section=i.get_text()
                    break
            if not 'may refer to' in first_section:
                try:
                    dot_idx = first_section.index(dot)
                    first_section = first_section[:dot_idx]
                except:
                    pass
                self.respond(first_section, "IsThink")
            else:
                speak = "please be more specific"
                self.respond(speak, "IsHappy")

        #CHECK FOR MASK
        elif speech == 'hello' and self.visual_memory["people"]["mask"] == 'No Mask':
            self.respond("hi please wear your mask robots are also prone to viruses", "IsLaugh")
        
        elif speech == 'hello' and self.visual_memory["people"]["mask"] == 'Mask':
            self.respond("hi thanks for wearing your mask", "IsHappy")
            
        #LIVE STATE
        elif speech == 'exit':
            self.commands["live"] = 0
            
        #READ TEXT
        elif speech == 'read':
            self.commands["eye"] = "read"
         
        #OBJECT DETECT RESPOND    
        elif speech == 'what do you see':       
            speak = ""
            print(objects)
            for obj_name, count in objects.items():                    
                if count == 1: s = ""
                else: s = "s"
                if speak == "":
                    speak = "i see " + str(count) + " " + str(obj_name) +str(s)
                else:
                    speak += " and " + str(count) + " " + str(obj_name)+str(s)
            self.respond(speak, "IsHappy")
        
        #EXPERT SYSTEM
        elif self.visual_memory["people"]["name"] != "":
            sql = str('SELECT Not_Curious_Scenarios.Reply FROM (Emotions INNER JOIN Not_Curious_Scenarios ON Emotions.Emotion_ID = Not_Curious_Scenarios.Emotion_ID) INNER JOIN Texts ON Not_Curious_Scenarios.Text_ID = Texts.Text_ID'
            + str(' WHERE Texts.Text_Content = ?')
            +str(' AND Emotions.Emotion_Name = ?')
            +str(' AND Isknown = ?;'))
            params = (
            speech, 
            emotion,
            isKnown
            )
            cursor.execute(sql, params)  
            records = cursor.fetchall()
            if len(records) == 1: #EXPERT SYS ANSWERS
                for row in records:
                    speak = row[0]
                    
            sql = str('SELECT Not_Curious_Scenarios.Animation FROM (Emotions INNER JOIN Not_Curious_Scenarios ON Emotions.Emotion_ID = Not_Curious_Scenarios.Emotion_ID) INNER JOIN Texts ON Not_Curious_Scenarios.Text_ID = Texts.Text_ID'
            + str(' WHERE Texts.Text_Content = ?')
            +str(' AND Emotions.Emotion_Name = ?')
            +str(' AND Isknown = ?;'))
            params = (
            speech, 
            emotion,
            isKnown
            )
            cursor.execute(sql, params)  
            records = cursor.fetchall()
            if len(records) == 1: #EXPERT SYS ANSWERS
                for row in records:
                    anim = row[0]            
            #CHATBOT
            else:
                try:
                    speak = c.chatbot_response(speech)   
                    anim = "IsHappy"
                except:
                    speak = "Can you say that again"
                    anim = "IsSad"
                    
            
            #SIGNAL THE response TO respond FUNCTION
            self.respond(speak + " " + str(name), anim)

    
    def go(self):
        #CREATE THREADS FOR PARALLEL I/O DEVICES
        self.commands["live"] = 1
        t1 = threading.Thread(target=self.input_vocal)
        t2 = threading.Thread(target=self.input_visual)
        t1.start()
        t2.start()
        t1.join()
        t2.join()


        
if __name__ == '__main__':  
    #CREATE OBJECT
    o = omda()
    #INIT THE THREADS
    o.go()




