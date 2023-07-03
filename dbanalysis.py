from django.shortcuts import render
from .models import *
import json
from django.core import serializers
from django.http import HttpResponse, JsonResponse
from django.db.models import Q
from django.db.models import Count
import re
import os
from datetime import datetime
from datetime import date
from django.views.decorators.cache import never_cache
from django.core.files.storage import FileSystemStorage
import time
from threading import Thread
import cv2
import numpy as np 
import tensorflow as tf
from tensorflow.keras.models import load_model
import paho.mqtt.client as mqtt
import threading
import playsound
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import dlib
import keyboard
import glob
import pywhatkit
import warnings
warnings.filterwarnings('ignore')
client= mqtt.Client()

loaded_model = load_model("DD_app/static/model_mobilenet.h5")
width = 224
height = 224
file_counter=1
# stop_flag = False


####uncomment these lines if u didnt get location details (for test purpose)
latitude=9.875
longitude=76.567


s_time=[]
s_lat=[]
s_long=[]
dis_time=[]
dro_time=[]
dis_lat=[]
dis_long=[]
dro_lat=[]
dro_long=[]

# Create your views here.
@never_cache
def show_index(request):
    return render(request, "login.html", {})


@never_cache
def logout(request):
    if 'uid' in request.session:
        del request.session['uid']
    return render(request,'login.html')


def check_login(request):
    username = request.GET.get("uname")
    password = request.GET.get("password")

    print(username)
    print(password)

    if username=="admin@123" and password=="admin@123":
        request.session["uid"] = "admin"
        return HttpResponse("Login Successful")
    else:
        return HttpResponse("Invalid")

def sound_alarm(path):
    # play an alarm sound
    print ("Drowsy========================================")
    playsound.playsound(path)

@never_cache
###############ADMIN START
def show_home_admin(request):
    if 'uid' in request.session:
        try:
            obj3=Files.objects.all()

            return render(request,'home.html',{'files':obj3})
        except:
            return render(request,'home.html')
    else:
        return render(request,'login.html')



ALARM_ON = False

def distraction():
    # global stop_flag
    # while not stop_flag:
    import time
    global vs1
    vs1 = cv2.VideoCapture(1)
    count = 0  # eye
    max_time = 1 * 60
    start_time = time.time()
    f_counter=0


    while (True):
        ret,frame = vs1.read()
        if not ret:
            break

        # def prediction(path):
        # image=cv2.imread(path)
        resize_image = cv2.resize(frame,(height,width))
        out_image=np.expand_dims(resize_image,axis=0)/255
        # print(out_image.shape)

        my_pred = loaded_model.predict(out_image)
        # print(my_pred)
        my_pred=np.argmax(my_pred,axis=1)
        my_pred = my_pred[0]
        # print(my_pred)

        if my_pred ==0:
            print("Distraction")
            count += 1
            cv2.putText(frame, "Distracted-- Count: {}".format(count), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if count>20:
                if f_counter==0:
                    cv2.putText(frame, "Distraction ALERT!", (60, 120),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    print("Distraction ALERT!")
                    global ALARM_ON
                    if not ALARM_ON:
                        ALARM_ON = True

                        
                        t = Thread(target=sound_alarm,args=("alarm.wav",))
                        t.deamon = True
                        t.start()
                    ################################## Alert 
                    # my_value2 = "distracted"
                    # client.connect("broker.hivemq.com", 1883, 60)
                    # client.publish("drowsy", my_value2) 
                    ##################################
                f_counter+=1
                print("f_counter : ",f_counter)
                if f_counter==5:
                    cv2.putText(frame, "Distraction ALERT!", (60, 120),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    print("Distraction ALERT! Distraction ALERT!")

                    #pywhatkit.sendwhatmsg_instantly("+919061118584","Distraction Detected")
                    ################################## send message to authorized person
                    # my_value2 = "distracted"
                    # client.connect("broker.hivemq.com", 1883, 60)
                    # client.publish("drowsy", my_value2) 
                    ##################################
                    now = datetime.now()
                    time = now.strftime("%H:%M:%S")
                    dis_time.append(time)
                    dis_lat.append(latitude)
                    dis_long.append(longitude)


        elif my_pred ==1:
            count=0
            f_counter=0
            ALARM_ON = False
            print("Normal")
            cv2.putText(frame, "Safe", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


        #show the frame
        cv2.imshow("Distraction Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            t1.join()
            print("clicked break1")
            break
          
    vs1.release()
    cv2.destroyAllWindows()
 




def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])

    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear

#centre point of eye 
EYE_AR_THRESH = 0.2
 
EYE_AR_CONSEC_FRAMES = 10

COUNTER = 0

#face detector
detector = dlib.get_frontal_face_detector()

# Eye,nose .....
predictor = dlib.shape_predictor("DD_app/static/shape_predictor_68_face_landmarks.dat")

# Landmarks of left eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]

(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

def eye_drowsy():
    # global stop_flag
    # while not stop_flag:
    counts=[]
    my_adder=0
    # start the video stream thread
    print("[INFO] starting video stream thread...")
    global vs
    vs = VideoStream(src=0).start()
    # time.sleep(1.0)

    blinkrate=[]

    # loop over frames from the video stream
    COUNTER=0
    while True:
        
        frame = vs.read()

        frame = imutils.resize(frame, width=500)
           
        #grayscale convertion
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        rects = detector(gray, 0)
       

        # loop over the face detections
        for rect in rects:
                # shape prediction
                shape = predictor(gray, rect)

                # Convert to numpy array
                shape = face_utils.shape_to_np(shape)

                # extract the left and right eye coordinates, then use the
                # coordinates to compute the eye aspect ratio for both eyes
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]

                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)

                
                ear = (leftEAR + rightEAR) / 2.0
                # print(ear)

                #extract left eye and right eye,draw points
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)

                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                # check to see if the eye aspect ratio is below the blink
                # threshold, and if so, increment the blink frame counter
                if ear < EYE_AR_THRESH:
                        COUNTER += 1

                        print("Blinked")

                        
                        if COUNTER >= EYE_AR_CONSEC_FRAMES:
                            my_adder+=1

                            if my_adder==1:

                                ################################## Alert 
                                # my_value2 = "drowsy"
                                # client.connect("broker.hivemq.com", 1883, 60)
                                # client.publish("drowsy", my_value2) 
                                ##################################
                               
                                # draw an alarm on the frame
                                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                print("DROWSINESS ALERT!")
                                global ALARM_ON
                                if not ALARM_ON:
                                    ALARM_ON = True

                                    
                                    t = Thread(target=sound_alarm,args=("alarm.wav",))
                                    t.deamon = True
                                    t.start()

                            if my_adder==10:
                                ################################## send message to authorized person
                                # my_value2 = "drowsy"
                                # client.connect("broker.hivemq.com", 1883, 60)
                                # client.publish("drowsy", my_value2) 
                                ##################################
                                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                print("DROWSINESS ALERT! DROWSINESS ALERT!")
   
                                pywhatkit.sendwhatmsg_instantly("+919745450268","[Alert]:Drowsiness Detection")
                                now = datetime.now()
                                time = now.strftime("%H:%M:%S")
                                dro_time.append(time)
                                dro_lat.append(latitude)
                                dro_long.append(longitude)


                # otherwise, the eye aspect ratio is not below the blink
                # threshold, so reset the counter and alarm
                else:   
                        my_adder=0
                        counts.append(COUNTER)
                        COUNTER = 0
                        ALARM_ON = False
                #write text on the image
                # print(COUNTER)
                cv2.putText(frame, "Eye Ratio: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
        # show the frame
        cv2.imshow("Drowsy Detector", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            t2.join()
            print("clicked break2")
            break

    # vs.release()
    vs.stream.release()
    vs.stop()
    cv2.destroyAllWindows()
    

def get_place(received_lat,received_long):
    from geopy.geocoders import Nominatim

    # Initialize geocoding API
    geolocator = Nominatim(user_agent='my-app')


    # Get place name based on latitude and longitude
    location = geolocator.reverse(f"{received_lat}, {received_long}")
    place_name = location.address

    print(place_name) # Prints "Westminster, City of Westminster, London, Greater London, England, SW1A 0AA, United Kingdom"
    return place_name

def generate_fleet_analysis():
    global file_counter
    name='DD_app/static/Fleet_Analysis/file_%s.txt'%(file_counter)
    f1=open(name,'w')
    f1.write("\n**********Driving Report**********\n\n")
    if s_time:
        f1.write("Starting Time : "+str(s_time[0]))
        f1.write("\n")
    if s_lat:
        f1.write("Starting Latitude : "+str(s_lat[0]))
        f1.write("\n")
    if s_long:
        f1.write("Starting Longitude : "+str(s_long[0]))
        f1.write("\n")
        placename=get_place(s_lat[0],s_long[0])
        f1.write("Starting Place : "+str(placename))
        f1.write("\n")

    f1.write("\n")
    if dis_time:
        f1.write("Distraction Detected Time : "+str(dis_time[0]))
        f1.write("\n")
    if dis_lat:
        f1.write("Distraction Detected Latitude : "+str(dis_lat[0]))
        f1.write("\n")
    if dis_long:
        f1.write("Distraction Detected Longitude : "+str(dis_long[0]))
        f1.write("\n")
        placename1=get_place(dis_lat[0],dis_long[0])
        f1.write("Distraction Detected Place : "+str(placename1))
        f1.write("\n")
    f1.write("\n")
    if dro_time:
        f1.write("Drowsiness Detected Time : "+str(dro_time[0]))
        f1.write("\n")
    if dro_lat:
        f1.write("Drowsiness Detected Latitude : "+str(dro_lat[0]))
        f1.write("\n")
    if dro_long:
        f1.write("Drowsiness Detected Longitude : "+str(dro_long[0]))
        f1.write("\n")
        placename2=get_place(dro_lat[0],dro_long[0])
        f1.write("Drowsiness Detected Place : "+str(placename2))
        f1.write("\n")

    f1.close()
    file_counter+=1

    now1 = datetime.now()
    time1 = now1.strftime("%H:%M:%S")

    today1 = date.today()
    current_date1 = today1.strftime("%d/%m/%Y")
    basename=os.path.basename(name)

    obj2=Files(filename=basename,c_date=current_date1,c_time=time1)
    obj2.save()


def webcam(request):
    global s_time,s_lat,s_long
    now = datetime.now()
    time = now.strftime("%H:%M:%S")
    s_time.append(time)
    s_lat.append(latitude)
    s_long.append(longitude)

    # try:
    global t1,t2
    t1=threading.Thread(target=distraction).start()

    t2=threading.Thread(target=eye_drowsy).start()
    

    return render(request,'home.html')
    # return HttpResponse("<script>alert('Fleet Analysis Generated');window.location.href='/show_home_admin/'</script>")

def gen(request):
    global s_time,s_lat,s_long,dis_time,dro_time,dis_lat,dis_long,dro_lat,dro_long

    generate_fleet_analysis()

    s_time=[]
    s_lat=[]
    s_long=[]
    dis_time=[]
    dro_time=[]
    dis_lat=[]
    dis_long=[]
    dro_lat=[]
    dro_long=[]

    obj3=Files.objects.all()

    vs1.release()
    vs.stream.release()
    cv2.destroyAllWindows()

    return render(request,'home.html',{'files':obj3})


def download(request):
    f_id=request.POST.get("f_id")
    filename=request.POST.get("filename")

    if True:
        
        file1_path = "DD_app/static/Fleet_Analysis/"+filename
        print(os.path.exists(file1_path))
        print(file1_path)

        if os.path.exists(file1_path):
            with open(file1_path, 'rb') as fh:
                response = HttpResponse(fh.read(), content_type="application/vnd.ms-excel")
                response['Content-Disposition'] = 'inline; filename=' + os.path.basename(file1_path)
                return response
        raise HttpResponse("<script>alert('File does not exists');window.location.href='/show_home_admin/'</script>")
    return HttpResponse("<script>alert('File Downloaded Successfully');window.location.href='/show_home_admin/'</script>")

@never_cache
def refresh(request):
    files=os.listdir('DD_app/static/Fleet_Analysis/')
    pathh='DD_app/static/Fleet_Analysis/'
    for f in files:
        os.remove(pathh+f)

    obj1=Files.objects.all().delete()

    return HttpResponse("<script>alert('Refreshed Successfully');window.location.href='/show_home_admin/'</script>")


##############  mqtt start ############



# def on_connect(client, userdata, flags, rc):
#     #print("Connected with result code "+str(rc))
#     client.subscribe("drowsymonitor")


# def on_message(client, userdata, msg):##############
#     print(msg.topic+" -- "+str(msg.payload))

#     val=str(msg.payload)[2:-1]       #rfid extracting
#     splitted_val=val.split(',')
#     global latitude,longitude
#     latitude=splitted_val[0]
#     longitude=splitted_val[1]


# client.on_connect = on_connect
# client.on_message = on_message
# client.connect("broker.hivemq.com", 1883, 60)
# def run(n):
#     client.loop_start()
# t3 = threading.Thread(target=run, args=(10,))
# t3.start()
# t3.join()

##############  mqtt end ############
########threading.Thread(target=start).start()
