import cv2
import numpy as np
import os 

attend = {"Jadeed":0, "Erin":0, "Anupama":0, "Hana":0, "Fadhiya":0}
lister =[]

#Method for checking existence of path i.e the directory
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

# Create Local Binary Patterns Histograms for face recognization
recognizer = cv2.face.LBPHFaceRecognizer_create()

assure_path_exists("saved_model/")

# Load the  saved pre trained mode
recognizer.read('saved_model/s_model.yml')

# Load prebuilt classifier for Frontal Face detection
cascadePath = "haarcascade_frontalface_default.xml"

# Create classifier from prebuilt model
faceCascade = cv2.CascadeClassifier(cascadePath);

# font style
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize and start the video frame capture from webcam
# cam = cv2.VideoCapture(0)


# Looping starts here
while True:
    # Read the video frame
    # ret, im =cam.read()

    im = cv2.imread("C:/Users/HP/Pictures/test3.jpg", cv2.IMREAD_COLOR)
    # Convert the captured frame into grayscale
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    # Getting all faces from the video frame
    faces = faceCascade.detectMultiScale(gray, 1.2,5) #default

    # For each face in faces, we will start predicting using pre trained model
    for(x,y,w,h) in faces:

        # Create rectangle around the face
        cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)

        # Recognize the face belongs to which ID
        Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])  #Our trained model is working here

        # Set the name according to id
        if Id == 1:
            Id = "Jadeed {0:.2f}%".format(round(100 - confidence, 2))
            attend["Jadeed"] = round(100 - confidence, 2)
            # Put text describe who is in the picture
        elif Id == 2 :
            Id = "Erin {0:.2f}%".format(round(100 - confidence, 2))
            attend["Erin"] = round(100 - confidence, 2)
            # Put text describe who is in the picture
        elif Id == 3:
            Id = "Hana {0:.2f}%".format(round(100 - confidence, 2))
            attend["Hana"] = round(100 - confidence, 2)
        elif Id==4:
            Id = "Anupama {0:.2f}%".format(round(100 - confidence, 2))
            attend["Anupama"] = round(100 - confidence, 2)
        elif Id==5:
            Id = "Fadhiya {0:.2f}%".format(round(100 - confidence, 2))
            attend["Fadhiya"] = confidence
        else:
            pass

        # Set rectangle around face and name of the person
        cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
        cv2.putText(im, str(Id), (x,y-40), font, 1, (255,255,255), 3)

        confidence = 0

    # Display the video frame with the bounded rectangle
    cv2.imshow('im',im) 

    # press q to close the program
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Terminate video

# Close all windows
cv2.destroyAllWindows()

print(attend)

for key, value in attend.items():
    if value >= 50:
        lister.append(key)

print(lister)
f = open("attendance.txt","w+")

for value in lister:
    f.write(value)
    f.write("\n")

f.close
