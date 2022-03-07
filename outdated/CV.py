#%%
import tensorflow as tf
import tensorflow_hub as hub
import cv2
from matplotlib import pyplot as plt
import urllib.request
import urllib.parse
import numpy as np

# Optional if you are using a GPU
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1?tf-hub-format=compressed')
movenet = model.signatures['serving_default']
ROBBERY=[]

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

def sendSMS(apikey, numbers, sender, message):
    data =  urllib.parse.urlencode({'apikey': apikey, 'numbers': numbers,
        'message' : message, 'sender': sender})
    data = data.encode('utf-8')
    request = urllib.request.Request("https://api.txtlocal.com/send/?")
    f = urllib.request.urlopen(request, data)
    fr = f.read()
    return(fr)

def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    #print(shaped.shape)
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 6, (0,255,0), -1)
    
def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 4)

def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):
    y, x, c = frame.shape
    scared_people=0

    for person in keypoints_with_scores:
        draw_connections(frame, person, edges, confidence_threshold)
        draw_keypoints(frame, person, confidence_threshold)
        shaped = np.squeeze(np.multiply(person, [y,x,1]))

        ls_y,ls_x,ls_c=shaped[5] #left shoulder
        lw_y,lw_x,lw_c=shaped[9] #left wrist
        lh_y,lh_x,lh_c=shaped[11] #left hip

        rs_y,rs_x,rs_c=shaped[6] #right shoulder
        rw_y,rw_x,rw_c=shaped[10] #right wrist
        rh_y,rh_x,rh_c=shaped[12] #right hip

        if(ls_c>confidence_threshold)&(lw_c>confidence_threshold)&(rs_c>confidence_threshold)&(rw_c>confidence_threshold ):
            if(rw_y-rs_y<0)&(lw_y-ls_y<0):
                scared_people+=1
        elif(lh_c>confidence_threshold)&(rh_c>confidence_threshold)&(ls_c>confidence_threshold)&(rs_c>confidence_threshold):
            #print(ls_y,lh_y,rs_y,rh_y)
            if((abs(ls_y-lh_y)<35) & (abs(rs_y-rh_y)<35) ):
                scared_people+=1
    if(scared_people>=1): #should be bigger than 1 for multiple people

        print("ROBBERY")

        # FOR SENDING SMS OVER API IN JSON FORMAT
        #resp =  sendSMS('apikey', '905352878636','Barış Köse', 'There is a problem in my location and I need assistance!') 

        """ 
        #DELAYED ALARM
        ROBBERY.append(1)
        if(ROBBERY[-100:]==[1]*100):
            print("ROBBERY")
        """

# Function to loop through each person detected and render

cap = cv2.VideoCapture(0) #0 is webcam, might be different for your computer
while cap.isOpened():
    ret, frame = cap.read()
    
    # Resize image
    img = frame.copy()
    frame_x,frame_y,frame_ch=frame.shape
    frame_ratio=frame_x/frame_y
    relative_x=int(256*frame_ratio)
    
    #print(frame.shape, relative_x)
    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), relative_x,256) #change the numbers relative to frame.shape
    input_img = tf.cast(img, dtype=tf.int32)
    
    # Detection section
    results = movenet(input_img)
    keypoints_with_scores = results['output_0'].numpy()[:,:,:51].reshape((6,17,3))
    
    # Render keypoints 
    loop_through_people(frame, keypoints_with_scores, EDGES, 0.3)
    
    cv2.imshow('Movenet Multipose', frame)
    
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
