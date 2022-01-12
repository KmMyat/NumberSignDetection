import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import numpy as np
import cv2
import operator
import sys, os
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model('sbLCNN.h5')
categories = {0: 'ZERO', 1: 'ONE', 2: 'TWO', 3: 'THREE', 4: 'FOUR', 5: 'FIVE'}

class VideoTransformer(VideoTransformerBase):

    def transform(self, frame):

        frame = frame.to_ndarray(format="bgr24")
        frame = cv2.flip(frame, 1)




        x1 = int(0.5*frame.shape[1])
        y1 = 10
        x2 = frame.shape[1]-10
        y2 = int(0.5*frame.shape[1])
        cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
        roi = frame[y1:y2, x1:x2]
        roi = cv2.resize(roi, (64, 64))

        kernel = np.ones((3,3),np.uint8)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0,25,120], dtype=np.uint8)
        upper_skin = np.array([15,145,255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        roi = cv2.GaussianBlur(mask,(5,5),100)
        filtering = cv2.filter2D(roi, -1, kernel)
        roi = cv2.GaussianBlur(filtering,(5,5),100)
        roi = cv2.dilate(roi,kernel, iterations = 1)
        roi = cv2.erode(roi, kernel, iterations = 1)

        while True:
            result = model.predict(roi.reshape(1, 64, 64, 1))
            prediction = {'ZERO': result[0][0], 
                          'ONE': result[0][1], 
                          'TWO': result[0][2],
                          'THREE': result[0][3],
                          'FOUR': result[0][4],
                          'FIVE': result[0][5]}
            # Sorting based on top prediction
            prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
            
            # Displaying the predictions
            cv2.putText(frame, prediction[0][0], (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1) 

        return frame

def newtest():
    
    camera = cv2.VideoCapture(0)

    while True:
        _, frame = camera.read()
        st.image(frame)
    else:
        st.write('Stopped')
    
def main():

    st.title("Real Time Face Emotion Detection Application")
    activiteis = ["Home", "Webcam Face Detection", "About","new"]
    choice = st.sidebar.selectbox("Select Activity", activiteis)
    st.sidebar.markdown(
        """ Developed by Mohammad Juned Khan    
            Email : Mohammad.juned.z.khan@gmail.com  
            [LinkedIn] (https://www.linkedin.com/in/md-juned-khan)""")
    if choice == "Home":
        html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Face Emotion detection application using OpenCV, Custom CNN model and Streamlit.</h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        st.write("""
                 The application has two functionalities.
                 1. Real time face detection using web cam feed.
                 2. Real time face emotion recognization.
                 """)
    elif choice == "Webcam Face Detection":
        st.header("Webcam Live Feed")
        st.write("Click on start to use webcam and detect your face emotion")
        webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

    elif choice == "About":
        st.subheader("About this app")
        html_temp_about1= """<div style="background-color:#6D7B8D;padding:10px">
                                    <h4 style="color:white;text-align:center;">
                                    Real time face emotion detection application using OpenCV, Custom Trained CNN model and Streamlit.</h4>
                                    </div>
                                    </br>"""
        st.markdown(html_temp_about1, unsafe_allow_html=True)

        html_temp4 = """
                             		<div style="background-color:#98AFC7;padding:10px">
                             		<h4 style="color:white;text-align:center;">This Application is developed by Mohammad Juned Khan using Streamlit Framework, Opencv, Tensorflow and Keras library for demonstration purpose. If you're on LinkedIn and want to connect, just click on the link in sidebar and shoot me a request. If you have any suggestion or wnat to comment just write a mail at Mohammad.juned.z.khan@gmail.com. </h4>
                             		<h4 style="color:white;text-align:center;">Thanks for Visiting</h4>
                             		</div>
                             		<br></br>
                             		<br></br>"""

        st.markdown(html_temp4, unsafe_allow_html=True)
    elif choice == 'new':
        newtest()
    else:
        pass


if __name__ == "__main__":
    main()
