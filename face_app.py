import streamlit as st
import pickle
from io import BytesIO
from keras.models import load_model
import numpy as np
from keras.preprocessing import image
import cv2
from PIL import Image


cnn = load_model('anand_vgg16_model.h5')

def classify(num):
    if num[0][0]==0:
        return 'anand'
    else:
        return 'unknown'

def ocv(img):
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) #faces will store the information of all the faces as an array
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cropped_face = img[y:y+h, x:x+w]
        st.image(cropped_face)
        return cropped_face

def main():
    st.title("Face Recognition")
    html_temp = """
    <div style="background-color:teal ;padding:20px">
    <h2 style="color:white;text-align:center;">Face Recognition</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    file=st.file_uploader("Upload image",type=["jpg"])
    show_file=st.empty()
    if not file:
        show_file.info("Please upload an image : {}".format(' '.join(["jpg"])))
        return

    content = file.getvalue()
    st.image(content,caption='face',use_column_width=True)


    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    test_image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    #test_image=ocv(test_image)

    test_image = cv2.resize(test_image, (224, 224))
    test_image = Image.fromarray(test_image,'RGB')
    test_image = image.img_to_array(test_image)#predict method expects a 2 d array
    test_image = np.expand_dims(test_image, axis = 0)#to fit in batch of 32

    st.text('The face is :')
    st.success(classify(cnn.predict(test_image)))

if __name__=='__main__':
    main()


#The div tag is generally used by web developers to group HTML elements together and apply 
#CSS styles to many elements at once. For example: If you wrap a set of paragraph elements into
 #a div element so you can take the advantage of CSS styles and apply font style to all
#  paragraphs at once instead of coding the same style for each paragraph element.

