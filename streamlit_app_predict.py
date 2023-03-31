import streamlit as st
from roboflow import Roboflow
from pathlib import Path
import os
from PIL import Image
import cv2
import numpy as np





def load_image():
    opencv_image = None 
    path = None
    f = None
    uploaded_file = st.file_uploader(label='Pick an image to test')
    print(uploaded_file)
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        image_data = uploaded_file.getvalue() 
        #st.image(image_data)
        name = uploaded_file.name
        path = os.path.abspath(name)
        print("abs path")
        print(path)
	
        cv2.imwrite("main_image.jpg", opencv_image)
       
    return path, opencv_image
       


	
def loadSegFormModel():
    print("...loading...segformer..")
    rf = Roboflow(api_key="uhDFc9G6MKjrEvbfHt6B")
    project = rf.workspace().project("fleetguardcrack")
    #model = project.version(1).model
    model = project.version(4).model
    return model
	
def segFormCrack(cl, x, y, w, h, cnf, saved_image):
    print(".....inside segFormCrack......")
    img = cv2.imread(saved_image)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #print(img.shape)
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    bias = 0
    roi = img[y-h//2+bias:y+h//2-bias, x-w//2+bias:x+w//2-bias, :]
    #st.image(roi, caption="ROI")
    cv2.imwrite("saved_ROI.jpg", roi)
    segform_model = loadSegFormModel()
    preds = segform_model.predict("saved_ROI.jpg").save("crack_pred.jpg")
    crck_pred = Image.open('crack_pred.jpg')
    st.image(crck_pred, caption='crack localization')
	
def drawBoundingBox(saved_image ,x, y, w, h, cl, cf):
    #img = Image.open(saved_image)
    

    #img = cv2.imread(saved_image)
    img = cv2.cvtColor(saved_image,cv2.COLOR_BGR2RGB)
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    start_pnt = (x-w//2,y-h//2)
    end_pnt = (x+w//2, y+h//2)
    txt_start_pnt = (x-w//2, y-h//2-15)
    if(cl == "Crack" or cl == "No-Crack"):
        cl = "Non-Broken"
    
    img = cv2.rectangle(img, start_pnt, end_pnt, (0,255,0), 10)
    img = cv2.putText(img, cl, txt_start_pnt, cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 10, cv2.LINE_AA)	
    st.image(img, caption='Resulting Image')	
    


def predict(model, url):
    return model.predict(url, confidence=40, overlap=30).json()
    #return model.predict(url, hosted=True).json()
	
	
def main():
    st.title('Crack Detection Demo')
    option = st.selectbox('Image Type',('Normal', 'Zoomed-in'))
   
    #Model api for fleet-crack trained on 25_03_23	
    #rf = Roboflow(api_key="SNxIPCnRCYWXUM9lBAp4")
    #project = rf.workspace().project("fleet-crack-2-wg5qy")
    #model = project.version(1).model
     
    image, svd_img = load_image()
    #st.write('Enter the image URL')
    #url = st.text_input('URL', '')
    result = st.button('Predict')
    if(result and option == "Normal"):
        st.write('Calculating results...')
        #Model api for FleetGuard trained on 27_03_23
        rf2 = Roboflow(api_key="uhDFc9G6MKjrEvbfHt6B")
        project2 = rf2.workspace().project("fleetguard")
        model2 = project2.version(1).model
	
        #Model api for FleetGuard trained on 31_03_23
        #rf3 = Roboflow(api_key="pAMgIogxlcl3zF1xSCsE")
        #project3 = rf3.workspace().project("fleetguardobjdet")
        #model3 = project3.version(1).model
	
	
        results = predict(model2, svd_img)
        #results = predict(model2, url)
        print("Prediction Results are...")	
        print(results)
        if len(results['predictions']) == 0:
            st.image(svd_img)
            st.write("No object is detected")
        else:
		
            new_img_pth = results['predictions'][0]['image_path']
            x = results['predictions'][0]['x']
            y = results['predictions'][0]['y']
            w = results['predictions'][0]['width']
            h = results['predictions'][0]['height']
            cl = results['predictions'][0]['class']
            cnf = results['predictions'][0]['confidence']
            print("printing saved image")
            #print(svd_img.name)
	
            #st.image(svd_img, "saved image")
            drawBoundingBox(svd_img,x, y, w, h, cl, cnf)
            #st.write(cl)
            #st.write(cnf)
            if(cl == "Crack" or cl == "No-Crack"):
                sem_seg_res = segFormCrack(cl, x, y, w, h, cnf, "main_image.jpg")

    elif(result and option == "Zoomed-in"):
        st.write('Calculating results...')
        segform_model = loadSegFormModel()
        preds = segform_model.predict("main_image.jpg").save("crack_pred.jpg")
        crck_pred = Image.open('crack_pred.jpg')
        st.image(crck_pred, caption='crack localization')
        
  	
                  
        
       
        #st.image(res_img, caption='Resulting Image')
        
	#results = predict(model, path)
        #cls_res = results["predictions"][0]["predictions"][0]["class"]
        #conf_res = results["predictions"][0]["predictions"][0]["confidence"]
        #if(cls_res == "Crack" and conf_res <= 0.57):
        #    cls_res = "No-Crack"

        #st.write(cls_res)
        #st.write(conf_res)
        #print(results["predictions"][0]["predictions"][0]["class"])
        #print(results["predictions"][0]["predictions"][0]["confidence"])



    
    

if __name__ == '__main__':
    main()
