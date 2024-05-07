import tensorflow as tf
import numpy as np
import pandas as pd
from keras.models import load_model
from PIL import Image, ImageOps
import cv2
from playsound import playsound
from gtts import gTTS
import os
import argparse


class Detection:
    """
    A Custom class that recieves a image, performs prediction and plays the description using microphone.
    """
    def __init__(self,image_path,weights,labels,csv_file):
        
        self.image_path=image_path
        self.weights=weights
        self.labels=labels
        self.model=load_model(weights, compile=False)
        self.class_names = open(labels, "r").readlines()
        self.csv_file=csv_file
        self.size=(224,224)

    
    def classify_image(self):
        # Read Image
        self.image=Image.open(self.image_path)

        # Image Preprocessing
        self.image=ImageOps.fit(self.image,self.size)
        self.data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        self.image = np.asarray(self.image)[:,:,:3]
        self.image = (self.image.astype(np.float32) / 127.5) - 1
        self.data[0] = self.image

        # Prediction
        prediction = self.model.predict(self.data)
        index = np.argmax(prediction)
        class_name = self.class_names[index]
        class_name=class_name.split(' ')[1]+' '+class_name.split(' ')[2][:-1]
        confidence_score = prediction[0][index]
        
        print(class_name,confidence_score)

        return class_name,confidence_score

    def retrieve_label(self,class_name):
        
        df=pd.read_csv(self.csv_file)
        text=str(df.loc[df['Class']==class_name]['Description'].iloc[0])
        return text

    def play_audio(self,text,class_name):
        myobj=gTTS(text=text,slow=False)
        if os.path.exists(class_name+'.mp3'):
            os.remove(class_name+'.mp3')
            
        myobj.save(class_name+'.mp3')
        playsound(class_name+'.mp3')
        
        

    def run(self):
        
        class_name,confidence_score=self.classify_image()
        text=self.retrieve_label(class_name)
        self.play_audio(text,class_name)


def main():
    parser=argparse.ArgumentParser(description="Running a Image through Object detection pipeline")
    parser.add_argument('image_path',type=str)
    parser.add_argument('weights_path',type=str)
    parser.add_argument('labels_path',type=str)
    parser.add_argument('csv_path',type=str)

    args=parser.parse_args()
    
    d=Detection(args.image_path,args.weights_path,args.labels_path,args.csv_path)
    d.run()

if __name__=='__main__':
    main()

