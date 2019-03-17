import nltk
import numpy as np
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pyttsx3
import speech_recognition as sr
import wikipedia
from textblob import TextBlob
from s2sb import talk
r = sr.Recognizer()

engine = pyttsx3.init()
engine.setProperty('rate',150)#100 words per minute
engine.setProperty('volume',0.9) 
a = (engine.getProperty('voices'))
engine.setProperty('voice', a[1].id)

flag=True

while(flag==True):
    #with sr.Microphone() as source:
   #     print('Say Something')
      #  r.adjust_for_ambient_noise(source, duration=5)
     #   audio = r.listen(source)
     #   print('Done')
	
#    try:
    user_response = input()
      #  user_response = r.recognize_google(audio)
    

		
        #user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("Friday: You are welcome..")
            engine.say("You are welocome")
            engine.runAndWait()

        else:
            user_response = talk(user_response)
            print(user_response)
            engine.say(user_response)
            engine.runAndWait()
    else:
        flag=False
        print("Friday: Bye! take care..")
        engine.say('Bye take care')
        engine.runAndWait()
 #   except Exception:
    #    print("Something wrong")