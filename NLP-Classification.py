import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("Yelp Restaurant Reviews.csv")
data["Liked"] = 0
data["Review"] = data["Review Text"]
for i in range (0, len(data)):
    if data["Rating"][i] > 3:
        data["Liked"][i] = 1
    else:
        data["Liked"][i] = 0
data = data.drop(["Yelp URL", "Date", "Rating", "Review Text"], axis=1)
data = data.head(1000)

import re
import nltk
nltk.download("stopwords")
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
from nltk.corpus import stopwords

derlem = []
for i in range(len(yorumlar)):
    text = re.sub("[^a-zA-Z]", " ", data["Review"][i]) #Noktalama İşaretlerini Kaldırma
    text = text.lower() #Büyük-Küçük Harfleri Aynı Tipe Getirme (Küçük Harf Yapma)
    text = text.split() #Cümle İçindeki Kelimeleri Liste Haline Getirme
    text = [ps.stem(kelime) for kelime in text if not kelime in set(stopwords.words("english"))] #Stop Wordleri Temizleme 
    text = " ".join(text) #Tekrar String(Cümle) Haline Getirme
    derlem.append(text)

#Öznitelik Çıkarımı - Feature Extracting
#Bag Of Words (BAG)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2000) #En fazla kullanılan 2000 kelimeyi al

X = cv.fit_transform(derlem).toarray()
y = data.iloc[:,0].values

#Sınıflandırma
from sklearn.model_selection import train_test_split #Test verilerini ve Train verilerini belirleme
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.2)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
print(ac, "\n", cm)
