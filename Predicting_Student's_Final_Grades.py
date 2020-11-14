# linear regression eğer data içindeki değişkenlerin kendi aralarında güçlü bir ilişki varsa kullanılır !!
# Kütüphaneleri aktaralım
import numpy as np
import pandas as pd
from sklearn import linear_model
import sklearn
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

data = pd.read_csv("student-mat.csv", sep=';')  # data array imizin içine internetten bulduğumuz veriyi yazıyoruz

print(data.head())  # datamızı görmek amacıyla yazdırıyoruz(kırpılmış data için .head komutunu kullanıyoruz

data = data[["G1", "G2", "G3", "studytime", "failures","absences"]]  # burada ise datamızda istediğmiz değişkenleri kırpıp datamıza yerleştiriyoruz

print(data.head())  # kırpılmış datamızı yazdırıyoruz ( .head metodu datanızın ilk 5 elemanını kırpabilmenize yarar.)

predict = "G3"  # tahmin edeceğimiz labelı belirtiyoruz

X = np.array(data.drop([predict], 1))  # datamızdan G3 yi kesip yeni bir data oluşturuyoruz (tahmin edilecek için kullanılacak data)

y = np.array(data[predict])  # sadece labelımızın olduğu bir array oluşturuyoruz (tahmin edilecek attribute)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

"""best=0

for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1) #burada ise datamızı 4 parçaya bölüyoruz

    #x_test ve x_train verisi verdiğimiz inputlar y_test ve y_train tahmin etmeye çalışacağımız outputlar

    linear = linear_model.LinearRegression() #algoritmamızı uyguluyoruz

    linear.fit(x_train, y_train) #train datamız için best fit line ımızı oluşturuyoruz yani ai ımızı eğitiyoruz

    accuracy = linear.score(x_test, y_test) #tahmin skorummuzu buluyoruz

    print(accuracy)


    if accuracy > best:

        best = accuracy

        with open("Student Final Grade Model.pickle","wb") as f: #modelimizi kaydediyoruz yani bir dosyaya yazıyoruz
            pickle.dump(linear, f)
"""

pickle_in = open("Student Final Grade Model.pickle", "rb")  # açıyoruz ve okutuyoruz

linear = pickle.load(pickle_in)  # açtığımız ve okuttuğumuz kayıt edilen modelimizi kullanacağımızı belirtiyoruz

print("Coefficients = ")  # weights
print(linear.coef_)  # değişkenlerimizin tahmin edeceğimiz label a etkisini öğrenmek için katsayıları buluyoruz
print("Intercept = ")
print(linear.intercept_)  # biases

predictions = linear.predict(
    x_test)  # tahminler değişkeni oluşturuyoruz ve eğittiğimiz ai ı yukarıda datadan kestiğimiz x_test array i ile test ediyoruz

for x in range(len(predictions)):  # tahmiblerinizi ve asıllarını karşılaştırmak için yan yana yazdırıyoruz
    print(int(predictions[x]), x_test[x], y_test[x])

#best fit line ımızı gözlemlemek için bir grafik çıkarıyoruz

style.use("ggplot")

p = 'absences' #diğer değişkenleri de deneyebiliriz "studytime","G2","failures","absences"

plt.scatter(data[p], data["G3"])
plt.xlabel(p)
plt.ylabel("Final Grade")
plt.show()
