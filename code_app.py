import pandas as pd
#import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
#from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_roc_curve, plot_precision_recall_curve, confusion_matrix, accuracy_score, mean_squared_error, roc_auc_score, roc_curve, cohen_kappa_score,classification_report
from sklearn.linear_model import LogisticRegression
#from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn import tree
#from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import streamlit as st
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

st.title("Kategorik Veri Çözümlemesi")
st.sidebar.title("Analiz Araçları")
st.sidebar.markdown("Veri Hakkında")

st.cache(persist=True)
veri = pd.read_csv("kategorik_anket.csv")
veri.drop(["Unnamed: 4"], axis=1, inplace=True)


if st.sidebar.checkbox("Veriyi Göster", False):
    st.subheader("Hacettepe Üniversitesi Öğrencilerinin Kitap Okuma Alışkanlıklarına Ait Kategorik Veriler ")
    st.markdown("Veri setinde 20 değişken ve her bir değişkende 100 gözlem bulunmaktadır. Her ne kadar yalnızca yaş değişkeni nicellik belirtse de; akademik ortalama, sınıf ve toplam kitap sayısı da nicelik belirtmektedir. Ancak anket belli aralık olarak oluşturulduğundan kategorik veri tipinde bulunur.")
    st.write(veri)
    
    a = veri["Cinsiyetiniz"].value_counts()
    b = veri["Kitap Okuma Alışkanlığınızı Nasıl Ölçeklendirirsiniz?"].value_counts()
    st.write(a)
    st.write(b)
    
    kategori= [59,41]
    etiketler = ["Yüksek","Düşük"]
    
    fig, ax = plt.subplots()
    plt.pie(kategori, labels=etiketler)
    plt.legend(title= "Etiketler")
    st.pyplot(fig)
    
    st.markdown("Veri seti içerisinde hiçbir özellikte eksik veri bulunmamaktadır. Ayrıca analiz öncesinde kategorik verilerde değişken dönüşümü gerçekleştirilmiştir.")
    info = Image.open("info.jpg")    
    st.image(info, caption=("Veri Tipleri"))
    
## One hot encoding
label_encoding= preprocessing.LabelEncoder()
#veri["Kitap Okuma Alışkanlığınızı Nasıl Ölçeklendirirsiniz?"] = label_encoding.fit_transform(veri["Kitap Okuma Alışkanlığınızı Nasıl Ölçeklendirirsiniz?"])
veri["Cinsiyetiniz"]= label_encoding.fit_transform(veri["Cinsiyetiniz"])
veri["En çok hangi eserleri okursunuz?"] = label_encoding.fit_transform(veri["En çok hangi eserleri okursunuz?"])
veri["Çizgi Romanların kitap okuma alışkanlığı kazandırdığını düşünüyor musunuz?"] =label_encoding.fit_transform(veri["Çizgi Romanların kitap okuma alışkanlığı kazandırdığını düşünüyor musunuz?"])
veri["Sevmediğiniz kitapları yarım bırakır mısınız?"] = label_encoding.fit_transform(veri["Sevmediğiniz kitapları yarım bırakır mısınız?"])
veri["Okuduğunuz bölümün sizi kitap okumaya teşvik ettiğini düşünüyor musunuz?"] = label_encoding.fit_transform(veri["Okuduğunuz bölümün sizi kitap okumaya teşvik ettiğini düşünüyor musunuz?"])
veri["Üniversiteye başladıktan sonra kitap okuma alışkanlığınızda değişiklik oldu mu?"] = label_encoding.fit_transform(veri["Üniversiteye başladıktan sonra kitap okuma alışkanlığınızda değişiklik oldu mu?"])
veri["Çevreniz kitap okuma alışkanlığınızı etkiliyor mu?"] = label_encoding.fit_transform(veri["Çevreniz kitap okuma alışkanlığınızı etkiliyor mu?"])


# In[6]:


veri["Kitap Okuma Alışkanlığınızı Nasıl Ölçeklendirirsiniz?"] = veri["Kitap Okuma Alışkanlığınızı Nasıl Ölçeklendirirsiniz?"].map({"Yüksek":1, "Düşük":0})


# In[7]:


veri = pd.get_dummies(data=veri, columns=["Fakülteniz"])
veri = pd.get_dummies(data=veri, columns=["Sınıfınız"])
veri = pd.get_dummies(data=veri, columns=["Genel Akademik Not Ortalamanız"])
veri = pd.get_dummies(data=veri, columns=["Okuduğunuz kitapların seçiminde etkili olan faktör nedir?"])
veri = pd.get_dummies(data=veri, columns=["En çok hangi tür eserleri okursunuz?"])
veri = pd.get_dummies(data=veri, columns=["Evinizde ne kadar kitap vardır?"])
veri = pd.get_dummies(data=veri, columns=["Kitap okumayı en sevdiğiniz yer neresidir?"])
veri = pd.get_dummies(data=veri, columns=["Sizi okumaya motive eden nedir?"])
veri = pd.get_dummies(data=veri, columns=["Hangi vakitler kitap okumayı tercih edersiniz?"])


st.cache(persist=True)
y = veri["Kitap Okuma Alışkanlığınızı Nasıl Ölçeklendirirsiniz?"]
x= veri.drop(["Zaman damgası","Kitap Okuma Alışkanlığınızı Nasıl Ölçeklendirirsiniz?"], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.20, random_state=42)

def metrics_list(metric):
    if "Confusion Matrix" in metric:
        st.subheader("Confusion Matrix")
        plot_confusion_matrix(model, x_test, y_test)
        st.pyplot()
        
    if "Roc Curve" in metric:
        st.subheader("Roc Curve")
        plot_roc_curve(model, x_test, y_test)
        st.pyplot()
        
    if "Precision Recall Curve" in metric:
        st.subheader("Precision Recall Curve")
        plot_precision_recall_curve(model, x_test, y_test)
        st.pyplot()
    
    if "Ağaç Yapısı" in metric:
        fig,ax = plt.subplots()
        fig = plt.figure(figsize=(60,55))
        tree.plot_tree(model,rounded=True, class_names=["Düşük","Yüksek"], feature_names=x.columns, filled=True)
        st.pyplot(fig)
        
    if "Features Plot" in metric:
        st.subheader("Features Plot")
        fig,ax = plt.subplots(figsize=(12,10))
        feature = pd.Series(model.feature_importances_,
                               index=x_train.columns).sort_values(ascending=False)
        sns.barplot(x=feature, y=feature.index)
        plt.xlabel("Değişkenler")
        plt.ylabel("Değişkenlerin önem skoru")
        plt.title("Değişken önem düzeyleri")
        st.pyplot(fig)
    
st.sidebar.subheader("Sınıflandırma Algoritmasını Seçiniz")
algoritma = st.sidebar.selectbox("Algoritmalar", ("Lojistik Regresyon", "Karar Ağacı", "Destek Vektör Makinesi", "Gradient Boosting", "Rastgele Ormanlar"))

if algoritma == "Lojistik Regresyon":
    st.sidebar.header("Hiperparametre Seçimi Yapınız")
    solver = st.sidebar.radio("Solver", ("lbfgs", "liblinear"), key="solver") 
    metric= st.sidebar.multiselect("Görselleştirme Tekniğini Seçiniz", ("Confusion Matrix", "Roc Curve", "Precision Recall Curve"))
    
    if st.sidebar.button("Analiz"):
        st.subheader("Lojistik Regresyon")
        model = LogisticRegression(solver=solver)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuary = accuracy_score(y_test, y_pred)
        
        st.write("Accuracy: ", accuary)
        metrics_list(metric)

if algoritma == "Karar Ağacı":
    st.sidebar.header("Hiperparametre Seçimini Yapınız")
    max_depth = st.sidebar.number_input("max_depth (Ağaç Genişliği)", 1,40, step=3)
    min_samples_split = st.sidebar.slider("Min Samples Split", 2, 50)
    metric = st.sidebar.multiselect("Görselleştirme Tekniğini Seçiniz", ("Confusion Matrix", "Roc Curve", "Precision Recall Curve", "Ağaç Yapısı"))
    
    if st.sidebar.button("Analiz"):
        st.subheader("Karar Ağacı")
        model = DecisionTreeClassifier(max_depth= max_depth, min_samples_split=min_samples_split)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuary = accuracy_score(y_test, y_pred)
        
        st.write("Accuary: ", accuary)
        metrics_list(metric)

if algoritma == "Destek Vektör Makinesi":
    st.sidebar.subheader("Hiperparametre")
    C = st.sidebar.number_input("C (Düzgünleştirme Parametresi)", 0.01, 10.0, step=0.01, key="C")
    kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key="kernel") 
    gamma = st.sidebar.radio("Gamma (Kernal coefficient", ("scale", "auto"), key="gamma")
    metric= st.sidebar.multiselect("Görselleştirme Tekniğini Seçiniz", ("Confusion Matrix", "Roc Curve", "Precision Recall Curve"))

    if st.sidebar.button("Analiz"):
        st.subheader("Destek Vektör Makineleri")
        model = SVC(C=C, kernel=kernel, gamma=gamma)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuary = accuracy_score(y_test, y_pred)
        st.write("Accuary", accuary)
        metrics_list(metric)
        
if algoritma == "Gradient Boosting":
    st.sidebar.subheader("Hiperparametre seçimi")
    learning_rate = st.sidebar.number_input("learning_rate (Öğrenme Oranı)", 0.1,  0.5, step=0.01)
    n_estimators = st.sidebar.slider("Ağaç sayısı", 100,600)
    max_depth = st.sidebar.number_input("max_depth (Ağaç Genişliği)", 1,40, step=3) 
    metric = st.sidebar.multiselect("Görselleştirme Tekniğini Seçiniz", ("Confusion Matrix", "Roc Curve", "Precision Recall Curve"))
 
    if st.sidebar.button("Analiz"):
        st.subheader("Gradient Boosting")
        model = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators= n_estimators, max_depth=max_depth)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuary = accuracy_score(y_test, y_pred)
        st.write("Accuary", accuary)
        metrics_list(metric)
        
if algoritma== "Rastgele Ormanlar":
    st.sidebar.subheader("Hiperparametre seçimi")
    n_estimators = st.sidebar.slider("Ağaç sayısı", 100,600)
    max_features = st.sidebar.number_input("Maksimum Özellik", 10,40, step=5)
    min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20)
    metric = st.sidebar.multiselect("Görselleştirme Tekniğini Seçiniz", ("Confusion Matrix", "Roc Curve", "Precision Recall Curve", "Features Plot"))

    if st.sidebar.button("Analiz"):
        st.subheader("Rastgele Ormanlar")
        model = RandomForestClassifier(max_features= max_features, n_estimators= n_estimators, min_samples_split= min_samples_split)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuary = accuracy_score(y_test, y_pred)
        st.write("Accuary", accuary)
        metrics_list(metric)

if st.sidebar.checkbox("Model Karşılaştırmaları"):
    st.subheader("Kurulan Sınıflandırma Modellerinin Başarı Kıyaslamaları")
    image = Image.open('indir.jpg')
    st.image(image, caption='ROC- AUC Değerleri')
    st.markdown("Hacettepe Üniversitesi öğrencileri ile yapılan ankette; kitap okuma alışkanlığını etkileyen faktörler ve öğrencilerin bulundukları bölümler arasındaki ilişki üzerine bir çalışma gerçekleştirilmiştir. Çalışmanın lojistik regresyon kısmı istatistiksel olarak detaylı bir şekilde R programı üzerinden yapılmıştır. Python üzerinden gerçekleştirilen uygulamalarda, sınıflandırma algoritmalarının tahmin performansına odaklanılmıştır.")
    st.markdown("1-) Lojistik Regresyon : Lojistik Regresyon, doğrusal regresyon varsayımlarının kategorik değişkenlerde uygulanmaması dolayısıyla geliştirilen regresyon metodudur. Genelleştirilmiş Doğrusal Modeller konusunda yer alan Lojistik Regresyon için logit bağ fonksiyonu kullanılmaktadır. Bu fonksiyon, bağımlı kategorik değişkenin dönüşüm ile 0-1 halini almasıyla regresyon işleminin kurulmasını sağlar. Özellikle algoritmanın 'solver' parametresinde belirtilen yöntem ile tahmin performanı değişiklik gösterir. Burada bizim lojistik regresyon ile aldığımız performans %70'dir. ")
    st.markdown("2-) Karar Ağaçları: Karar ağaçları yöntemi, sınıflandırma problemlerinde yaygın olarak tercih edilen bir karar verme algoritmasıdır. C4.5, C5.0 ve CART olmak üzere pek çok versiyonu bulunur. Verimiz üzerinde karar ağacını default değerler ile oluşturduğumda %65'lik bir tahmin performansı elde ettim. Özellikle bir önceki yaklaşık olan Lojistik Regresyon ile kıyasladığımızda iyi bir başarı yüzdesi olarak kabul edemeyiz. Bu nedenle çağraz ızgara arama yöntemi olan GridSearchSV yöntemini kullanarak en uygun hiperparametreleri tespit ettim. Burada ise dikkat etmemiz gereken en önemli nokta max_depth genişliğidir. Bu hiperparametre bize ağaç genişliğini ifade eder. Eğer ağaç genişliğimiz fazla olursa (örneğin 30) algoritma çok iyi bir öğrenme gerçekleştirir. Ancak bu öğrenmenin yeni değerler eklendiğinde oldukça yersiz olduğu anlaşılacaktır. Çünkü ağaç genişliği arttıkça algoritma eğitim veri setinde her şeyi öğrenmek isteyeceği için ezberleme (genelleme) yapar. Bu nedenle ağaç genişliği bir için önemli bir hiperparametredir. Karar Ağaçları algoritmasında aldığım tahmin performansı ise %80'dir.")
    st.markdown("3-) Support Vector Machine (Destek Vektör Makinesi): Support Vector Machines (Destek Vektör Makineleri) sağlam bir istatistiksel yaklaşım sunar. Bu nedenle SVM üzerinden kernel (çekirdek hilesi) yapabilme imkanı mümkündür. ")
    st.markdown("4-) Gradient Boosting: Gradient Boosting, ada boosting yaklaşımından esinlenen ve yine ağaca dayalı bir algoritma türüdür. Bu algoritmada amaç, zayıf tahmin edici karar ağaçlarının bir araya getirilerek yüksek tahmin edici bir model oluşturmaktır. Genellemeye yapmak her ne kadar doğru olmasa da yapılan testlere göre genelde rastgele ormanlar algoritmasından daha güçlü sonuçlar üretir. Çünkü rastgele ormanlar algoritmasının yaklaşımında rastgele özellikler ve rastgele gözlemlerden ağaçlar üreterek bu ağaçları bir araya getirerek anlamlı bir karar oluşturmaktır. Ancak Gradient Boosting algoritmasının yaklaşımında her ağaç önceden eğitilmiş bir ağacın düzeltilmiş versiyonu olduğundan tahmin performansı çok daha yüksektir. Hacettepe Üniversitesi öğrencilerinin kitap okuma alışkanlıkları üzerinden yapılan analize göre de ROC-AUC skoruna göre Gradient Boosting yaklaşımında default değerler ile %81, optimize edilmiş hiperparametreler ile de %85,9'luk bir başarı elde edilmiştir. ")
    st.markdown("5-) Random Forest: Random Forest (rastgele ormanlar) algoritmasının yaklaşımına bakıldığında %81'lik başarı elde edildiği görülmektedir. Algoritmaların her biri kendine göre bir özgünlük taşıdığından algoritmaları kıyaslamak doğru değildir. Her ne kadar Gradient Boosting bu alanda başarılı olarak görülse de, Random Forest algoritması özelliklerin önemi hakkında bize oldukça önemli bilgiler vermektedir. Burada her ne kadar analizin istatistiksel boyutunun üzerinde çok durulmasa da, modele dahil olan özellikler içerisinde öğrencilerin akademik not ortalamasının öneminin en yüksek olduğu görülmektedir. Yani buradan Hacettepe Üniversitesi öğrencilerinin kitap okuma alışkanlıklarını genel akamik not ortalamasının arttırıcı yönde bir etkisi olduğundan bahsedilebilir.")
    st.markdown("Son olarak; verilere ait detaylı istatistiksel sonuçlara ve Lojistik Regresyon analizine ulaşmak için https://github.com/seherkuutlu/Lojistik-Regresyon-Analizi adresini ziyaret edebilirsiniz.  ")
