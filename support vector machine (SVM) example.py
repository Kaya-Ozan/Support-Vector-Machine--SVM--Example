import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource

# Veri kümesini yükleyin (ör. iris veri kümesi)
iris = datasets.load_iris()
X = iris.data[:, :2]  # Sadece ilk iki özellik alınarak görselleştirme kolaylaştırılır
y = iris.target

# Veriyi eğitim ve test setlerine bölmek
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=35)

# Veriyi ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVM sınıflandırıcısını oluşturma
svm = SVC(kernel='linear', C=1.0, random_state=35)

# Modeli eğitme
svm.fit(X_train_scaled, y_train)

# Tahminlerde bulunma
y_pred = svm.predict(X_test_scaled)

# Doğruluk skorunu hesaplama
accuracy = accuracy_score(y_test, y_pred)
print(f"Modelin doğruluk skoru: {accuracy:.2f}")

# Verileri görselleştirme
output_file("denem.html")

colors = ['red', 'blue', 'green']
color_map = {0: 'red', 1: 'blue', 2: 'green'}

# Eğitim verileri için renk kodlarını hazırlayın
train_colors = [color_map[label] for label in y_train]
# Test verileri için renk kodlarını hazırlayın
test_colors = [color_map[label] for label in y_pred]

# Sınırları belirlemek için minimum ve maksimum değerleri alın
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1

# Karar sınırlarını çizmek için bir ızgara oluşturun
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Izgara üzerindeki tahminleri yapın
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Bokeh veri kaynağını oluşturun
source_train = ColumnDataSource(data=dict(x=X_train_scaled[:, 0], y=X_train_scaled[:, 1], colors=train_colors))
source_test = ColumnDataSource(data=dict(x=X_test_scaled[:, 0], y=X_test_scaled[:, 1], colors=test_colors))
                                      
                                        
 # Bokeh grafiğini oluşturun
p = figure(title="SVM Sınıflandırması", x_axis_label="Özellik 1", y_axis_label="Özellik 2")

# Karar sınırlarını çiz
p.image(image=[Z], x=x_min, y=y_min, dw=(x_max-x_min), dh=(y_max-y_min), palette="Spectral11", level="image")

# Eğitim verilerini çiz
p.circle('x', 'y', source=source_train, color='colors', legend_label="Eğitim verisi", size=10)

# Test verilerini çiz
p.square('x', 'y', source=source_test, color='colors', legend_label="Test verisi", size=10)

# Efsaneyi ayarlayın ve grafiği gösterin
p.legend.location = "top_left"
p.legend.click_policy = "hide"
show(p)