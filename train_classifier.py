import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Periksa dan normalkan bentuk data
# Tentukan panjang maksimum dari elemen-elemen data jika panjangnya bervariasi
max_length = max(len(d) for d in data_dict['data'])

# Sesuaikan data menjadi array 2D dengan padding jika panjang bervariasi
# atau ratakan elemen jika sudah dalam bentuk konsisten
data = np.array([np.pad(d, (0, max_length - len(d)), 'constant') if len(d) < max_length else d 
                 for d in data_dict['data']])
labels = np.asarray(data_dict['labels'])

# Bagi data menjadi data latih dan uji
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Inisialisasi model dan pelatihan
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Prediksi dan evaluasi
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly!'.format(score * 100))

# Simpan model ke file
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
