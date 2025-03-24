import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# 1. Veri Augmentasyonu ve Veri Yükleme
train_datagen = ImageDataGenerator(
    rescale=1./255,        # Piksel değerlerini [0, 1] aralığına çekme
    shear_range=0.2,       # Eğilme (shearing)
    zoom_range=0.2,        # Zoom
    horizontal_flip=True   # Yatay çevirme
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Eğitim verilerini yükleyin
train_set = train_datagen.flow_from_directory(
    'C:/Users/celik/OneDrive/Masaüstü/denseNet/train',  # Gerçek dosya yolunu yazın
    target_size=(64, 64),  # Resimlerin yeniden boyutlandırılması (örneğin 64x64)
    batch_size=32,         # Her seferinde kaç resim işlenecek
    class_mode='binary'    # 2 sınıf olduğu için binary sınıf modu
)

# Doğrulama verilerini yükleyin
validation_set = validation_datagen.flow_from_directory(
    'C:/Users/celik/OneDrive/Masaüstü/denseNet/validation',  # Gerçek dosya yolunu yazın
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# Test verilerini yükleyin
test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory(
    'C:/Users/celik/OneDrive/Masaüstü/denseNet/test',  # Gerçek dosya yolunu yazın
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# 2. Modeli Oluşturma
model = Sequential()

# Conv2D: Konvolüsyonel katman (Resimlerden özellik çıkarma)
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # 2D veriyi 1D'ye çeviriyoruz
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Aşırı öğrenmeyi engellemek için dropout
model.add(Dense(1, activation='sigmoid'))  # Binary sınıflama için sigmoid

# Modeli derleme
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# 3. Modeli Eğitme
history = model.fit(
    train_set,
    steps_per_epoch=train_set.samples // train_set.batch_size,  # Her batch için kaç adım yapılacağı
    epochs=10,  # Kaç epoch eğitilecek
    validation_data=validation_set,
    validation_steps=validation_set.samples // validation_set.batch_size  # Doğrulama için adım sayısı
)

# 4. Test Seti ile Modeli Değerlendirme
test_loss, test_acc = model.evaluate(test_set, steps=test_set.samples // test_set.batch_size)
print(f'Test accuracy: {test_acc}')

# 5. Modeli Kaydetme
model.save('C:/Users/celik/OneDrive/Masaüstü/denseNet/model/inme_model.h5')  # Modeli kaydedin

# 6. Modeli Yükleme (Test Aşamasında)
# model = tf.keras.models.load_model('C:/Users/celik/OneDrive/Masaüstü/denseNet/model/inme_model.h5')
