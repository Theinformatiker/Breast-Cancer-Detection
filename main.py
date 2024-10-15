# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 11:11:04 2024

@author: vkkor
"""

# Importieren der benötigten Bibliotheken
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2


# Zufallskeim für Reproduzierbarkeit setzen
tf.random.set_seed(42)
np.random.seed(42)

# Laden und Vorverarbeiten der Daten
df = pd.read_csv("breast-cancer.xls")

# Zielvariable encodieren (M = 1, B = 0)
le = LabelEncoder()
y = le.fit_transform(df['diagnosis'])

# Encodierte Labels in den DataFrame einfügen
df['diagnosis'] = y

# Nicht benötigte Spalten entfernen
X = df.drop(['id', 'diagnosis'], axis=1)

# Daten skalieren
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Korrelationen berechnen, um wichtige Features zu identifizieren
corr_matrix = df.corr()
corr_target = abs(corr_matrix["diagnosis"])
relevant_features = corr_target[corr_target > 0.5]
relevant_features = relevant_features.drop('diagnosis')
selected_features = relevant_features.index.tolist()
print("Ausgewählte Features:", selected_features)

# Datensatz mit ausgewählten Features erstellen
X_selected = df[selected_features]
X_scaled_selected = scaler.fit_transform(X_selected)

# Daten in Trainings- und Testdaten aufteilen
X_train, X_test, y_train, y_test = train_test_split(X_scaled_selected, y, test_size=0.2, random_state=42, stratify=y)

# Datensatz mit SMOTE balancieren
sm = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)

# Klassen-Gewichte berechnen (optional, da Daten balanciert sind)
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_resampled), y=y_train_resampled)
class_weights = dict(enumerate(class_weights))

# Modell definieren (vereinfachte Architektur)
model = Sequential([
    Dense(512, input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dropout(0.3),
    
    Dense(256, kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dropout(0.3),
    
    Dense(128, kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dropout(0.3),
    
    Dense(64, kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dropout(0.3),
    
    Dense(32, kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dropout(0.3),
    
    Dense(1, activation='sigmoid')  # For binary classification
])

# Modell kompilieren mit angepasstem Optimierer und Lernrate
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Callbacks für frühes Stoppen und Lernratenreduzierung
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)

# Modell trainieren
history = model.fit(X_train_resampled, y_train_resampled,
                    validation_split=0.2,
                    epochs=100,
                    batch_size=32,
                    callbacks=[early_stopping, reduce_lr],
                    verbose=1)

# Modell auf Testdaten evaluieren
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy}")

# Vorhersagen auf Testdaten treffen
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int)

# Zusätzliche Metriken berechnen
print("Klassifikationsbericht:")
print(classification_report(y_test, y_pred))

# ROC AUC Score berechnen
auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC Score: {auc}")

"""
# ROC-Kurve plotten
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC Kurve (AUC = {auc:.2f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Kurve')
plt.legend(loc='lower right')
plt.show()
"""
# Konfusionsmatrix erstellen
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Vorhergesagte Klasse')
plt.ylabel('Tatsächliche Klasse')
plt.title('Konfusionsmatrix')
plt.show()

# Trainingshistorie plotten
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Modellgenauigkeit')
plt.xlabel('Epoche')
plt.ylabel('Genauigkeit')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Modellverlust')
plt.xlabel('Epoche')
plt.ylabel('Verlust')
plt.legend()

plt.tight_layout()
plt.show()
