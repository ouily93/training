print("Hello")


from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np



                 #### couches cachées à tester
hidden_layers_configs = [
    [128],
    [128, 64],
    [128, 96, 64],
    [128, 96, 64, 48],
    [128, 96, 64, 48, 32],
    [128, 96, 64, 48, 32, 24],
    [128, 96, 64, 48, 32, 24, 16]
]

results = []  #### les résultats de chaque configuration


for config in hidden_layers_configs:
    print(f"Testing configuration: {config}")
    
    # Définir le modèle
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.3))
    
    # faire une boucle sur  les couches cachées dynamiquement
    for units in config:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(0.3))
    
    # Ajouter la couche de sortie
    model.add(Dense(y_train_encoded.shape[1], activation='softmax'))
    
    # Compiler le modèle
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
 # X_train, X_test, y_train, y_test  
    # Entrainement du modèle le modèle
    history = model.fit(X_train, y_train_encoded,
                       validation_data=(X_val, y_val_encoded),
                        epochs=20, 
                       
                    batch_size=32 )
    
    # Évaluation du modèle
    val_loss, val_accuracy = model.evaluate(X_test, y_test_encoded, verbose=0)
    print(f"Validation accuracy for config {config}: {val_accuracy}")
    
    # Stocker les résultats
    results.append((config, val_accuracy))

#  meilleure configuration
best_config, best_score = max(results, key=lambda x: x[1])
print(f"Best configuration: {best_config} with accuracy: {best_score}")
