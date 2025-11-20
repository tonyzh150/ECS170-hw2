import numpy as np
import mnist_reader
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models


X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

X_train = X_train / X_train.max()
X_test = X_test / X_test.max()


# last 12000 samples as validation set
X_val = X_train[-12000:]
y_val = y_train[-12000:]
# remove them from training set
X_train = X_train[:-12000]
y_train = y_train[:-12000]

# reshape for CNN input
X_train = X_train.reshape(-1, 28, 28, 1)
X_val   = X_val.reshape(-1, 28, 28, 1)
X_test  = X_test.reshape(-1, 28, 28, 1)

model = models.Sequential([
  # 2D convolutional layer, 28 filters, 3x3 window size, ReLU activation
  layers.Conv2D(28, (3,3), activation = 'relu', input_shape = (28,28,1)),
  # 2x2 max pooling
  layers.MaxPooling2D(pool_size=(2,2)),
  # 2D convolutional layer, 56 filters, 3x3 window size, ReLU activation
  layers.Conv2D(56, (3,3), activation = 'relu'),
  layers.Flatten(),
  # fully-connected layer, 56 nodes, ReLu activation
  layers.Dense(56, activation='relu'), 
  # fully-connected layer, 10 nodes, softmax activation
  layers.Dense(10, activation='softmax') 
])

model.summary()

# total number of trainable parameters (from gpt)
trainable_params = np.sum(
  [tf.keras.backend.count_params(w) for w in model.trainable_weights]
)
print("Number of trainable parameters:", trainable_params)


# Adam optimizer, 32 observations per batch, and sparse categorical cross-entropy loss. Train for 10 epochs.
model.compile(
  optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy']
)
eachEpo = model.fit(
  X_train, y_train,
  epochs=10,
  batch_size=32,
  validation_data=(X_val, y_val)
)

plt.figure()
plt.plot(eachEpo.history['accuracy'], label='train accuracy')
plt.plot(eachEpo.history['val_accuracy'], label='val accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('imgs/training_validation_accuracy.png', dpi=150)
plt.show()
# check submission.md for the saved plot

# Evaluate accuracy on the test set.
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print("Test accuracy:", test_acc)


# Show an example from the test set for each class where the model misclassifies.
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

class_names = [
  "T-shirt_top", "Trouser", "Pullover", "Dress", "Coat",
  "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

misclassfied = np.where(y_pred != y_test)[0]
print(f"Total misclassified samples: {len(misclassfied)}")

chosen = {}
for idx in misclassfied:
  true_label = int(y_test[idx])
  if true_label not in chosen:
    chosen[true_label] = idx
  if len(chosen) == 10:
    break
    
for i, class_id in enumerate(sorted(chosen.keys())):
  index = chosen[class_id]
  img = X_test[index].squeeze()
  plt.figure()
  plt.imshow(img, cmap='gray')
  plt.axis('off')
  plt.title(
    f"True: {class_names[class_id]}\nPredicted: {class_names[y_pred[index]]}"
  )
  plt.savefig(f'imgs/misclassfied_{class_names[class_id]}.png', dpi=150)
  plt.show()
  # check submission.md for the saved plots
