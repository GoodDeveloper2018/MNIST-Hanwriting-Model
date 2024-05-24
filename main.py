import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load MNIST data
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

# Normalize pixel values
train_images = train_images / 255.0
test_images = test_images / 255.0

# Model setup
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu, name='dense_1'),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5)

# Extract features from the intermediate layer
intermediate_layer_model = keras.Model(inputs=model.input, outputs=model.get_layer('dense_1').output)
intermediate_train_features = intermediate_layer_model.predict(train_images)
intermediate_test_features = intermediate_layer_model.predict(test_images)

# Reshape the features for clustering
reshaped_train_features = intermediate_train_features.reshape((intermediate_train_features.shape[0], -1))
reshaped_test_features = intermediate_test_features.reshape((intermediate_test_features.shape[0], -1))

# Perform K-Means clustering
num_clusters = 10
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(reshaped_train_features)

# Visualize the clusters using representative images
fig, axes = plt.subplots(1, num_clusters, figsize=(15, 5))

for i in range(num_clusters):
    cluster_indices = np.where(kmeans.labels_ == i)[0]  # Get indices of samples in the current cluster
    representative_image_idx = cluster_indices[0]  # Choose the first image as representative

    # Reshape and plot the representative image
    representative_image = train_images[representative_image_idx].reshape((28, 28))
    axes[i].imshow(representative_image, cmap='gray')
    axes[i].set_title(f'Cluster {i + 1}')

plt.suptitle('K-Means Clustering with Representative Images')
plt.tight_layout()
plt.show()

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
test_acc_percent = (test_acc * 100)
print(f'Test accuracy: {test_acc} or {round(test_acc_percent)}%')

# Analyze cluster centroids
print("\nCluster centroid analysis:")
for i, centroid in enumerate(kmeans.cluster_centers_):
    centroid_reshaped = centroid.reshape((128,))
    print(f"Cluster {i + 1} Centroid (first 10 features): {centroid_reshaped[:10]}")

# Assuming that the clusters could relate to handwriting anomalies
# Further analysis could be implemented here based on domain knowledge of handwriting features
for i in range(num_clusters):
    cluster_indices = np.where(kmeans.labels_ == i)[0]
    cluster_images = train_images[cluster_indices]
    # Example: Calculate mean image for the cluster
    mean_image = np.mean(cluster_images, axis=0)

    plt.figure()
    plt.imshow(mean_image, cmap='gray')
    plt.title(f'Mean Image for Cluster {i + 1}')
    plt.show()

    # Further statistical analysis or feature extraction could be added here
    # Example: Variance of the cluster images
    variance_image = np.var(cluster_images, axis=0)

    plt.figure()
    plt.imshow(variance_image, cmap='hot')
    plt.title(f'Variance Image for Cluster {i + 1}')
    plt.show()

    # Additional analysis based on domain-specific features can be added here
    print(f'Cluster {i + 1} might represent a specific pattern in handwriting features.\n')
