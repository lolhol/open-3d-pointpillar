import tensorflow as tf

class TrainingPipeline:
    def __init__(self, model, data_loader, max_epochs=50, verbose=False):
        self.model = model
        self.data_loader = data_loader
        self.max_epochs = max_epochs
        self.verbose = verbose
    
    def train_all(self, optimizer = tf.keras.optimizers.Adam(), max_epochs=None):
        epochs = self.max_epochs if max_epochs is None else max_epochs

        for epoch in range(epochs):
            total_loss = 0
            
            for i in range(len(self.data_loader)):
                data = self.data_loader[i]
                with tf.GradientTape() as tape:
                    output = self.model(data['point'], training=True)
                    loss = self.compute_loss(output, data['label'])

                gradients = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                total_loss += loss

            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss}")

    def compute_loss(self, predictions, labels):
        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, predictions))