import tensorflow as tf
import os

def trainer(model, train_data, test_data, cktp_name, epochs):
    model.compile(optimizer=tf.optimizers.Adam(0.0001),
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.metrics.sparse_categorical_accuracy],)

    checkpoint_filepath = os.path.join('cktp', f'{cktp_name}.h5')

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_sparse_categorical_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
        )

    history = model.fit(train_data[0], train_data[1],
        batch_size=32,
        epochs=epochs,
        validation_data=(test_data[0], test_data[1]),
        verbose=1,
        callbacks=[model_checkpoint_callback],
        )
    print('MAX ACC : ', max(history.history['val_sparse_categorical_accuracy']))

    model.load_weights(checkpoint_filepath)
    print("Evaluate on test data")
    results = model.evaluate(test_data[0], test_data[1], batch_size=64)
    print("test loss, test acc:", results)

    return model

def eval(model, test_data):
    model.compile(optimizer=tf.optimizers.Adam(0.0001),
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.metrics.sparse_categorical_accuracy],)
    # checkpoint_filepath = os.path.join('cktp', f'{cktp_name}.h5')
    # model.load_weights(checkpoint_filepath)
    print("Evaluate on test data")
    results = model.evaluate(test_data[0], test_data[1], batch_size=64, verbose=1,)
    print("test loss, test acc:", results)


    
