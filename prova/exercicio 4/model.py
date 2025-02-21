import os 
from tensorflow.keras import models, layers, activations, optimizers, utils, losses, initializers, metrics, callbacks

model_path = '/kaggle/input/bengali-digits/bengali_digits/'
epochs = 20
batch_size = 50
patience = 11
learning_rate = 0.001

model = models.Sequential([
    layers.Resizing(50, 50),
    layers.Rescaling(1.0/255),
    layers.RandomRotation((-0.2, 0.2)),
    layers.Conv2D(16, (3,3), # 16 @ 48 x 48
        activation = 'relu',
        kernel_initializer = initializers.RandomNormal()
    ),
    layers.MaxPooling2D((2,2)), # 16 @ 24 x 24
    layers.Conv2D(32, (4,4), # 512 @ 21 x 21
        activation  = 'relu',
        kernel_initializer = initializers.RandomNormal()
    ),  
    layers.MaxPooling2D((3,3)), # 512 @ 7 x 7
    layers.Conv2D(64, (4,4), # 32768 @ 4 x 4
        activation  = 'relu',
        kernel_initializer = initializers.RandomNormal()
    ),  
    layers.MaxPooling2D((2,2)), # 32768 @ 2 x 2
    layers.Flatten(),
    layers.Dense(128,
        activation = 'relu',
        kernel_initializer = initializers.RandomNormal()
    ),
    layers.Dense(64,
        activation = 'relu',
        kernel_initializer = initializers.RandomNormal()
    ),
    layers.Dense(64,
         activation = 'relu',
         kernel_initializer = initializers.RandomNormal()
    ),
    layers.Dense(10,
         activation = 'sigmoid',
         kernel_initializer = initializers.RandomNormal()
    )
])

model.compile(
    optimizer = optimizers.Adam(
        learning_rate = learning_rate
    ),
    loss = losses.SparseCategoricalCrossentropy(),
    metrics = [ metrics.SparseCategoricalAccuracy() ]
)

train = utils.image_dataset_from_directory(
    model_path,
    validation_split = 0.2,
    subset = "training",
    seed = 123,
    shuffle = True,
    image_size = (57, 64),
    batch_size = batch_size
)

test = utils.image_dataset_from_directory(
    model_path,
    validation_split = 0.2,
    subset = "validation",
    seed = 123,
    shuffle = True,
    image_size = (57, 64),
    batch_size = batch_size
)

model.fit(train,
    epochs = epochs,
    validation_data = test,
    callbacks= [
        callbacks.EarlyStopping(
            monitor = 'val_loss',
            patience = patience,
            verbose = 1
        ) 
    ]
)