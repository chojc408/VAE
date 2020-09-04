import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers  import Input, Conv2D, Flatten, Dense, Lambda
from tensorflow.keras.layers  import Reshape, Conv2DTranspose, Layer
from tensorflow.keras.models  import Model, load_model
from tensorflow.keras         import backend as K
from tensorflow.keras.utils   import plot_model
from tensorflow.keras.metrics import mse, binary_crossentropy

def get_mnist_dataset(file_name="mnist.npz"):
    # returns normalized images
    mnist = np.load(file_name, allow_pickle=True)
    # print(mnist.files)
    X_train = mnist['x_train']
    X_test  = mnist['x_test']
    y_train = mnist['y_train']
    y_test  = mnist['y_test']
    # image reshape to (28, 28, 1)
    image_size = X_train.shape[1]
    X_train = np.reshape(X_train, [-1, image_size, image_size, 1])
    X_test  = np.reshape(X_test,  [-1, image_size, image_size, 1])
    # pixel rescaling
    X_train = X_train.astype('float')/255
    X_test  = X_test.astype('float')/255
    return X_train, X_test, y_train, y_test

def view_first_mnist_images_of_digits(X, y):
    # view first images of digits
    fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
    ax = ax.flatten()
    for i in range(10):
        img = X[y==i][0].reshape(28,28)
        ax[i].imshow(img, cmap='Greys')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()

def view_first_10_mnist_image_of_digit(X, y, digit):
    # view first images for digits
    fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
    ax = ax.flatten()
    for i in range(10):
        img = X[y==digit][i].reshape(28,28)
        ax[i].imshow(img, cmap='Greys')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()

def view(X_in, X_out):
    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
    ax = ax.flatten()
    ax[0].imshow(X_in[0].reshape(28,28), cmap='Greys')
    ax[1].imshow(X_out[0].reshape(28,28), cmap='Greys')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()

# === Dataset
X_train, X_test, y_train, y_test = get_mnist_dataset(file_name="mnist.npz") 
X_train = X_train[:1000]

# === Network parameters ======
batch_size   = 128
epochs       = 100
entry_shape  = (28, 28, 1)
# shape of data point (e.g., shape of each image)
# -----------------------------
n_channels   = entry_shape[-1]       # : 1
original_dim = np.prod(entry_shape)  # : 28 * 28 * 1 = 784
# -----------------------------
intermed_dim = 64 # Not needed
latent_dim   = 16
conv_filters = [32, 64]
kernel_size  = 3
strides      = 2
# -----------------------------

# === Encoder part ============
x_in  = Input(shape=entry_shape)
x_enc = x_in
x_enc = Conv2D(filters=conv_filters[0], kernel_size=kernel_size,
               strides=strides, padding='same',
               activation='relu')(x_enc)
x_enc = Conv2D(filters=conv_filters[1], kernel_size=kernel_size,
               strides=strides, padding='same',
               activation='relu')(x_enc)
conv_output_shape = K.int_shape(x_enc)
# (None,7, 7, 64): Decoder needs this info
"""
Consider He initialization: kernel_initializer='he_uniform' with 'relu'
instead of default kernel_initializer='glorot_uniform' for large dataset.
Also consider BatchNormalization()(x_enc).
"""
# --- latent space
x_enc = Flatten()(x_enc)
flat_output_shape = K.int_shape(x_enc)
# (None, 3136) = (None, 7*7*64): Decoder needs this info
z_mean    = Dense(latent_dim, activation=None)(x_enc)
z_log_var = Dense(latent_dim, activation=None)(x_enc)
def reparameterization(args):
    z_mean, z_log_var = args
    batch_size = K.shape(z_mean)[0]
    dim        = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch_size, dim), mean=0, stddev=1) ##
    return z_mean + K.exp(0.5*z_log_var)*epsilon
z = Lambda(reparameterization)([z_mean,z_log_var])
Encoder = Model(x_in, [z_mean, z_log_var, z], name='encoder')
Encoder.summary()
# === Decoder part ============
z_in  = Input(shape=latent_dim)
x_dec = z_in
x_dec = Dense(flat_output_shape[1], activation=None)(x_dec) # (7*7*64=3136)
x_dec = Reshape((conv_output_shape[1:]))(x_dec)
x_dec = Conv2DTranspose(filters=conv_filters[1], kernel_size=kernel_size,
                        strides=strides,padding='same',
                        activation='relu')(x_dec)
x_dec = Conv2DTranspose(filters=conv_filters[0], kernel_size=kernel_size,
                        strides=strides, padding='same',
                        activation='relu')(x_dec)
# --- to original shape ------
x_out = Conv2DTranspose(filters=n_channels, kernel_size=kernel_size,
                        strides=1, padding='same',
                        activation='sigmoid')(x_dec)
# --- build model
Decoder = Model(z_in, x_out, name='decoder')
Decoder.summary()
# === Variational Inference Layer (custom)
class VariationalInferenceLayer(Layer):
    def vae_loss(self, x_in, x_out):
        # --- reconstruction loss
        x_in_flat  = K.flatten(x_in)
        x_out_flat = K.flatten(x_out)
        recon_loss = binary_crossentropy(x_in_flat, x_out_flat)
        #recon_loss = mse(x_in_flat, x_out_flat)
        recon_loss = recon_loss * original_dim
        # --- KL loss
        KL_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        KL_loss = (-0.5)*K.sum(KL_loss, axis=-1)
        KL_loss = KL_loss
        # --- total loss
        return K.mean(recon_loss + KL_loss)
    def call(self, inputs):
        x_in  = inputs[0]
        x_out = inputs[1]
        loss  = self.vae_loss(x_in, x_out)
        self.add_loss(loss, inputs=inputs)
        return x_out # dummy output    
# === Autoencoder ============
answer = int(input("fit and save[1] or load[2]:"))
if answer == 1:
    # --- build VAE
    [z_mean, z_log_var, z] = Encoder(x_in)
    x_out = Decoder(z)
    dummy = VariationalInferenceLayer(name='loss_layer')([x_in, x_out])
    VAE   = Model(x_in, dummy, name='autoencoder')
    VAE.compile(optimizer='adam', loss=None, experimental_run_tf_function = False)
    VAE.summary()
    # --- train VAE and save 'Encoder and Decoder separately'
    VAE.fit(x=X_train, y=None, batch_size=batch_size, epochs=epochs,
            validation_split=0.2, shuffle=True, verbose=2)
    Encoder.save("VAE_encoder.h5")
    Decoder.save("VAE_decoder.h5")
elif answer == 2:
    print("loading saved models (encoder and decoder)...")
    Encoder = load_model("VAE_encoder.h5"); Encoder.summary()
    Decoder = load_model("VAE_decoder.h5"); Decoder.summary()
    # --- build VAE
    [z_mean, z_log_var, z] = Encoder(x_in)
    x_out = Decoder(z)
    dummy = VariationalInferenceLayer(name='loss_layer')([x_in, x_out])
    VAE   = Model(x_in, dummy, name='autoencoder'); VAE.summary()

# === Reconstruction Example
for i in range(5):
    x_ori = X_test[i:i+1]
    x_rec = VAE.predict(x_ori)
    view(x_ori, x_rec)

# === Check normality of latent space
import seaborn as sns
X_train_ori = X_train
[z_mean, z_log_var, z] = Encoder(X_train_ori)
z = np.array(z)
for dim in range(latent_dim):
    values_in_dim = z[:, dim]
    print("dim", dim, ":", np.mean(values_in_dim), np.std(values_in_dim))
    sns.kdeplot(values_in_dim, shade=True)
    plt.show()

# === Single digit image

for i in range(5):
    z = np.random.normal(0, 1, size=latent_dim)
    z = z.reshape(1,latent_dim)
    x_rec = Decoder.predict(z)
    view(x_rec, x_rec)

"""
# === Draw digit images in 2D latent space
from scipy.stats import norm
n = 15 #(15-by-15 digit grid)
digit_size = 28
figure = np.zeros((digit_size*n, digit_size*n))
# --- sampling z
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
        x_recons = Decoder.predict(z_sample)
        digit = x_recons[0].reshape(digit_size, digit_size)
        figure[i*digit_size:(i+1)*digit_size,
               j*digit_size:(j+1)*digit_size] = digit
plt.figure(figsize=(10,10))
plt.imshow(figure, cmap='Greys_r')
plt.show()
"""
