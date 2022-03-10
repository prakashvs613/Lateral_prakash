# -*- coding: utf-8 -*-
# Description: This code is used to split a dataset into Training and Testing dataset
# Date: 02/01/2020, Haile Woldesellasse
"""
Created on Tue Jun  2 10:53:05 2020

@author: haile01
"""

# Clear variables
from IPython import get_ipython
get_ipython().magic('reset -sf')


# Import Libraries
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import keras
from keras import Input, Model
from keras.layers import Dense, LeakyReLU, concatenate
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from math import sqrt
from keras.layers import Activation, Dropout, Flatten, Dense, Input, LeakyReLU
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score
import time

start = time.time()

seed = 1991

# Path to file location
path = 'D:/Lateral spread displacement/'

data = pd.read_csv(path+"YoudHansenBartlett_Transformed.csv")
data = data.drop(['EARTHQUAKE'], axis=1)  
data = data.drop(['b0','log Wff'], axis=1) 


# Depending on the equation, drop log Sgs and log Wff (ground slope and free face)

X= data.iloc[:,1:9].values
y = data.iloc[:,0].values




# Split the data into training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=seed) 


# Prepare the training and testing dataset to be saved in excel
Training = np.column_stack((y_train,X_train))
Testing = np.column_stack((y_test,X_test))


Training = pd.DataFrame(data=Training, columns=data.columns)
Testing = pd.DataFrame(data=Testing, columns=data.columns)


# Save the training and validation datasets into excel files 
df1 = Training
df1 = pd.DataFrame(data=df1)
df1.to_csv(path+'YoudHansenBartlett_Training.csv',index=False)

df2 = Testing
df2 = pd.DataFrame(data=df2)
df2.to_csv(path+'YoudHansenBartlett_Testing.csv',index=False)


# Assign the training and testing datasets

dataTraining = Training
dataTesting = Testing



X_train = dataTraining.iloc[:, 1:9].values
y_train = dataTraining.iloc[:,0].values

X_test = dataTesting.iloc[:, 1:9].values
y_test = dataTesting.iloc[:,0].values



x_input_size = X_train.shape[1]
y_input_size = 1
z_input_size = 3
n_samples = 150

 

# Normalize a dataset

#scaler = MinMaxScaler(feature_range=(-1,1))
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# Add the size of x_input_size and y_input_size for the discriminator
# Add the size of x_input_size and z_input_size for the generator

# Declaring empty lists to save the losses for plotting
optimizer_gen = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
optimizer_disc = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
activation= 'relu'
kerner_initializer = keras.initializers.he_normal(seed=seed)
random_normal = keras.initializers.RandomNormal(seed=seed)
random_uniform = keras.initializers.RandomUniform(seed=seed)


# Generator
def generator():
  
    x = Input(shape=(x_input_size,), dtype='float')
    x_output = Dense(100, activation=activation, kernel_initializer=kerner_initializer)(x)

    noise = Input(shape=(z_input_size,))
    noise_output = Dense(100, activation=activation, kernel_initializer=kerner_initializer)(noise)

    concat = concatenate([x_output, noise_output])

    output = Dense(100, activation=activation, kernel_initializer=kerner_initializer)(concat)
    output = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(output)
    output = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(output)
    output = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(output)
    output = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(output)
    output = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(output)
    output = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(output)
    output = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(output)    
    output = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(output)
    output = Dense(y_input_size, activation="linear", kernel_initializer=random_normal)(output)
   
    return Model(inputs=[noise, x], outputs=output)


# Discriminator
def discriminator():

    x = Input(shape=(x_input_size,), dtype='float')
    x_output = Dense(100, activation=activation, kernel_initializer=kerner_initializer)(x)

    label = Input(shape=(y_input_size,))
    label_output = Dense(100, activation=activation, kernel_initializer=kerner_initializer)(label)

    concat = concatenate([x_output, label_output])
    concat = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(concat)
    concat = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(concat)
    concat = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(concat)
    concat = Dense(75, activation=activation, kernel_initializer=kerner_initializer)(concat)
    validity = Dense(1, activation="sigmoid", kernel_initializer=random_uniform)(concat)
        
    return Model(inputs=[x, label], outputs=validity)


# Build and compile the discriminator
discriminator = discriminator()
discriminator.compile(loss=['binary_crossentropy'], optimizer=optimizer_disc, metrics=['accuracy'])


# Build the generator
generator = generator()

# The generator takes noise and the target label as input
# and generates the corresponding digit of that label
noise = Input(shape=(z_input_size,))
x = Input(shape=(x_input_size,))
label = generator([noise, x])

# For the combined model we will only train the generator
discriminator.trainable = False

# The discriminator takes generated image as input and determines validity
# and the label of that image
valid = discriminator([x, label])

# The combined model  (stacked generator and discriminator)
# Trains generator to fool discriminator
combined = Model([noise, x], valid)
combined.compile(loss=['binary_crossentropy'], optimizer=optimizer_gen)

epochs= 500
batch_size = 16
n_eval=10
# train the generator and discriminator

def train(xtrain, ytrain, epochs, batch_size= batch_size, verbose=True):
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        dLossErr = np.zeros([epochs, 1])
        dLossReal = np.zeros([epochs, 1])
        dLossFake = np.zeros([epochs, 1])
        gLossErr = np.zeros([epochs, 1])
        genPred = np.zeros([epochs, 1])
        genReal = np.zeros([epochs, 1])

        for epoch in range(epochs):
            for batch_idx in range(int(xtrain.shape[0] // batch_size)):
                # ---------------------
                #  Train Discriminator
                # ---------------------
                # Select a random half batch of images
                idx = np.random.randint(0, xtrain.shape[0], batch_size)
                x, true_labels = xtrain[idx], ytrain[idx]
                # Sample noise as generator input
                noise = np.random.normal(0, 1, (batch_size, z_input_size))
                # Generate a half batch of new images
                fake_labels = generator.predict([noise, x])
                # Train the discriminator
                d_loss_real = discriminator.train_on_batch([x, true_labels], valid)
                d_loss_fake = discriminator.train_on_batch([x, fake_labels], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Train Generator
                # ---------------------
                # Condition on x
                idx = np.random.randint(0, xtrain.shape[0], batch_size)
                sample = xtrain[idx]
                # Train the generator
                g_loss = combined.train_on_batch([noise, sample], valid)

            dLossErr[epoch] = d_loss[0]
            dLossReal[epoch] = d_loss_real[0]
            dLossFake[epoch] = d_loss_fake[0]
            gLossErr[epoch] = g_loss

            if verbose:
                print(f"Epoch: {epoch} / dLoss: {d_loss[0]} / gLoss: {g_loss}")

            ypred = predict(xtrain)
            genPred[epoch] = np.average(ypred)
            genReal[epoch] = np.average(ytrain)

        return dLossErr, dLossReal, dLossFake, gLossErr, genPred, genReal


def predict(xtest):
    noise = np.random.normal(0, 1, (xtest.shape[0], z_input_size))
    ypred = generator.predict([noise, xtest])
    return ypred


# sample is used to generate instances equal to n_Samples, due to the instability of the model
def sample(xtest, n_samples):
    y_samples_gan = predict(xtest)
    for i in range(n_samples - 1):
        ypred_gan = predict(xtest)
        y_samples_gan = np.hstack([y_samples_gan, ypred_gan])
    median = []
    mean = []
    for j in range(y_samples_gan.shape[0]):
        median.append(np.median(y_samples_gan[j, :]))
        mean.append(np.mean(y_samples_gan[j, :]))

    return np.array(mean).reshape(-1, 1), np.array(median).reshape(-1, 1), y_samples_gan

      
        
def plots(d_loss_err, d_loss_true, d_loss_fake, g_loss_err, g_pred, g_true, fig_dir="", save_fig=False):
    plt.plot(d_loss_err, label="Discriminator Loss")
    plt.plot(d_loss_true, label="Discriminator Loss - True")
    plt.plot(d_loss_fake, label="Discriminator Loss - Fake")
    plt.plot(g_loss_err, label="Generator Loss")
    plt.legend(loc="upper left")
    plt.xlabel("Epoch")
    plt.title("Loss")
    if save_fig:
        plt.savefig(f"{fig_dir}/gan_loss.png")
    plt.show()

    plt.plot(g_pred, label="Average Generator Prediction")
    plt.plot(g_true, label="Average Generator Reality")
    plt.legend(loc="upper left")
    plt.xlabel("Epoch")
    plt.title("Average Prediction")

    if save_fig:
        plt.savefig(f"{fig_dir}/{basename(fig_dir)}gan_ave_pred.png")
    plt.show()        


d_loss_err, d_loss_true, d_loss_fake, g_loss_err, g_pred, g_true = train(X_train_scaled, y_train, 
                                                                              epochs=epochs,
                                                                              batch_size=batch_size)

ypred_gan_test = predict(X_test_scaled)


plots(d_loss_err, d_loss_true, d_loss_fake, g_loss_err, g_pred, g_true, fig_dir="", save_fig=False)

ypred_mean_gan_test, ypred_median_gan_test, ypred_gan_sample_test = sample(X_test_scaled, n_samples)



# The mean is reported over 10 evaluation runs. Mean and std are calculated from these
n_eval_runs = 10
mse_gan_= []
mae_gan_ = []
rmse_mean_gan_original =[] # rmse for the original variable
rmse_median_gan_original =[] # rmse for the original variable
for i in range(n_eval_runs):
    ypred_mean_gan_test_, ypred_median_gan_test_, _ = sample(X_test_scaled, n_samples)
    ypred_mean_test_unscaled=10 ** ypred_mean_gan_test_
    ypred_median_test_unscaled=10 ** ypred_median_gan_test_
    y_test_unscaled=10 ** y_test
    rmse_mean_gan_original.append (((y_test_unscaled - ypred_mean_test_unscaled) ** 2).mean() ** .5)
    rmse_median_gan_original.append (((y_test_unscaled - ypred_median_test_unscaled) ** 2).mean() ** .5)    
    mae_gan_.append(mean_absolute_error(y_test, ypred_median_gan_test_))
    mse_gan_.append(mean_squared_error(y_test, ypred_mean_gan_test_))



# Error calculation
gan_mae_mean = np.mean(np.asarray(mse_gan_))
gan_mae_std = np.std(np.asarray(mse_gan_))

gan_rmse_mean = np.mean(np.asarray(rmse_mean_gan_original))
gan_rmse_mean_std = np.std(np.asarray(rmse_mean_gan_original))

gan_rmse_median = np.mean(np.asarray(rmse_median_gan_original))
gan_rmse_median_std = np.std(np.asarray(rmse_median_gan_original))


print(f"CGAN MAE test: {gan_mae_mean} +- {gan_mae_std}")
print(f"CGAN RMSE(mean) test: {gan_rmse_mean} +- {gan_rmse_mean_std}")
print(f"CGAN RMSE(median) test: {gan_rmse_median} +- {gan_rmse_median_std}")

# Model comparison
plt.subplot(2, 1, 2)
plt.plot(range (0,len(y_test)),y_test,label='Original Data')
plt.plot(range (0,len(y_test)),ypred_mean_gan_test,label='GAN')
plt.title('Model comparison')
plt.ylabel('Score')
plt.xlabel('number of observations')
plt.legend(loc='best')
r_squared2 = r2_score(y_test, ypred_mean_gan_test)
plt.text(10,-0.5, 'R-squared = %0.2f' % r_squared2)
plt.show()        
     

y_test = 10 ** y_test
ypred_mean_gan_test = 10 ** ypred_mean_gan_test

dfOrig= pd.DataFrame(data = y_test, columns=['Observed Displacement (m)'])
dfModel= pd.DataFrame(data = ypred_mean_gan_test, columns=['Predicted Displacement (m)'])

Comp = pd.concat([dfOrig,dfModel], axis=1)
Comp.to_csv(path+'ScatterPlot.csv',index=False)





ypred_gan_train = predict(X_train_scaled)

y_train = 10 ** y_train
ypred_gan_train = 10 ** ypred_gan_train

dfOrig= pd.DataFrame(data = y_train, columns=['Observed Displacement (m)'])
dfModel= pd.DataFrame(data = ypred_gan_train, columns=['Predicted Displacement (m)'])

Comp = pd.concat([dfOrig,dfModel], axis=1)
Comp.to_csv(path+'ScatterPlotT.csv',index=False)


"""
ypred_mean_gan_train, ypred_median_gan_train, ypred_gan_sample_train = sample(X_train_scaled, n_samples)
ypred_train_unscaled=10 ** ypred_mean_gan_train
y_train_unscaled=10 ** y_train
rmse_train = (((y_train_unscaled - ypred_train_unscaled) ** 2).mean() ** .5)


#Scatter Plot
colors = (0,0,0)
area = np.pi*3
plt.scatter(ypred_train_unscaled, y_train_unscaled, s=area, c=colors, alpha=0.5)
xl= np.asarray(list(range(0, 16)))
yl = 2*xl
yu= 0.5*xl
yp = xl
plt.plot(xl, yl,color='k')
plt.plot(xl, yu,color='k')
plt.plot(xl, yp,color='k')
plt.xlim(0, 15)
plt.ylim(0, 15)
plt.title('CGAN')
plt.xlabel('Predicted Displacement (m)')
plt.ylabel('Measured Displacement (m)')
plt.show()
"""