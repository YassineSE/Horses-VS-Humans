import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np


def modelEvaluation(model, test_data_directory):
    # Get the Validation Data
    validation_dir = test_data_directory
    validation_datagen = ImageDataGenerator(rescale=1/255)
    validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(300, 300),
    class_mode='binary'
    )
        
    # Evaluate the Model with the Validation Data (Generator)
    evaluation = model.evaluate(validation_generator)

    return evaluation        


def plotTrainingData(callback_file):
    log_data = pd.read_csv(callback_file, sep=',', engine='python')
    plt.plot(log_data["epoch"], log_data["loss"], label = "Loss")
    plt.plot(log_data["epoch"], log_data["accuracy"], label="Model Accuracy")
    plt.legend()
    plt.show()

def realdataTest(model, directory):
    test_images = glob.glob(directory + "/*")
    print("Testing the model on these images: \n")

    accuracy = 0.0

    for i in test_images:
        img = image.load_img(i, target_size=(300,300))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)/255.
        classes = model.predict(x)
        pred = "human" if classes[0] > 0.5 else "horse"

        test = "human" if "human" in str(i) else "horse"

        print(f"{i}: Is: {test}, Predicted: {pred}\n")

        if pred==test: accuracy+=1.0

    print(f"Accuracy on the test images: {np.round(accuracy*100/len(test_images),2)}%")



if __name__ == "__main__":

    test_directory = "horse-or-human/validation"

    # Evaluating and Testing the first model
    model_1 = tf.keras.models.load_model("model_1.h5")
    evaluation_1 = modelEvaluation(model_1, test_directory)
    print(evaluation_1)
    plotTrainingData('training.log')
    realdataTest(model_1, "test_images")
    # We can see that the first model doesn't perform very well on the test images

    
