import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

# Load pretrained model
model=keras.applications.resnet50.ResNet50(weights="imagenet")

def run_OneImage(imagepath):
    # Load images
    i=keras.preprocessing.image.load_img(imagepath)

    # Convert image files to vectors
    # Each entry is between 0 and 255 (RGB)
    inparr1=keras.preprocessing.image.img_to_array(i)


    # Apparently images cannot be too big for np.array()
    # Resize to resnet50's requirement
    inparr1=tf.image.resize(inparr1,[224,224])

    images=np.array([inparr1])

    # Show resized image
    # keras.preprocessing.image.array_to_img(images[0]).show()

    # Default preprocessing for resnet50
    inputs=keras.applications.resnet50.preprocess_input(images)

    # Predict
    Y_proba=model.predict(inputs)

    # Top k predictions
    top_k=keras.applications.resnet50.decode_predictions(Y_proba,top=5)

    # Output top k possible classes
    # output(images,top_k)
    
    IsMasked(top_k)


def IsMasked(top_k):
    for i in top_k[0]:
        if i[1]=="mask":
            print("True  {:.2f}%".format(i[2]*100))
            return
    print("False")
    return

def output(images,top_k):
    for image_index in range(len(images)):
        print("image {}".format(image_index))
        for class_id,name,y_proba in top_k[image_index]:
            print("{} - {:12s} {:.2f}%".format(class_id,name,y_proba*100))
            print()

run_OneImage("testdata/20210103_141215.jpg")
run_OneImage("testdata/0000416_25869-halyard-health-blue-procedure-mask.jpg")
run_OneImage("testdata/20210103_172442.jpg")
run_OneImage("testdata/mask.jpg")
run_OneImage("testdata/Most-Beautiful-Sunset-Pictures-16.jpg")
run_OneImage("testdata/Venice-Italy-worlds-most-beautiful-city.jpg")

