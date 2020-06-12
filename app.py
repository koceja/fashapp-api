# When redeploying, run $ pip freeze > requirements.txt
# to save dependencies
from tensorflow.keras.models import load_model
import tensorflow as tf 
import numpy as np 
import flask 
import io 
  
app = flask.Flask(__name__) 
model = None
  
def get_model(): 
      
    # global variables, to be used in another function 
    global model      
    model = load_model("my_model.h5")
    global graph  
    graph = tf.compat.v1.get_default_graph()

def prepare_image(image, target): 
    ### do something to prepare image
  
    return image 
  
@app.route("/predict", methods =["POST"]) 
def predict(): 
    data = {} # dictionary to store result 
    data["success"] = False
  
    # Check if image was properly sent to our endpoint 
    if flask.request.method == "POST": 
        url = flask.request.args.get('imageUrl')

        ## get image from url
        image = None

        image = prepare_image(image, target =(224, 224)) 

        img = np.zeros((1,100,100,1))
        x = np.vstack([img]) # just append to this if we have more than one image.
        classes = model.predict_classes(x)

        # cast data as python data type to be serializable
        data["classes"] = int(classes[0])

        data["success"] = True
  
    # return JSON response 
    return flask.jsonify(data) 
  
  
  
if __name__ == "__main__": 
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started")) 
    get_model() 
    app.run(debug = False, threaded = False) 