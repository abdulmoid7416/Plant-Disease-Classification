from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='model_xception.h5'

# Load your trained model
model = load_model(MODEL_PATH)




def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
   

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
   # x = preprocess_input(x)

    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    
    if preds==0:
        preds="""This is the leaf of an Apple plant,
        It is affected with Apple_scab disease.
        SOLUTION: 
        1. Remove and destroy the infected leaves and apples.
        2. Do not overcrowd plants, and make sure the canopy has proper airflow. This will decrease the conditions favorable for disease development.
        3. Fungicide applications at 2-week intervals beginning at the start of its season (Spring)."""
    
    elif preds==1:
        preds="""This is the leaf of an Apple plant,
        It is affected with Black_rot disease.
        SOLUTION: 
        Prune the dead or infected branches from apple trees and grapevines, 
        disinfect the pruning shears with a 70 percent alcohol solution after each cut. 
        Also improve circulation in the center of the plant by pruning away congested branches."""
    
    elif preds==2:
        preds="""This is the leaf of an Apple plant,
        It is affected with Cedar_Apple_Rust.
        SOLUTION: Remove and properly dispose the infected leaves and apples.
        Use fungicides"""

    
    elif preds==6:
        preds="""This is the leaf of a cherry plant,
        It is affected with Cherry__Powdery_mildew.
        Solution: 
        1. Rake up and destroy infected leaves to reduce the spread of spores. 
        2. Give plants good drainage and ample air circulation. 
        3. Avoid overhead watering at night; mid-morning is preferred to allow foliage to dry before evening.
        4. Use commercial fungicides (Garden Fungicide) which inhibits the attack of fungal diseases."""
    
    elif preds==7:
        preds="""This is the leaf of Corn(maize) crop.
        It is affected with Cercospora_leaf_spot/Gray_leaf_spot.
        Management strategies for gray leaf spot include tillage, crop rotation and planting resistant hybrids.
        Fungicides may be needed to prevent significant loss when plants are infected early and environmental conditions favor disease."""
    
    elif preds==8:
        preds="""This is the leaf of Corn(maize) crop.
        It is affected with Common Rust
        SOLUTION: An early fungicide application is necessary for effective disease control. 
        The use of resistant hybrids is the primary management strategy for the control of common rust.
        Timely planting of corn early during the growing season may help to avoid high inoculum levels or environmental conditions that would promote disease development.
        """
    elif preds==9:
        preds="The leaf is a healthy corn(maize) leaf"
    
    if preds==10:
        preds="""This is the leaf of Corn(maize) crop.
        It is affected with Northern Leaf Blight.
        SOLUTION:  Foliar fungicides may be applied early in the growing season to corn seedlings as a risk-management tool for northern corn leaf blight and other corn diseases, including anthracnose leaf blight and corn eyespot."""
    
    elif preds==11:
        preds="""This is the leaf of a Grape plant.
        It is affected with Black_rot.
        SOLUTION: 
        Prune the dead or infected branches from apple trees and grapevines, 
        disinfect the pruning shears with a 70 percent alcohol solution after each cut. 
        Also improve circulation in the center of the plant by pruning away congested branches."""
    
    elif preds==12:
        preds="""This is the leaf of a grape plant,
        It is affected with Esca (Black Measles).
        SOLUTION: 
        Cut the infected parts of the plant and discard it and apply fungicides."""
    
    elif preds==13:
        preds="This is a healthy leaf of grape plant"
    
    elif preds==14:
        preds="""This is the leaf of a Grape Plant,
        It is affected with Leaf_blight_(Isariopsis_Leaf_Spot).
        SOLUTION: 
        1. Remove and destroy the infected parts of the plant
        2. A mild solution of bicarbonate of soda (baking soda), using ½ teaspoon per gallon (2.5 mL. per 4 L.) of water can be used as organic treatment."""
    
    elif preds==15:
        preds="""This is the leaf of an orange plant,
        It is affected with Haunglongbing (Citrus greening).
        SOLUTION: Cultural methods include antibacterial management, sanitation, removal of infected plants are efffecive in the treatment of this disease."""
    
    elif preds==16:
        preds="""This is the leaf of Peach plant.
        It is affected with Bacterial_spot.
        SOLUTION: Destroy infected plants and apply fungicide."""
    
    elif preds==17:
        preds="The leaf is a healthy peach leaf"
    
    elif preds==18:
        preds="""This is the leaf of a Pepper-bell plant.
        It is affected with Bacterial_spot.
        SOLUTION: Destroy infected plants and apply fungicide."""
    
    elif preds==19:
        preds="The leaf is a healthy Pepper-bell leaf"
    
    elif preds==20:
        preds="""This is the leaf of a Potato plant,
        It is affected with Early_blight....   
        SOLUTION: 
        1. Prune or stake plants to improve air circulation. 
        2. Make sure to disinfect your pruning shears (one part bleach to 4 parts water) after each cut.
        3. For best control, apply copper-based fungicides early, when disease first appears, and repeat every 7-10 days for as long as needed. Or when weather forecasts predict a long period of wet weather."""
    
    elif preds==21:
        preds="The leaf is a healthy Potato leaf"
    
    elif preds==22:
        preds="""This is the leaf of the Potato plant,
        It is affected with Late_blight.
        SOLUTION:
        Apply a copper based fungicide (2 oz/ gallon of water) every 7 days or less, following heavy rain or when the amount of disease is increasing rapidly.
         Foliar spray, 'Organocide® Plant Doctor' can be used to spray the entire plant to prevent fungal problems from occurring and attack existing many problems. Mix 2 tsp/ gallon of water and spray at transplant, then at 1-2 week intervals as required to control disease."""
    
  
    elif preds==25:
        preds="""This is the leaf of a Squash plant,
        It is affected with Powdery_mildew. 
        Solution:
        1. Rake up and destroy infected leaves to reduce the spread of spores.\n
        2. Give plants good drainage and ample air circulation. \n
        3. Avoid overhead watering at night; mid-morning is preferred to allow foliage to dry before evening.
        4. Use commercial fungicides (Garden Fungicide) which inhibits the attack of fungal diseases. """
    
    elif preds==26:
        preds="The leaf is a healthy strawberry leaf"
    
    elif preds==27:
        preds="""This is the leaf of a Strawberry plant, 
        It is affected with Leaf_scorch.
        SOLUTION:
        Once leaf scorch has occurred, there is no cure. The leaves that have already turned brown will not recover, BUT as long as you water properly, the rest of the plant should survive. Deep watering is recommended – a slow, deep soaking of the soil at the roots."""
    
    elif preds==28:
        preds="""This is the leaf of a Tomato plant.
        It is affected with Bacterial_spot.
        SOLUTION: Destroy infected plants and apply fungicide."""
    
    elif preds==29:
        preds="""This is the leaf of a Tomato Plant,
        It is affected with Early_blight...
        SOLUTION: 
        1. Prune or stake plants to improve air circulation. 
        2. Make sure to disinfect your pruning shears (one part bleach to 4 parts water) after each cut.
        3. For best control, apply copper-based fungicides early, when disease first appears, and repeat every 7-10 days for as long as needed. Or when weather forecasts predict a long period of wet weather."""
    
    elif preds==30:
        preds="The leaf is a healthy tomato leaf"
    
    elif preds==31:
        preds="""This is the leaf of a Tomato Plant,
        It is affected with Late_blight.
        SOLUTION:
        Apply a copper based fungicide (2 oz/ gallon of water) every 7 days or less, following heavy rain or when the amount of disease is increasing rapidly.
         Foliar spray, 'Organocide® Plant Doctor' can be used to spray the entire plant to prevent fungal problems from occurring and attack existing many problems. Mix 2 tsp/ gallon of water and spray at transplant, then at 1-2 week intervals as required to control disease."""
    
    elif preds==32:
        preds="""This is the leaf of a Tomato Plant,
        It is affected with Leaf_Mold.
        SOLUTION:
        1. Upon noticing the infected areas, the first step is to let the plants air out and dry.  One thing you can do to help keep the leaves as dry as possible is to water in the early morning hours, that way the plant has plenty of time to dry before the sun comes out, which will keep the humidity around the leaves low. You can also try drip irrigation methods, or soak watering methods to attempt to water the soil without ever wetting the leaves of the plant.
        2. Another treatment option is fungicidal sprays. When using fungicide sprays, be sure to thoroughly cover all parts of the plant that is above ground, focusing specifically on the underside of leaves. Calcium chloride sprays are among the most highly recommended types for leaf mold. 
        3. An apple-cider and vinegar mix is also believed to treat the mold effectively."""
    
    elif preds==33:
        preds=""""This is the leaf of a Tomato Plant,
        It is affected with Septoria_leaf_spot.
        Treatment: 
        1. Remove infected leaves immediately, and be sure to wash your hands and pruners thoroughly before working with uninfected plants.
        2.  Fungicides containing either copper or potassium bicarbonate will help prevent the spreading of the disease. Begin spraying as soon as the first symptoms appear and follow the label directions for continued management.
        """
    
    elif preds==34:
        preds="""This is the leaf of a Tomato Plant,
        It is affected with Spider_mites Two-spotted_spider_mite.
        SOLUTION: The best way to begin treating for two-spotted mites is to apply a pesticide specific to mites called a miticide. Ideally, you should start treating for two-spotted mites before your plants are seriously damaged. Apply the miticide for control of two-spotted mites every 7 days or so."""
    
    elif preds==35:
        preds="""This is the leaf of a Tomato Plant,
        It is affected with Target Spot. 
        SOLUTION: 
        Use Fungicides containing chlorothalonil, mancozeb, and copper oxychloride."""
    
    elif preds==36:
        preds="""This is the leaf of Tomato plant.
        It is affected with mosaic_virus.  
        SOLUTION: 
        1. Remove all infected plants and destroy them. Do NOT put them in the compost pile, as the virus may persist in infected plant matter. 
        2. Burn infected plants or throw them out with the garbage.
        3. Monitor the rest of your plants closely, especially those that were located near infected plants.
        4. Disinfect gardening tools after every use. Keep a bottle of a weak bleach solution or other antiviral disinfectant to wipe your tools down with."""
    
    elif preds==37:
        preds="""This is the leaf of a Tomato plant,
        It is affected with Tomato_Yellow_Leaf_Curl_Virus.
        SOLUTION:
         Use a neonicotinoid insecticide, such as dinotefuran (Venom) imidacloprid (AdmirePro, Alias, Nuprid, Widow, and others) or thiamethoxam (Platinum). 
        Other methods to control the spread of TYLCV include planting resistant/tolerant lines, crop rotation, and breeding for resistance of TYLCV."""

    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(port=5001,debug=True)
    
