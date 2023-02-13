# Imports
from fastapi import FastAPI
import pickle, uvicorn, os
import uvicorn
from pydantic import BaseModel
from sklearn.preprocessing import OneHotEncoder as ohe
import pandas as pd
import numpy as np

#########################################################################
# Config & setup


## Variable of environment (path of ml items)
DIRPATH = os.path.dirname(__file__)
ASSETSDIRPATH = os.path.join(DIRPATH, 'assets')
ML_ITEMS_PKL = os.path.join(ASSETSDIRPATH, 'ML_items.pkl')

print(f"{'*'*10} Config {'*'*10}\n INFO: DIRPAHT = {DIRPATH} \n INFO: ASSETSDIRPATH = {ASSETSDIRPATH} ")


## API Basic config
app = FastAPI(title = 'Titanic Survival API',
              version = '0.0.1',
              description = 'Prediction Titanic Survival')


## Loading of assets
with open(ML_ITEMS_PKL,"rb") as f:
    loaded_items = pickle.load(f)
print("INFO: Loaded assets:", loaded_items)

model = loaded_items['model']
encoder = loaded_items['encoder']
num_cols = loaded_items['numeric_columns']
cat_cols = loaded_items['categorical_columns']

#########################################################################
# API core


## BaseModel
class ModelInput(BaseModel):
    # Survived : float
    Pclass : int
    Sex : str
    Age : float
    SibSp : int
    Parch : int
    Fare : float
    Cabin : str
    Embarked : str
    
## Utilities

# def feature_engeneering(
#     dataset, encoder, imputer, FE=encoder
# ):  
#     "Cleaning, Processing and Feature Engineering of the input dataset."
#     """:dataset pandas.DataFrame"""

    
#     output_dataset = dataset.copy()

#     if FE is not None:
#         output_dataset = FE.transform(output_dataset)


    # output_dataset = dataset(encoder.transform(cat_cols))

    
                                
  

    # return output_dataset

def processing_FE(dataset, encoder, imputer=None, scaler=None, FE=encoder):
    "Cleaning, Processing and Feature Engineering of the input dataset."
    """:dataset pandas.DataFrame"""

    if imputer is not None:
        output_dataset = imputer.transform(dataset)
    else:
        output_dataset = dataset.copy()
    
    output_dataset = encoder.transform(cat_cols)

    if scaler is not None:
        output_dataset = scaler.transform(output_dataset)

    if FE is not None:
        output_dataset = FE.transform(cat_cols)

    return output_dataset

# categ = ['Sex', 'Cabin', 'Embarked']
# num_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']


def make_predict(Pclass, Sex, Age, SibSp, Parch, Fare, Cabin, Embarked):
    ""
    df = pd.DataFrame([[Pclass, Sex, Age, SibSp, Parch, Fare, Cabin, Embarked]], 
                      columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked'])


    # X = processing_FE(dataset=df, scaler=None, imputer=None, encoder=encoder, FE=None)
    
    X = processing_FE(dataset=df, encoder=encoder,)
    model_output = model.predict(X).tolist()

    # print(type(model_output))
    # print(model_output)

    return model_output


## Endpoints
@app.post("/Titanic")
async def predict(input:ModelInput):
    """ __descr__

    __datails__
    """
    output_pred = make_predict(Pclass = input.Pclass,
                          Sex = input.Sex,
                          Age = input.Age,
                          SibSp = input.SibSp,
                          Parch = input.Parch,
                          Fare = input.Fare,
                          Cabin = input.Cabin,
                          Embarked = input.Embarked,
                          )
    
    if (output_pred[0]>0):
        output_pred ="This passenger DID NOT survive the Titanic shipwreck."
    else:
        output_pred ="This passenger SURVIVED the Titanic shipwreck."
    return {
        "prediction": output_pred,
       # "input": input,
    }
    


#########################################################################
# Execution

if __name__=='__main__':
    uvicorn.run('api:app',
            reload = True,
            )