import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso,Ridge,ElasticNet,LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from src.logger import logging 
from src.exception import CustomException
from sklearn.neighbors import KNeighborsRegressor
from src.utils import save_object
from src.utils import evaluate_model
from dataclasses import dataclass
import os
import sys


@dataclass
class ModeltrainingConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class Modeltrainer:
    def __init__(self):
        self.model_trainer_config=ModeltrainingConfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info("Spliting dependent and Independent Variables ")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
                "LinearRegression":LinearRegression(),
                "Lasso":Lasso(),
                "Ridge":Ridge(),
                "Elastic_Net":ElasticNet(),
                "DecisionTree":DecisionTreeRegressor(),
                "KNeighborsRegressor":KNeighborsRegressor(),
                "RandomForgestRegressor":RandomForestRegressor()
            }
            
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print("\n=================================================================================================================================================================")
            logging.info(f"Model Report:{model_report}")

            ## To select Best Model
            Best_model_score=max(sorted(model_report.values()))

            ## Getting Best Model Name
            Best_model_name=list(model_report.keys())[
                list(model_report.values()).index(Best_model_score)
            ]

            Best_model=models[Best_model_name]

            print(f"Best Model Found, Best model is {Best_model_name} Best R2_score is {Best_model_score}")
            print("\n=============================================================================================================")
            logging.info(f"Best Model Found, Best model is {Best_model_name} Best R2_score is {Best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=Best_model
            )

        except Exception as e:
            logging.info("Exception Occurs at Model training")
            raise CustomException(e,sys)