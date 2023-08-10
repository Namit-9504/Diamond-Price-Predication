from sklearn.impute import SimpleImputer ## Handling Missing Values
from sklearn.preprocessing import StandardScaler ## Feature Scaliing 
from sklearn.preprocessing import OrdinalEncoder ### Catagorical to Numerical 
## pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging 
import sys,os
from dataclasses import dataclass

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Data Transformation Initiated")
            
            ## Catagorical and Numerical columns
            catagorical_col=['cut', 'color', 'clarity'],
            numerical_col=['carat', 'depth', 'table', 'x', 'y', 'z']

            ## Define Custom ranking for Oridinal Encoder
            cut_categories=["Fair","Good","Very Good","Premium","Ideal"]
            color_categories=["D","E","F","G","H","I","J"]
            clarity_categories=["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"]

            logging.info("Data Transfromation Pipeline Initiated")


            ## pipelines 

            num_pipeline =Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )


            cat_pipeline= Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("encoder",OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                    ("scaler",StandardScaler())
                ]
            )

            preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_col),
                ("cat_pipeline",cat_pipeline,catagorical_col)
            ])

            logging.info("Data Transformation Completed")

            return preprocessor

        except Exception as e:
            logging.info("Exception Occurs in Data Transformation")
            raise CustomException(e,sys)
        

    def initiate_data_trasformation(self,train_data_path,test_data_path):

        try:
            train_df=pd.read_csv(train_data_path)
            test_df=pd.read_csv(test_data_path)

            logging.info("Read Train and Test Data Completed")
            logging.info(f"Train DataFrame Head : \n {train_df.head().to_string()}")
            logging.info(f"Test DataFrame Head : \n {test_df.head().to_string()}")

            preprocessing_obj=self.get_data_transformation_object()

            target_column="price"
            drop_columns=[target_column,"id"]


            ## Dividing DataSet into Independent and Dependent Dataset
            ## training data

            X_train_arr=train_df.drop(columns=drop_columns,axis=1)
            y_train1=train_df[target_column]

            ## test_data

            X_test_arr=test_df.drop(columns=drop_columns,axis=1)
            y_test1=test_df[target_column]


            ## Data Transformation
            X_train_arr=preprocessing_obj.fit_transform(X_train_arr)
            X_test_arr=preprocessing_obj.transform(X_test_arr)

            logging.info("Appling Preprocessing object on training and testing dataset.")

            train_arr=np.c_[X_train_arr,np.array(y_train1)]
            test_arr=np.c_[X_test_arr,np.array(y_test1)]

            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path

            )
        except Exception as e:
            raise CustomException(e,sys)