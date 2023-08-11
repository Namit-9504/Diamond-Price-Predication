import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from src.components.data_ingestion import DataIngestion

from src.components.data_trasformation import DataTransformation
from src.components.model_trainer import Modeltrainer


if __name__=='__main__':
    obj=DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    print(train_data_path,test_data_path)
    data_trasformation=DataTransformation()
    train_arr,test_arr,ob_path=data_trasformation.initiate_data_transformation(train_data_path,test_data_path)

    model_trainer=Modeltrainer()
    model_trainer.initiate_model_training(train_arr,test_arr)