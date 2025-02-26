import pandas as pd 

from ..nielsen_data_cleaning.product_data_creation import creating_product_data_for_comparison

def list_retailers_with_predictions(path: str):
    data = pd.read_csv(path)
    retailers_list = set(data['store_code_uc'])
    return retailers_list


def dict_retailers_brands(path:str)-> dict:
    data = pd.read_csv(path)
    return data.groupby('')[''].unique.apply(list).to_dict()


def filter_observed_by_predicted_data(path:str) -> pd.DataFrame:
    
    return 


def main():
    #obtener los códigos de los retailers para los que se generaron las comparaciones 


    #obtener las marcas por retailer 


    #crear la base de datos con toda la información



    #filtrar la baes de datos 


    #merge de la base de datos con predicciones con la base de datos de los precios observados 

    pass