import os
import re
import json
import sys

import pandas as pd 
import json
import matplotlib.pyplot as plt
import sys

from ..nielsen_data_cleaning.descarga_merge import movements_file, stores_file, products_file, extra_attributes_file, filter_row_weeks, retail_market_ids_identifier, retail_market_ids_fips
from ..nielsen_data_cleaning.precios_ingresos_participaciones import total_income, total_units, unitary_price, price


def load_price_predictions(data: pd.DataFrame, directory: str=os.getcwd()):
    pattern = 'price_predictions_'
    iteration = 1
    files = sorted(os.listdir(directory), key=lambda x: int(re.search(r'_(\d+)\.json', x).group(1)))
    for file in files:
        with open(os.path.join(directory, file), 'r') as f:
            this_data = json.load(f)
        data[f'price_prediction_{iteration}'] = this_data['price_prediction']
        iteration += 1
    
    return data


def calculate_stats(row):
    stats = {
        "mean": row.mean(),
        "median": row.median(),
        "std": row.std(),
        "min": row.min(),
        "max": row.max(),
    }
    return pd.Series(stats)


def add_descriptive_stats(data):
    prediction_columns = [col for col in data.columns if col.startswith("price_prediction")]
    stats = data[prediction_columns].apply(calculate_stats, axis=1)
    stats.columns = [f"price_prediction_{stat}" for stat in stats.columns]
    data_descriptive = pd.concat([data, stats], axis=1)
    return data_descriptive


def plot_histogram(data_descriptive, prediction_columns):
    data_descriptive.iloc[0][prediction_columns].plot.hist(bins=20, title="Histogram of Series")
    plt.xlabel("Bins")
    plt.ylabel("Frequency")
    plt.show()


def creating_product_data_for_comparison(main_dir: str,
                      movements_path:str,
                      stores_path:str,
                      products_path:str,
                      extra_attributes_path: str,
                      first_week: int=0,
                      num_weeks: int=1,
                      rcl: bool= True):
    # os.chdir(f'/oak/stanford/groups/polinsky/Nielsen_data/Mergers/{DIRECTORY_NAME}/nielsen_extracts/RMS/{YEAR}/Movement_Files/{DEPARTMENT_CODE}_{YEAR}/')
    os.chdir(path= main_dir)

    # Descarga los datos
    movements_data = movements_file(movements_path=movements_path, 
                                    filter_row_weeks=filter_row_weeks, 
                                    first_week=first_week, 
                                    num_weeks=num_weeks)
    
    print('movements_data:', movements_data.shape)
    print(sorted(set(movements_data['week_end'])))
    stores_data = stores_file(stores_path=stores_path)
    print('stores_data: ', stores_data.shape)
    products_data = products_file(products_path=products_path)
    print('product_data: ', products_data.shape)
    extra_attributes_data = extra_attributes_file(extra_attributes_path=extra_attributes_path, 
                                                  moves_data=movements_data)
    print('extra_ats: ', extra_attributes_data.shape )

    # Combina los datos
    product_data = pd.merge(movements_data, stores_data, on='store_code_uc', how='left')
    print('1 product_data: ', product_data.shape)
    product_data = pd.merge(product_data, products_data, on='upc', how='left')
    print('2 product_data: ', product_data.shape)
    product_data = pd.merge(product_data, extra_attributes_data, on='upc', how='left')
    print('3 product_data: ', product_data.shape)

    # Crea variables 
    product_data['week_end_ID'] = pd.factorize(product_data['week_end'])[0]
    product_data['market_ids'] = product_data.apply(retail_market_ids_identifier, axis=1)
    product_data['market_ids_fips'] = product_data.apply(retail_market_ids_fips, axis=1)
    product_data['firm_ids'] = None
    product_data['brand_descr'] = product_data['brand_descr'].fillna('Not_identified')
    product_data['total_income'] = product_data.apply(total_income, axis=1)
    product_data['total_individual_units'] = product_data.apply(total_units, axis=1)
    product_data['unitary_price'] = product_data.apply(unitary_price, axis=1)
   
    # Cambia el nombre de una variable
    product_data.rename(columns={'store_zip3':'zip'}, inplace=True)

    # Agrega ventas a nivel de tienda y marca 
    product_data = product_data.groupby(['market_ids', 'brand_descr', 'store_code_uc'], as_index=False).agg({
                'zip':'first' ,
                'week_end':'first' ,
                'week_end_ID':'first',
            #     'upc':'first', # se pierde al agregar a través de marcas
                'market_ids_fips':'first',
                'fips_state_code':'first', 
                'fips_state_descr':'first', 
                'fips_county_code':'first', 
                'fips_county_descr':'first',
                'firm_ids':'first', #No está definido aún. 
                'brand_code_uc': 'first',
                'brand_descr':'first',
                'units': 'sum',
                'unitary_price':'mean',#,No vale la pena agregarlo porque no se puede calcular como el promedio simple de todas las observaciones
                'price': 'mean',
                'total_individual_units': 'sum',
                'total_income': 'sum',
                
            #     'prices': 'mean'  ,
            #     'total dollar sales': 'sum' ,
                'style_code': 'mean' ,
                'style_descr': 'first',
                'type_code': 'mean' ,
                'type_descr': 'first' ,
                'strength_code': 'mean',
                'strength_descr': 'first', 
            #     'total dollar sales': 'sum',   # Summing up the 'Value1' column
            #     'Value2': 'mean'   # Calculating mean of the 'Value2' column
            })

    product_data.rename(columns={'unitary_price':'unitary_price_x_reemplazar', 'price':'price_x_reemplazar'}, inplace=True)
    
    # Crea variable precios
    product_data['prices'] = product_data.apply(price, axis=1)

    # Elimina ventas que no tienen identificada la marca
    product_data = product_data[product_data['brand_code_uc'].notna()]

    # Cambio del nombre de IDS de mercados y genera indicador para numérico para estos 
    product_data.rename(columns={'market_ids_fips':'market_ids_string'}, inplace=True)
    product_data['market_ids']=product_data['market_ids_string'].factorize()[0]

    # Save product_data DataFrame to the specified location


    # Crea directorio para guardar las predicciones
    week_dir = list(set(product_data['week_end']))[0] if len(set(product_data['week_end'])) == 1 else None
    output_dir = f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/Observed/{week_dir}/'
    os.makedirs(output_dir, exist_ok=True)
    # product_data.to_csv(os.path.join(output_dir, 'product_data.csv'), index=False)

    # Save product_data DataFrame to JSON
    product_data.to_json(os.path.join(output_dir, 'product_data.json'), orient='records', lines=True)

    return product_data


def main():
    os.chdir('/oak/stanford/groups/polinsky/Mergers/Cigarettes/results/price_predictions')      
    with open('price_predictions_0.json', 'r') as file:
        data = json.load(file)
    
    data = load_price_predictions(data=data)
    data = pd.DataFrame(data)
    
    prediction_columns = [col for col in data.columns if col.startswith("price_prediction")]

    stats = data[prediction_columns].apply(calculate_stats, axis=1)
    data = pd.concat([data, stats], axis=1)

    print('ran')


def main2():
    product_data = creating_product_data_for_comparison(main_dir='/oak/stanford/groups/polinsky/Mergers/cigarettes',
                      movements_path='/oak/stanford/groups/polinsky/Mergers/Cigarettes/Nielsen_data/2016/Movement_Files/4510_2016/7460_2016.tsv',
                      stores_path='/oak/stanford/groups/polinsky/Mergers/Cigarettes/Nielsen_data/2016/Annual_Files/stores_2016.tsv',
                      products_path='/oak/stanford/groups/polinsky/Mergers/Cigarettes/Nielsen_data/Master_Files/Latest/products.tsv',
                      extra_attributes_path='/oak/stanford/groups/polinsky/Mergers/Cigarettes/Nielsen_data/2016/Annual_Files/products_extra_2016.tsv',
                      first_week=4,
                      num_weeks=1)

if __name__ == '__main__':
    pass
    

