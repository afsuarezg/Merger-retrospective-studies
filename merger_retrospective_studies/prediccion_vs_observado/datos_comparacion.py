import os
import pandas as pd
import datetime
import pyblp
import numpy as np
import sys
import json
from itertools import chain
from scipy.linalg import svd

# from dotenv import load_dotenv

from ..nielsen_data_cleaning.descarga_merge import movements_file, stores_file, products_file, extra_attributes_file, retail_market_ids_fips, retail_market_ids_identifier, filter_row_weeks
from ..nielsen_data_cleaning.caracteristicas_productos import match_brands_to_characteristics, list_of_files, characteristics
from ..nielsen_data_cleaning.empresas import find_company_pre_merger, find_company_post_merger, brands_by_company_pre_merger, brands_by_company_post_merger
from ..nielsen_data_cleaning.consumidores_sociodemograficas import read_file_with_guessed_encoding, process_file, get_random_samples_by_code, KNNImputer, add_random_nodes
from ..nielsen_data_cleaning.precios_ingresos_participaciones import total_income, total_units, unitary_price, price, fraccion_ventas_identificadas, prepend_zeros, shares_with_outside_good
from ..estimaciones.plain_logit import plain_logit
from ..estimaciones.rcl_without_demographics import rcl_without_demographics, rename_instruments
from ..estimaciones.rcl_with_demographics import rcl_with_demographics
from ..estimaciones.utils import save_dict_json
from ..estimaciones.post_estimation_merger_simulation import predict_prices
from ..estimaciones.optimal_instruments import results_optimal_instruments
from ..nielsen_data_cleaning.utils import find_first_non_collinear_matrix


def creating_comparison_product_data_rcl(main_dir: str,
                      movements_path:str,
                      stores_path:str,
                      products_path:str,
                      extra_attributes_path: str,
                      stores_to_include: list[int],
                      first_week: int=0,
                      num_weeks: int=1,
                      ):
    """
    Creates and processes product data by merging various datasets and performing multiple transformations.
    Parameters:
    main_dir (str): The main directory path where the data files are located.
    movements_path (str): Path to the movements data file.
    stores_path (str): Path to the stores data file.
    products_path (str): Path to the products data file.
    extra_attributes_path (str): Path to the extra attributes data file.
    first_week (int, optional): The first week to filter the data. Defaults to 0.
    num_weeks (int, optional): The number of weeks to filter the data. Defaults to 1.
    lower_threshold_identified_sales (float, optional): The lower threshold for identified sales. Defaults to 0.8.
    Returns:
    pd.DataFrame: A DataFrame containing the processed product data.
    """
    # os.chdir(f'/oak/stanford/groups/polinsky/Nielsen_data/Mergers/{DIRECTORY_NAME}/nielsen_extracts/RMS/{YEAR}/Movement_Files/{DEPARTMENT_CODE}_{YEAR}/')
    os.chdir(path= main_dir)
    #-------------------------------------------------------------------
    # Descarga los datos
    movements_data = movements_file(movements_path=movements_path, 
                                    filter_row_weeks=filter_row_weeks, 
                                    first_week=first_week, 
                                    num_weeks=num_weeks)
    
    print('movements_data:', movements_data.shape)
    print(sorted(set(movements_data['week_end'])))
    
    # Información de las tiendas
    stores_data = stores_file(stores_path=stores_path)
    print('stores_data: ', stores_data.shape)
    
    # Información de los productos
    products_data = products_file(products_path=products_path)
    print('product_data: ', products_data.shape)
    
    # Información extra o adicional de los productos
    # extra_attributes_data = extra_attributes_file(extra_attributes_path=extra_attributes_path, 
    #                                               moves_data=movements_data)
    # print('extra_ats: ', extra_attributes_data.shape )

    # Combina los datos
    product_data = pd.merge(movements_data, stores_data, on='store_code_uc', how='left')
    print('1 product_data: ', product_data.shape)
    product_data = pd.merge(product_data, products_data, on='upc', how='left')
    print('2 product_data: ', product_data.shape)
    # product_data = pd.merge(product_data, extra_attributes_data, on='upc', how='left')
    # print('3 product_data: ', product_data.shape)

    # Crea variables 
    product_data['week_end_ID'] = pd.factorize(product_data['week_end'])[0]
    product_data['market_ids'] = product_data.apply(retail_market_ids_identifier, axis=1)
    # product_data['market_ids_fips'] = product_data.apply(retail_market_ids_fips, axis=1) # //TODO Revisar cómo se está usando esta variable porque no estoy seguro de que esté creando un ID con información de FIPS
    product_data['firm_ids'] = None
    product_data['brand_descr'] = product_data['brand_descr'].fillna('Not_identified')
    product_data['total_income'] = product_data.apply(total_income, axis=1)
    product_data['total_individual_units'] = product_data.apply(total_units, axis=1)
    product_data['unitary_price'] = product_data.apply(unitary_price, axis=1)
   
    # Cambia el nombre de una variable
    product_data.rename(columns={'store_zip3':'zip'}, inplace=True)

    print('product_data columns: ', product_data.columns)
    # Agrega ventas a nivel de mercado y marca 
    product_data = product_data.groupby(['market_ids', 'brand_descr'], as_index=False).agg({
                'zip':'first' ,
                'week_end':'first' ,
                'week_end_ID':'first',
                'store_code_uc':'first',
                #'market_ids_fips':'first',
                # 'fips_state_code':'first', 
                # 'fips_state_descr':'first', 
                # 'fips_county_code':'first', 
                # 'fips_county_descr':'first',
                # 'firm_ids':'first', #No está definido aún. 
                'brand_code_uc': 'first',
                'brand_descr':'first',
                'units': 'sum',
                # 'unitary_price':'mean',#,No vale la pena agregarlo porque no se puede calcular como el promedio simple de todas las observaciones
                # 'price': 'mean',
                'total_individual_units': 'sum',
                'total_income': 'sum',
                # 'style_code': 'mean' ,
                # 'style_descr': 'first',
                # 'type_code': 'mean' ,
                # 'type_descr': 'first' ,
                # 'strength_code': 'mean',
                # 'strength_descr': 'first', 
            })

    # product_data.rename(columns={'unitary_price':'unitary_price_x_reemplazar', 'price':'price_x_reemplazar'}, inplace=True)
    
    # Crea variable precios
    product_data['prices'] = product_data.apply(price, axis=1)

    #-------------------------------------------------------------------
    # # Determinar porción de ventas identificadas para cada tienda (recordar que los mercados se definieron como la combinación entre tienda y semana) a través de ingresos 
    # #total de ventas por mercado, identificadas y sin identificar.
    # total_sales_per_marketid = pd.DataFrame(product_data.groupby(by=['market_ids'], as_index=False).agg({'total_income': 'sum'}))
    # total_sales_per_marketid.rename(columns={'total_income':'total_income_market'}, inplace=True)
    
    # #total de ventas por mercado, solo de marcas identificadas.
    # total_sales_identified_per_marketid = pd.DataFrame(product_data[product_data['brand_descr']!='Not_identified'].groupby(by=['market_ids'],as_index=False).agg({'total_income': 'sum'}))
    # total_sales_identified_per_marketid.rename(columns={'total_income':'total_income_market_known_brands'}, inplace=True)

    # #Agrega columnas de total_income_market y total_income_market_known_brands
    # product_data = product_data.merge(total_sales_per_marketid, how ='left', on='market_ids')
    # product_data = product_data.merge(total_sales_identified_per_marketid, how='left', on='market_ids')
    
    # #fillna para evitar errores de NaN
    # product_data.fillna({'total_income_market_known_brands': 0.0}, inplace=True)
    
    # #Calcula y agrega columna de fraccion de ventas identificadas
    # product_data['fraction_identified_earnings'] = product_data.apply(fraccion_ventas_identificadas, axis=1)

    # #Elimina mercados con fraccion de ventas identificadas inferiores a un threshold previamente definido.
    # product_data = product_data[product_data['fraction_identified_earnings'] >= lower_threshold_identified_sales]
    #-------------------------------------------------------------------
   
    # # Después de identificar los mercados para los que las ventas identificadas por valor son superiores al threshold definido, se suman las unidades vendidas para dichos mercados. 
    # # Suma total de unidades identificadas y vendidas por tienda 
    # total_units_sold_per_marketid = pd.DataFrame(product_data.groupby(by=['market_ids'], as_index=False).agg({'units': 'sum'}))
    # total_units_sold_per_marketid.rename(columns={'units':'total_units_market'}, inplace=True)
    # product_data = product_data.merge(total_units_sold_per_marketid, how ='left', on=['market_ids'])

    # # Elimina ventas que no tienen identificada la marca
    # product_data = product_data[product_data['brand_code_uc'].notna()] 
    #-------------------------------------------------------------------    
    
    # # Adición de información poblacional 
    # #TODO Crear una carpeta con la información poblacional para diferentes años y actualizar el path para que el archivo de fips correspondiente se cargue automáticamente 
    # fips_pop= pd.read_excel('/oak/stanford/groups/polinsky/Tamaño_mercado/PopulationEstimates.xlsx', skiprows=4)
    # fips_pop=fips_pop[['FIPStxt','State','CENSUS_2020_POP']]
    # # fips_pop=fips_pop[['FIPStxt', 'State', f'CENSUS_{YEAR}_POP']]
    # fips_pop['fips'] = fips_pop['FIPStxt'].astype('int')
    # # fips_pop['FIPStxt']=fips_pop['FIPStxt'].astype(str)
    # product_data['fips'] = product_data.apply(prepend_zeros, axis=1).astype('int')
    # product_data=product_data.merge(fips_pop[['CENSUS_2020_POP','fips']], how='left', on='fips')
    # product_data.rename(columns={'fips':'FIPS', 'fips_state_code':'GESTFIPS'}, inplace=True)

    #-------------------------------------------------------------------

    # # Calculo de participaciones de mercado incluyendo la participación del bien externo. 
    # # TODO Revisar que las participaciones de mercado estimadas del modelo sean cercanas a las observadas en otros estudios. Aunque es difícil que ocurra porque no se tiene toda la demanda por ubicación geográfica.
    # product_data['shares']=product_data.apply(shares_with_outside_good, axis=1)    
    # product_data.rename(columns={'CENSUS_2020_POP':'poblacion_census_2020'}, inplace=True)

    #-------------------------------------------------------------------
    # Asignación de las marcas por empresa -> Realiza el mapping entre las marcas y las empresas con anterioridad y posterioridad a la integración. 
    # print('Marcas antes de mapping con firmas: ', set(product_data['brand_descr']))
    product_data['brand_descr']=product_data['brand_descr'].str.lower()
    # print('Diferencia entre base de datos y diccionario: ', set(product_data['brand_descr']).intersection(set(list(chain(*[v for k,v in brands_by_company_pre_merger.items()])))))

    product_data['firm']=product_data.apply(find_company_pre_merger, axis=1, company_brands_dict=brands_by_company_pre_merger)
    product_data['firm_ids']=pd.factorize(product_data['firm'])[0]
    product_data['firm_post_merger']=product_data.apply(find_company_post_merger, axis=1,company_brands_dict=brands_by_company_post_merger)
    product_data['firm_ids_post_merger']=pd.factorize(product_data['firm_post_merger'])[0]

    #-------------------------------------------------------------------
    # # Agrega características de los productos al archivo de product data
    # brands_to_characteristics = pd.read_json('/oak/stanford/groups/polinsky/Mergers/Cigarettes/Firmas_marcas/brands_to_characteristics2.json')
    # brands_to_characteristics['from Nielsen']=brands_to_characteristics['from Nielsen'].str.lower()

    # product_data = product_data.merge(brands_to_characteristics, how='inner', left_on='brand_descr', right_on='from Nielsen')
    # product_data = product_data.merge(characteristics, how='inner', left_on='from characteristics', right_on='name')

    # # Elimina productos para los que no se pudieron identificar las características.
    # product_data = product_data[product_data['name'].notna()]
    # product_data = product_data.dropna(subset=['tar', 'nicotine', 'co', 'nicotine_mg_per_g', 'nicotine_mg_per_g_dry_weight_basis', 'nicotine_mg_per_cig'])
    #-------------------------------------------------------------------
    # # Cambio del nombre de IDS de mercados y genera indicador numérico para estos 

    # product_data['market_ids']=product_data['market_ids'].factorize()[0]
    # # Creacion de dataframe organizando por nivel de ingresos identificados    
    # # Creación de identificador numérico para los productos

    # product_data['product_ids'] = pd.factorize(product_data['brand_descr'])[0]
    #-------------------------------------------------------------------
    product_data = product_data[product_data['store_code_uc'].isin(stores_to_include)]
    # product_data=product_data[['market_ids','zip', 'week_end', 'store_code_uc', 'brand_descr','brand_code_uc', 'prices' ]]
    week=product_data['week_end'].iloc[0]
    product_data.to_pickle(fr'/oak/stanford/groups/polinsky/Mergers/Cigarettes/Comparison_product_data/product_data_stores_week_{week}.pkl')


    return product_data
