import os
import pandas as pd
import datetime
import pyblp
import numpy as np
import sys
import json
from itertools import chain

# from dotenv import load_dotenv

from .nielsen_data_cleaning.descarga_merge import movements_file, stores_file, products_file, extra_attributes_file, retail_market_ids_fips, retail_market_ids_identifier, filter_row_weeks
from .nielsen_data_cleaning.caracteristicas_productos import match_brands_to_characteristics, list_of_files, characteristics
from .nielsen_data_cleaning.empresas import find_company_pre_merger, find_company_post_merger, brands_by_company_pre_merger, brands_by_company_post_merger
from .nielsen_data_cleaning.consumidores_sociodemograficas import read_file_with_guessed_encoding, process_file, get_random_samples_by_code, KNNImputer, add_random_nodes
from .nielsen_data_cleaning.precios_ingresos_participaciones import total_income, total_units, unitary_price, price, fraccion_ventas_identificadas, prepend_zeros, shares_with_outside_good
from .estimaciones.plain_logit import plain_logit
from .estimaciones.rcl_without_demographics import rcl_without_demographics, rename_instruments
from .estimaciones.rcl_with_demographics import rcl_with_demographics
from .estimaciones.estimaciones_utils import save_dict_json
from .estimaciones.post_estimation import predict_prices
from .estimaciones.optimal_instruments import results_optimal_instruments


DIRECTORY_NAME = 'Reynolds_Lorillard'
DEPARTMENT_CODE = 4510 #aka product_group_code
# PRODUCT_MODULE = 7460
# NROWS = 10000000
YEAR = 2014
# WEEKS = [20140125, 20140201]



print(brands_by_company_pre_merger)

print(brands_by_company_post_merger)


def creating_product_data_for_comparison(main_dir: str,
                      movements_path:str,
                      stores_path:str,
                      products_path:str,
                      extra_attributes_path: str,
                      first_week: int=0,
                      num_weeks: int=2,
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
    week_dir = list(set(product_data['week_end']))[0] if len(set(product_data['week_end'])) == 1 else None
    output_dir = f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/Observed/{week_dir}/'
    os.makedirs(output_dir, exist_ok=True)
    # product_data.to_csv(os.path.join(output_dir, 'product_data.csv'), index=False)

    # Save product_data DataFrame to JSON
    product_data.to_json(os.path.join(output_dir, 'product_data.json'), orient='records', lines=True)

    return product_data


def creating_product_data_rcl(main_dir: str,
                      movements_path:str,
                      stores_path:str,
                      products_path:str,
                      extra_attributes_path: str,
                      first_week: int=0,
                      num_weeks: int=1,
                      fractioned_identified_earning: float=0.5):
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

    # Identificar porción de ventas identificadas para cada tienda a través de ingresos 
    total_sales_per_marketid = pd.DataFrame(product_data.groupby(by=['market_ids','store_code_uc'], as_index=False).agg({'total_income': 'sum'}))
    total_sales_per_marketid = total_sales_per_marketid.rename(columns={'total_income':'total_income_market'})
    total_sales_identified_per_marketid = pd.DataFrame(product_data[product_data['brand_descr']!='Not_identified'].groupby(by=['market_ids','store_code_uc'],as_index=False).agg({'total_income': 'sum'}))
    total_sales_identified_per_marketid = total_sales_identified_per_marketid.rename(columns={'total_income':'total_income_market_known_brands'})
    product_data = product_data.merge(total_sales_per_marketid, how ='left', on=['market_ids','store_code_uc'])
    product_data = product_data.merge(total_sales_identified_per_marketid, how='left', on=['market_ids','store_code_uc'])
    product_data.fillna({'total_income_market_known_brands': 0.0}, inplace=True)
    product_data['fraction_identified_earnings'] = product_data.apply(fraccion_ventas_identificadas, axis=1)

    # Suma total de unidades vendidas por tienda 
    total_sold_units_per_marketid = pd.DataFrame(product_data.groupby(by=['market_ids', 'store_code_uc'], as_index=False).agg({'units': 'sum'}))
    total_sold_units_per_marketid.rename(columns={'units':'total_units_retailer'}, inplace=True)
    product_data = product_data.merge(total_sold_units_per_marketid, how ='left', on=['market_ids','store_code_uc'])

    # Elimina ventas que no tienen identificada la marca
    product_data = product_data[product_data['brand_code_uc'].notna()]

    # Adición de información poblacional.
    fips_pop= pd.read_excel('/oak/stanford/groups/polinsky/Tamaño_mercado/PopulationEstimates.xlsx', skiprows=4)
    fips_pop=fips_pop[['FIPStxt','State','CENSUS_2020_POP']]
    fips_pop['FIPS'] = fips_pop['FIPStxt'].astype('int')
    fips_pop['FIPStxt']=fips_pop['FIPStxt'].astype(str)
    product_data['fip'] = product_data.apply(prepend_zeros, axis=1).astype('int')
    fips_pop = fips_pop.rename(columns={'FIPS': 'fip'})
    product_data=product_data.merge(fips_pop[['CENSUS_2020_POP','fip']], how='left', on='fip')
    product_data=product_data.rename(columns={'fip':'FIPS', 'fips_state_code':'GESTFIPS'})

    # Calculo de participaciones de mercado incluyendo la participación del bien externo. 
    product_data['shares']=product_data.apply(shares_with_outside_good, axis=1)    
    product_data.rename(columns={'CENSUS_2020_POP':'poblacion_census_2020'}, inplace=True)

    # Asignación de las marcas por empresa 
    # print('Marcas antes de mapping con firmas: ', set(product_data['brand_descr']))
    product_data['brand_descr']=product_data['brand_descr'].str.lower()
    print('Diferencia entre base de datos y diccionario: ', set(product_data['brand_descr']).intersection(set(list(chain(*[v for k,v in brands_by_company_pre_merger.items()])))))

    product_data['firm']=product_data.apply(find_company_pre_merger, axis=1, company_brands_dict=brands_by_company_pre_merger)
    product_data['firm_ids']=(pd.factorize(product_data['firm']))[0]
    product_data['firm_post_merger']=product_data.apply(find_company_post_merger, axis=1, company_brands_dict=brands_by_company_post_merger)
    product_data['firm_ids_post_merger']=(pd.factorize(product_data['firm_post_merger']))[0]


    # Save product_data DataFrame to the specified directory
    output_dir = '/oak/stanford/groups/polinsky/Mergers/Cigarettes/Pruebas'
    os.makedirs(output_dir, exist_ok=True)
    product_data.to_csv(os.path.join(output_dir, 'product_data_previo.csv'), index=False)

    brands_to_characteristics = pd.read_json('/oak/stanford/groups/polinsky/Mergers/Cigarettes/Firmas_marcas/brands_to_characteristics2.json')
    brands_to_characteristics['from Nielsen']=brands_to_characteristics['from Nielsen'].str.lower()

    print('4 product_data: ', product_data.shape)
    product_data = product_data.merge(brands_to_characteristics, how='inner', left_on='brand_descr', right_on='from Nielsen')
    print('4.1 product_data: ', product_data.shape)
    product_data = product_data.merge(characteristics, how='inner', left_on='from characteristics', right_on='name')
    print('4.2 product_data: ', product_data.shape)
    product_data = product_data[product_data['name'].notna()]
    print('5 product_data: ', product_data.shape)

    product_data = product_data.dropna(subset=['tar', 'nicotine', 'co', 'nicotine_mg_per_g', 'nicotine_mg_per_g_dry_weight_basis', 'nicotine_mg_per_cig'])

    # Cambio del nombre de IDS de mercados y genera indicador para numérico para estos 
    product_data.rename(columns={'market_ids_fips':'market_ids_string'}, inplace=True)
    product_data['market_ids']=product_data['market_ids_string'].factorize()[0]
    print('6 product_data: ', product_data.shape)
    # Creacion de dataframe organizando por nivel de ingresos identificados 
    markets_characterization =product_data[['zip',
                          'market_ids_string',
                          'market_ids',
                          'total_income_market',
                          'total_income_market_known_brands',
                          'fraction_identified_earnings']].sort_values(by=['fraction_identified_earnings'], axis=0, ascending=False)
    
    # Creación de identificador numérico para los productos
    # product_data = product_data[(product_data['total_income_market_known_brands'] > 700) & (product_data['fraction_identified_earnings'] >0.4 )].reset_index()
    # product_data = product_data[(product_data['fraction_identified_earnings'] >= fractioned_identified_earning)].reset_index()
    # del product_data['index']
    product_data['product_ids'] = pd.factorize(product_data['brand_descr'])[0]
    print('7 product_data: ', product_data.shape)
    # Elimina productos con características no identificadas
    
    return product_data


def creating_instruments_data(product_data: pd.DataFrame):
    # Creación de instrumentos
    formulation = pyblp.Formulation('0 + tar + nicotine + co + nicotine_mg_per_g + nicotine_mg_per_g_dry_weight_basis + nicotine_mg_per_cig')
    blp_instruments = pyblp.build_blp_instruments(formulation, product_data)
    blp_instruments = pd.DataFrame(blp_instruments)
    blp_instruments.rename(columns={i:f'blp_instruments{i}' for i in blp_instruments.columns}, inplace=True)

    local_instruments = pyblp.build_differentiation_instruments(formulation, product_data)
    local_instruments = pd.DataFrame(local_instruments, columns=[f'local_instruments{i}' for i in range(local_instruments.shape[1])])

    quadratic_instruments = pyblp.build_differentiation_instruments(formulation, product_data, version='quadratic')
    quadratic_instruments = pd.DataFrame(quadratic_instruments, columns=[f'quadratic_instruments{i}' for i in range(quadratic_instruments.shape[1])])

    print(type(blp_instruments))
    print(type(local_instruments))
    print(type(quadratic_instruments))

    return formulation, blp_instruments, local_instruments, quadratic_instruments


def creating_agent_data(product_data: pd.DataFrame, 
                        record_layout_path: str, 
                        agent_data_path: str):
    
    output = process_file(record_layout_path)
    agent_data_pop = pd.read_fwf(agent_data_path, widths= [int(elem) for elem in output.values()] )

    column_names = output.keys()
    agent_data_pop.columns = column_names
    agent_data_pop=agent_data_pop[agent_data_pop['GTCO']!=0]
    agent_data_pop['FIPS'] = agent_data_pop['GESTFIPS']*1000 + agent_data_pop['GTCO']
    agent_data_pop.reset_index(inplace=True, drop=True)

    # product_data=product_data.rename(columns={'fip':'FIPS', 'fips_state_code':'GESTFIPS'})
    
    demographic_sample = get_random_samples_by_code(agent_data_pop, product_data['GESTFIPS'], 200)[['FIPS', 'GESTFIPS', 'HEFAMINC', 'PRTAGE', 'HRNUMHOU','PTDTRACE', 'PEEDUCA']]
    demographic_sample.replace(-1, np.nan, inplace=True)

    knn_imputer = KNNImputer(n_neighbors=2)
    demographic_sample_knn_imputed = pd.DataFrame(knn_imputer.fit_transform(demographic_sample[['HEFAMINC', 'PRTAGE', 'HRNUMHOU', 'PTDTRACE', 'PEEDUCA']]),
                                columns=['hefaminc_imputed', 'prtage_imputed', 'hrnumhou_imputed', 
                                        'ptdtrace_imputed', 'peeduca_imputed'])

    grouped = demographic_sample.groupby('GESTFIPS').size()

    demographic_sample['weights'] = demographic_sample['GESTFIPS'].map(1 / grouped)
    demographic_sample = pd.concat([demographic_sample[['FIPS', 'GESTFIPS','weights']],demographic_sample_knn_imputed], axis=1)
    demographic_sample = add_random_nodes(demographic_sample)

    demographic_sample = demographic_sample[['FIPS', 'GESTFIPS', 'weights',
                                            'nodes0', 'nodes1', 'nodes2', 'nodes3','nodes4',
                                            'hefaminc_imputed', 'prtage_imputed','hrnumhou_imputed', 
                                            'ptdtrace_imputed', 'peeduca_imputed']]
    
    agent_data = pd.merge(product_data[['market_ids', 'market_ids_string', 'GESTFIPS']].drop_duplicates(),
                                      demographic_sample, 
                                      how='inner', 
                                      left_on='GESTFIPS',
                                      right_on='GESTFIPS')

    return agent_data


def filtering_data_by_identified_sales(product_data: pd.DataFrame,
                                       blp_instruments: pd.DataFrame, 
                                       local_instruments: pd.DataFrame,
                                       quadratic_instruments: pd.DataFrame, 
                                       threshold_identified_earnings: float=0.4):
    
    condition = product_data['fraction_identified_earnings']>=threshold_identified_earnings
    kept_data = product_data.loc[condition].index

    product_data = product_data.loc[kept_data]

    local_instruments = local_instruments.loc[kept_data]
    quadratic_instruments = quadratic_instruments.loc[kept_data]
    blp_instruments = blp_instruments.loc[kept_data]

    product_data.reset_index(drop=True, inplace=True)
    blp_instruments.reset_index(drop=True, inplace=True)
    local_instruments.reset_index(drop=True, inplace=True)
    quadratic_instruments.reset_index(drop=True, inplace=True)

    return product_data, blp_instruments, local_instruments, quadratic_instruments


def filtering_data_by_number_brands(product_data: pd.DataFrame,
                                    blp_instruments: pd.DataFrame, 
                                    local_instruments: pd.DataFrame,
                                    quadratic_instruments: pd.DataFrame, 
                                    num_brands_by_market: int=2):
    """
    
    
    """
    market_counts = product_data['market_ids'].value_counts()
    valid_markets = market_counts[market_counts >= num_brands_by_market].index
    product_data = product_data[product_data['market_ids'].isin(valid_markets)]

    local_instruments = local_instruments.loc[product_data.index]
    quadratic_instruments = quadratic_instruments.loc[product_data.index]
    blp_instruments = blp_instruments.loc[product_data.index]

    product_data.reset_index(drop=True, inplace=True)
    blp_instruments.reset_index(drop=True, inplace=True)
    local_instruments.reset_index(drop=True, inplace=True)
    quadratic_instruments.reset_index(drop=True, inplace=True)

    return product_data, blp_instruments, local_instruments, quadratic_instruments


def matching_agent_and_product_data(product_data: pd.DataFrame, 
                                    agent_data: pd.DataFrame):
    agent_data = agent_data[agent_data['market_ids'].isin(set(product_data['market_ids']))]
    product_data = product_data[product_data['market_ids'].isin(agent_data['market_ids'].unique())]

    return agent_data, product_data


def save_processed_data(product_data, blp_instruments, local_instruments, quadratic_instruments, agent_data):
    week_dir = list(set(product_data['week_end']))[0] if len(set(product_data['week_end'])) == 1 else None
    os.makedirs(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/processed_data/{week_dir}', exist_ok=True)
    blp_instruments.to_csv(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/processed_data/{week_dir}/blp_instruments_{DIRECTORY_NAME}_{datetime.datetime.today()}.csv', index=False)
    local_instruments.to_csv(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/processed_data/{week_dir}/local_instruments_{DIRECTORY_NAME}_{datetime.datetime.today()}.csv', index=False)
    quadratic_instruments.to_csv(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/processed_data/{week_dir}/quadratic_instruments_{DIRECTORY_NAME}_{datetime.datetime.today()}.csv', index=False)
    agent_data.to_csv(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/processed_data/{week_dir}/agent_data_{DIRECTORY_NAME}_{datetime.datetime.today()}.csv', index=False)


def save_product_data(product_data: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    product_data.to_csv(os.path.join(output_dir, 'product_data.csv'), index=False)


def create_directories(product_data: pd.DataFrame):
    week_dir = list(set(product_data['week_end']))[0] if len(set(product_data['week_end'])) == 1 else None
    os.makedirs(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/Predicted/{week_dir}', exist_ok=True)
    os.makedirs(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/ProblemResults_class/pickle/{week_dir}', exist_ok=True)


def select_product_data_columns(product_data: pd.DataFrame) -> pd.DataFrame:

    return product_data[['market_ids', 'market_ids_string',
                        'store_code_uc', 'zip', 'FIPS', 'GESTFIPS', 'fips_county_code',
                        'week_end', 'week_end_ID',
                        'firm', 'firm_ids', 'firm_post_merger', 'firm_ids_post_merger', 'brand_code_uc', 'brand_descr', 'product_ids', 
                        'units', 'total_individual_units', 'total_units_retailer',
                        'shares', 'poblacion_census_2020',
                        'total_income', 'total_income_market_known_brands', 'total_income_market', 'fraction_identified_earnings',
                        'prices',
                        'from Nielsen', 'from characteristics', 'name',
                        'tar', 'nicotine', 'co', 'nicotine_mg_per_g', 'nicotine_mg_per_g_dry_weight_basis', 'nicotine_mg_per_cig']]


def compile_data(product_data: pd.DataFrame,
                          blp_inst: pd.DataFrame, 
                          local_inst: pd.DataFrame, 
                          quad_inst: pd.DataFrame, 
                          agent_data: pd.DataFrame):

    consolidated_product_data=pd.concat([product_data, local_inst], axis=1)
    dict_rename = rename_instruments(consolidated_product_data)
    consolidated_product_data=consolidated_product_data.rename(columns=dict_rename)

    # Restringe la información del consolidated_product_data a aquella que tienen información del consumidor en el agent_data
    consolidated_product_data = consolidated_product_data[consolidated_product_data['market_ids'].isin(agent_data['market_ids'].unique())]

    # Sort del product_data
    consolidated_product_data = consolidated_product_data.sort_values(by=['market_ids', 'product_ids'], ascending=[True, True], ignore_index=True)

    return consolidated_product_data


def run():
    product_data = creating_product_data_rcl(main_dir='/oak/stanford/groups/polinsky/Mergers/Cigarettes',
                                     movements_path='/oak/stanford/groups/polinsky/Mergers/Cigarettes/Nielsen_data/2014/Movement_Files/4510_2014/7460_2014.tsv' ,
                                     stores_path='Nielsen_data/2014/Annual_Files/stores_2014.tsv' ,
                                     products_path='Nielsen_data/Master_Files/Latest/products.tsv',
                                     extra_attributes_path='Nielsen_data/2014/Annual_Files/products_extra_2014.tsv', 
                                     first_week=16,
                                     num_weeks=1, 
                                     fractioned_identified_earning=0.34)
    
    optimization_algorithm = 'l-bfgs-b'

    product_data = select_product_data_columns(product_data=product_data)

    # Save product_data DataFrame to the specified directory
    output_dir = '/oak/stanford/groups/polinsky/Mergers/Cigarettes/Pruebas'
    os.makedirs(output_dir, exist_ok=True)
    product_data.to_csv(os.path.join(output_dir, 'product_data.csv'), index=False)

    # Crea directorio para guardar las predicciones
    week_dir = list(set(product_data['week_end']))[0] if len(set(product_data['week_end'])) == 1 else None
    os.makedirs(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/Predicted/{week_dir}/{optimization_algorithm}', exist_ok=True)
    os.makedirs(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/ProblemResults_class/pickle/{week_dir}/{optimization_algorithm}', exist_ok=True)

    ########## Creación de instrumentos ##########
    formulation = pyblp.Formulation('0 + tar + nicotine + co + nicotine_mg_per_g + nicotine_mg_per_g_dry_weight_basis + nicotine_mg_per_cig')
    blp_instruments = pyblp.build_blp_instruments(formulation, product_data)
    blp_instruments = pd.DataFrame(blp_instruments)
    blp_instruments.rename(columns={i:f'blp_instruments{i}' for i in blp_instruments.columns}, inplace=True)

    local_instruments = pyblp.build_differentiation_instruments(formulation, product_data)
    local_instruments = pd.DataFrame(local_instruments, columns=[f'local_instruments{i}' for i in range(local_instruments.shape[1])])

    quadratic_instruments = pyblp.build_differentiation_instruments(formulation, product_data, version='quadratic')
    quadratic_instruments = pd.DataFrame(quadratic_instruments, columns=[f'quadratic_instruments{i}' for i in range(quadratic_instruments.shape[1])])

    print(type(blp_instruments))
    print(type(local_instruments))
    print(type(quadratic_instruments))

    ####### Agregando información sociodemográfica #########
    # encoding_guessed = read_file_with_guessed_encoding('/oak/stanford/groups/polinsky/Nielsen_data/Mergers/Reynolds_Lorillard/otros/January_2014_Record_Layout.txt')
    output = process_file('/oak/stanford/groups/polinsky/Current_Population_Survey/2014/January_2014_Record_Layout.txt')
    agent_data_pop = pd.read_fwf('/oak/stanford/groups/polinsky/Current_Population_Survey/2014/apr14pub.dat', widths= [int(elem) for elem in output.values()] )
    column_names = output.keys()
    agent_data_pop.columns = column_names
    agent_data_pop=agent_data_pop[agent_data_pop['GTCO']!=0]
    agent_data_pop['FIPS'] = agent_data_pop['GESTFIPS']*1000 + agent_data_pop['GTCO']
    agent_data_pop.reset_index(inplace=True, drop=True)

    # product_data=product_data.rename(columns={'fip':'FIPS', 'fips_state_code':'GESTFIPS'})
    
    demographic_sample = get_random_samples_by_code(agent_data_pop, product_data['GESTFIPS'], 200)[['FIPS', 'GESTFIPS', 'HEFAMINC', 'PRTAGE', 'HRNUMHOU','PTDTRACE', 'PEEDUCA']]
    demographic_sample.replace(-1, np.nan, inplace=True)

    knn_imputer = KNNImputer(n_neighbors=2)
    demographic_sample_knn_imputed = pd.DataFrame(knn_imputer.fit_transform(demographic_sample[['HEFAMINC', 'PRTAGE', 'HRNUMHOU', 'PTDTRACE', 'PEEDUCA']]),
                                columns=['hefaminc_imputed', 'prtage_imputed', 'hrnumhou_imputed', 
                                        'ptdtrace_imputed', 'peeduca_imputed'])

    grouped = demographic_sample.groupby('GESTFIPS').size()

    demographic_sample['weights'] = demographic_sample['GESTFIPS'].map(1 / grouped)
    demographic_sample = pd.concat([demographic_sample[['FIPS', 'GESTFIPS','weights']],demographic_sample_knn_imputed], axis=1)
    demographic_sample = add_random_nodes(demographic_sample)

    demographic_sample = demographic_sample[['FIPS', 'GESTFIPS', 'weights',
                                            'nodes0', 'nodes1', 'nodes2', 'nodes3','nodes4',
                                            'hefaminc_imputed', 'prtage_imputed','hrnumhou_imputed', 
                                            'ptdtrace_imputed', 'peeduca_imputed']]
    
    agent_data = pd.merge(product_data[['market_ids', 'market_ids_string', 'GESTFIPS']].drop_duplicates(),
                                      demographic_sample, 
                                      how='inner', 
                                      left_on='GESTFIPS',
                                      right_on='GESTFIPS')
    
    # Restringiendo la muestra a los mercados que tienen cierto nivel de ventas identificadas
    print(product_data.shape)
    print(blp_instruments.shape)
    print(local_instruments.shape)
    print(quadratic_instruments.shape) 

    ##### Filtrar base a partir de ventas identificadas########
    condition = product_data['fraction_identified_earnings']>=0.4
    kept_data = product_data.loc[condition].index

    product_data = product_data.loc[kept_data]

    local_instruments = local_instruments.loc[kept_data]
    quadratic_instruments = quadratic_instruments.loc[kept_data]
    blp_instruments = blp_instruments.loc[kept_data]

    product_data.reset_index(drop=True, inplace=True)
    blp_instruments.reset_index(drop=True, inplace=True)
    local_instruments.reset_index(drop=True, inplace=True)
    quadratic_instruments.reset_index(drop=True, inplace=True)

    ####### Restringiendo la muestra a retailers que tienen 2 o más marcas identificadas ######## 
    market_counts = product_data['market_ids'].value_counts()
    valid_markets = market_counts[market_counts >= 2].index
    product_data = product_data[product_data['market_ids'].isin(valid_markets)]

    local_instruments = local_instruments.loc[product_data.index]
    quadratic_instruments = quadratic_instruments.loc[product_data.index]
    blp_instruments = blp_instruments.loc[product_data.index]

    product_data.reset_index(drop=True, inplace=True)
    blp_instruments.reset_index(drop=True, inplace=True)
    local_instruments.reset_index(drop=True, inplace=True)
    quadratic_instruments.reset_index(drop=True, inplace=True)

    ######### Manteniendo la información en agents y data con iguales market_ids ##########
    agent_data = agent_data[agent_data['market_ids'].isin(set(product_data['market_ids']))]
    product_data = product_data[product_data['market_ids'].isin(agent_data['market_ids'].unique())]

    ######### Salvando datos ###########
    os.makedirs(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/processed_data/{week_dir}', exist_ok=True)
    blp_instruments.to_csv(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/processed_data/{week_dir}/blp_instruments_{DIRECTORY_NAME}_{datetime.datetime.today()}.csv', index=False)
    local_instruments.to_csv(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/processed_data/{week_dir}/local_instruments_{DIRECTORY_NAME}_{datetime.datetime.today()}.csv', index=False)
    quadratic_instruments.to_csv(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/processed_data/{week_dir}/quadratic_instruments_{DIRECTORY_NAME}_{datetime.datetime.today()}.csv', index=False)
    agent_data.to_csv(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/processed_data/{week_dir}/agent_data_{DIRECTORY_NAME}_{datetime.datetime.today()}.csv', index=False)

    product_data = compile_data(product_data = product_data, 
                            blp_inst = blp_instruments, 
                            local_ints = local_instruments, 
                            quad_inst = quadratic_instruments)
    
    iter =  0
    while iter <= 20:
        print('------------------------------')
        print(iter)
        print('------------------------------')
        try:
            results= rcl_with_demographics(product_data=product_data, agent_data=agent_data)

            optimal_results = results_optimal_instruments(results=results)

            optimal_results.to_pickle(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/ProblemResults_class/pickle/{week_dir}/{optimization_algorithm}/iteration_{iter}.pickle')
            
            predicted_prices = predict_prices(product_data = product_data, results = optimal_results, merger=[3,0])

            predicted_prices = predicted_prices.tolist()
            price_pred_df = product_data[['market_ids','market_ids_string','store_code_uc', 'week_end', 'product_ids', 'brand_code_uc', 'brand_descr']].copy()
            price_pred_df.loc[:, 'price_prediction'] = predicted_prices 
            price_pred_df.to_json(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/Predicted/{week_dir}/{optimization_algorithm}/price_predictions_{iter}.json', index=False)

            # optimal_results = results_optimal_instruments(results)
        except Exception as e:
            print(e)
            

        iter += 1
        

    print('fin')



def run2():
    product_data = creating_product_data_rcl(main_dir='/oak/stanford/groups/polinsky/Mergers/Cigarettes',
                                     movements_path='/oak/stanford/groups/polinsky/Mergers/Cigarettes/Nielsen_data/2014/Movement_Files/4510_2014/7460_2014.tsv' ,
                                     stores_path='Nielsen_data/2014/Annual_Files/stores_2014.tsv' ,
                                     products_path='Nielsen_data/Master_Files/Latest/products.tsv',
                                     extra_attributes_path='Nielsen_data/2014/Annual_Files/products_extra_2014.tsv', 
                                     first_week=15,
                                     num_weeks=1, 
                                     fractioned_identfied_earning=0.3)
    
    week_dir = list(set(product_data['week_end']))[0] if len(set(product_data['week_end'])) == 1 else None


    ######### Save product_data DataFrame to the specified directory ###########
    save_product_data(product_data, '/oak/stanford/groups/polinsky/Mergers/Cigarettes/Pruebas')

    ########## Creación de instrumentos ##########
    formulation, blp_instruments, local_instruments, quadratic_instruments = creating_instruments_data(product_data=product_data)

    ####### Agregando información sociodemográfica #########
    agent_data = creating_agent_data(product_data=product_data, 
                                    record_layout_path='/oak/stanford/groups/polinsky/Current_Population_Survey/2014/January_2014_Record_Layout.txt',
                                    agent_data_pop='/oak/stanford/groups/polinsky/Current_Population_Survey/2014/apr14pub.dat')

    ##### Filtrar base a partir de ventas identificadas########
    product_data, blp_instruments, local_instruments, quadratic_instruments = filtering_data_by_identified_sales(product_data= product_data,
                                                                                                                 blp_instruments=blp_instruments,
                                                                                                                 local_instruments= local_instruments,
                                                                                                                 quadratic_instruments=quadratic_instruments,
                                                                                                                 threshold_identified_earnings=0.3)

    ####### Restringiendo la muestra a retailers que tienen 2 o más marcas identificadas ######## 
    product_data, blp_instruments, local_instruments, quadratic_instruments = filtering_data_by_number_brands(product_data=product_data,
                                                                                                              blp_instruments=blp_instruments, 
                                                                                                              local_instruments=local_instruments,
                                                                                                              quadratic_instruments=quadratic_instruments,
                                                                                                              num_brands_by_market = 2)    

    ######### Manteniendo la información en agents y data con iguales market_ids ##########
    agent_data, product_data = matching_agent_and_product_data(product_data = product_data,
                                                               agent_data = agent_data)

    ######### Salvando datos ###########
    create_directories(product_data)
    save_processed_data(blp_instruments, local_instruments, quadratic_instruments, agent_data)

    ###### Random coefficients logit ########
    iter =  0
    while iter <= 20:
        print('------------------------------')
        print(iter)
        print('------------------------------')
        try:
            results, consolidated_product_data = rcl_with_demographics(product_data=product_data,
                                                                        blp_inst=blp_instruments,
                                                                        local_inst=local_instruments,
                                                                        quad_inst=quadratic_instruments,
                                                                        agent_data=agent_data)
            # if results.converged == True:
            #     iter += 1

            optimal_results = results_optimal_instruments(results=results)

            optimal_results.to_pickle(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/ProblemResults_class/pickle/{week_dir}/iteration_{iter}.pickle')
            
            predicted_prices = predict_prices(product_data = consolidated_product_data,
                                                results = results, 
                                                merger=[3,0])

            # predicted_prices_path = f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/Predicted/{week_dir}/iteration_{iter}.json'
            predicted_prices = predicted_prices.tolist()
            price_pred_df = consolidated_product_data[['market_ids','market_ids_string','store_code_uc', 'week_end', 'product_ids', 'brand_code_uc', 'brand_descr']].copy()
            price_pred_df.loc[:, 'price_prediction'] = predicted_prices 
            price_pred_df.to_json(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/Predicted/{week_dir}/price_predictions_{iter}.json', index=False)

            # optimal_results = results_optimal_instruments(results)
        except Exception as e:
            print(e)
            

        iter += 1
        

    print('fin')


if __name__=='__main__': 
    run()
