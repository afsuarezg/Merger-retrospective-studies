import os
import pandas as pd
import datetime
import pyblp
import numpy as np
import sys

# from dotenv import load_dotenv

from .nielsen_data_cleaning.descarga_merge import movements_file, stores_file, products_file, extra_attributes_file, retail_market_ids_fips, retail_market_ids_identifier
from .nielsen_data_cleaning.caracteristicas_productos import match_brands_to_characteristics, list_of_files, characteristics
from .nielsen_data_cleaning.empresas import find_company, brands_by_company
from .nielsen_data_cleaning.consumidores_sociodemograficas import read_file_with_guessed_encoding, process_file, get_random_samples_by_code, KNNImputer, add_random_nodes
from .nielsen_data_cleaning.precios_ingresos_participaciones import total_income, total_units, unitary_price, price, fraccion_ventas_identificadas, prepend_zeros, shares_with_outside_good
from .estimaciones.plain_logit import plain_logit
from .estimaciones.rcl_without_demographics import rcl_without_demographics
from .estimaciones.rcl_with_demographics import rcl_with_demographics


DIRECTORY_NAME = 'Reynolds_Lorillard'
DEPARTMENT_CODE = 4510 #aka product_group_code
# PRODUCT_MODULE = 7460
# NROWS = 10000000
YEAR = 2014
# WEEKS = [20140125, 20140201]

def run():
    os.chdir(f'/oak/stanford/groups/polinsky/Nielsen_data/Mergers/{DIRECTORY_NAME}/nielsen_extracts/RMS/{YEAR}/Movement_Files/{DEPARTMENT_CODE}_{YEAR}/')

    # Descarga los datos
    movements_data = movements_file()
    stores_data = stores_file()
    products_data = products_file()
    extra_attributes_data = extra_attributes_file(movements_data)

    # Combina los datos
    product_data = pd.merge(movements_data, stores_data, on='store_code_uc', how='left')
    product_data = pd.merge(product_data, products_data, on='upc', how='left')
    product_data = pd.merge(product_data, extra_attributes_data, on='upc', how='left')

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
    total_sales_identified_per_marketid = pd.DataFrame(product_data[product_data['brand_descr']!='Not_identified'].groupby(by=['market_ids','store_code_uc'],
                            as_index=False).agg({'total_income': 'sum'}))
    total_sales_identified_per_marketid = total_sales_identified_per_marketid.rename(columns={'total_income':'total_income_market_known_brands'})
    product_data = product_data.merge(total_sales_per_marketid, 
                                    how ='left',
                                    on=['market_ids','store_code_uc'])
    product_data = product_data.merge(total_sales_identified_per_marketid, 
                                  how='left',
                                  on=['market_ids','store_code_uc'])
    product_data.fillna({'total_income_market_known_brands': 0.0}, inplace=True)
    # product_data['total_income_market_known_brands'].fillna(0.0, inplace=True)
    product_data['fraction_identified_earnings'] = product_data.apply(fraccion_ventas_identificadas, axis=1)

    # Suma total de unidades vendidas por tienda 
    total_sold_units_per_marketid = pd.DataFrame(product_data.groupby(by=['market_ids',
                                                                 'store_code_uc'], as_index=False).agg({'units': 'sum'}))
    total_sold_units_per_marketid.rename(columns={'units':'total_units_retailer'}, inplace=True)
    product_data = product_data.merge(total_sold_units_per_marketid, 
                                  how ='left',
                                  on=['market_ids','store_code_uc'])

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
   
    # Calculo de participaciones de mercado incluyendo la participación del bien externo. 
    product_data['shares']=product_data.apply(shares_with_outside_good, axis=1)    
    product_data.rename(columns={'CENSUS_2020_POP':'poblacion_census_2020'}, inplace=True)

    # Asignación de las marcas por empresa 
    product_data['firm']=product_data.apply(find_company, axis=1)
    product_data['firm_ids']=(pd.factorize(product_data['firm']))[0]

    # Adición de información sobre características de los productos
    product_data['brand_descr']=product_data['brand_descr'].str.lower()
    match_bases_datos = match_brands_to_characteristics(product_data, characteristics, threshold=85)
    characteristics_matches=pd.DataFrame.from_dict(match_bases_datos)
    product_data = product_data.merge(characteristics_matches, how='left', left_on='brand_descr', right_on='from_nielsen')
    product_data = product_data.merge(characteristics, how='left', left_on='from_characteristics', right_on='name')
    product_data = product_data[product_data['name'].notna()]
    
    # Organizando el dataframe
    product_data = product_data[['market_ids', 'market_ids_fips',
                             #variables relativas a la ubicacion
                             'store_code_uc', 'zip', 'fip', 'fips_state_code',  'fips_county_code', #'fips_county_descr',  'fips_state_descr',
                             #variables relativas al tiempo
                             'week_end', 'week_end_ID', 
                             #variables relativas a la compania y a la marca
                             'firm', 'firm_ids', 'brand_code_uc', 'brand_descr',
                             #varibles relativas a cantidades
                             'units', 'total_individual_units', 'total_units_retailer',
                             #variables relativas a partipación del mercado 
                             'shares', 'poblacion_census_2020',
                             #variables relativas a ingresos totales
                             'total_income', 'total_income_market_known_brands','total_income_market', 'fraction_identified_earnings',
                             #variablers relagtivas a los precios 
                             'prices',
                             #variables relativas a caracteristicas del producto obtenidas de Nielsen      
#                            'style_code', 'style_descr', 'type_code', 'type_descr', 'strength_code', 'strength_descr', 
                             #variables para hacer el merge con las características de los productos
                            'from_nielsen', 'from_characteristics', 'name', 
                             #relativas a características del producto obtenidas de otras fuentes, incluyendo la necesaria para hacer el merge con la base de las características  
                             'tar', 'nicotine', 'co', 'nicotine_mg_per_g', 'nicotine_mg_per_g_dry_weight_basis', 'nicotine_mg_per_cig']]

    # Cambio del nombre de IDS de mercados y genera indicador para numérico para estos 
    product_data.rename(columns={'market_ids_fips':'market_ids_string'}, inplace=True)
    product_data['market_ids']=product_data['market_ids_string'].factorize()[0]
    
    # Creacion de dataframe organizando por nivel de ingresos identificados 
    markets_characterization =product_data[['zip',
                          'market_ids_string',
                          'market_ids',
                          'total_income_market',
                          'total_income_market_known_brands',
                          'fraction_identified_earnings']].sort_values(by=['fraction_identified_earnings'], axis=0, ascending=False)
    
    # Creación de identificador numérico para los productos
    # product_data = product_data[(product_data['total_income_market_known_brands'] > 700) & (product_data['fraction_identified_earnings'] >0.4 )].reset_index()
    product_data['product_ids'] = pd.factorize(product_data['brand_descr'])[0]

    # Elimina productos con características no identificadas
    product_data = product_data.dropna(subset=['tar', 'nicotine', 'co',
       'nicotine_mg_per_g', 'nicotine_mg_per_g_dry_weight_basis',
       'nicotine_mg_per_cig'])
    
    # Creación de instrumentos
    formulation = pyblp.Formulation('0 + tar + nicotine + co + nicotine_mg_per_g + nicotine_mg_per_g_dry_weight_basis + nicotine_mg_per_cig')
    blp_instruments = pyblp.build_blp_instruments(formulation, product_data)
    blp_instruments = pd.DataFrame(blp_instruments)
    blp_instruments.rename(columns={i:f'blp_instruments{i}' for i in blp_instruments.columns}, inplace=True)

    local_instruments = pyblp.build_differentiation_instruments(formulation, product_data)
    local_instruments = pyblp.build_differentiation_instruments(formulation, product_data)
    local_instruments = pd.DataFrame(local_instruments, columns=[f'local_instruments{i}' for i in range(local_instruments.shape[1])])

    quadratic_instruments = pyblp.build_differentiation_instruments(formulation, product_data, version='quadratic')
    quadratic_instruments = pd.DataFrame(quadratic_instruments, columns=[f'quadratic_instruments{i}' for i in range(quadratic_instruments.shape[1])])

    print(type(blp_instruments))
    print(type(local_instruments))
    print(type(quadratic_instruments))

    # Agregando información sociodemográfica
    # encoding_guessed = read_file_with_guessed_encoding('/oak/stanford/groups/polinsky/Nielsen_data/Mergers/Reynolds_Lorillard/otros/January_2014_Record_Layout.txt')
    output = process_file('/oak/stanford/groups/polinsky/Nielsen_data/Mergers/Reynolds_Lorillard/otros/January_2014_Record_Layout.txt')
    agent_data_pop = pd.read_fwf('/oak/stanford/groups/polinsky/Nielsen_data/Mergers/Reynolds_Lorillard/apr14pub.dat', widths= [int(elem) for elem in output.values()] )
    column_names = output.keys()
    agent_data_pop.columns = column_names
    agent_data_pop=agent_data_pop[agent_data_pop['GTCO']!=0]
    agent_data_pop['FIPS'] = agent_data_pop['GESTFIPS']*1000 + agent_data_pop['GTCO']
    agent_data_pop.reset_index(inplace=True, drop=True)

    product_data=product_data.rename(columns={'fip':'FIPS', 'fips_state_code':'GESTFIPS'})
    
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
    
    product_data = product_data.reset_index()
    
    # Restringiendo la muestra a los mercados que tienen cierto nivel de ventas identificadas
    print(product_data.shape)
    print(blp_instruments.shape)
    print(local_instruments.shape)
    print(quadratic_instruments.shape) 

    # Check if the index is sequential for all the four previous data frames
    is_sequential_product_data = (product_data.index == range(len(product_data))).all()
    is_sequential_blp_instruments = (blp_instruments.index == range(len(blp_instruments))).all()
    is_sequential_local_instruments = (local_instruments.index == range(len(local_instruments))).all()
    is_sequential_quadratic_instruments = (quadratic_instruments.index == range(len(quadratic_instruments))).all()

    print("Is the product_data index sequential?", is_sequential_product_data)
    print("Is the blp_instruments index sequential?", is_sequential_blp_instruments)
    print("Is the local_instruments index sequential?", is_sequential_local_instruments)
    print("Is the quadratic_instruments index sequential?", is_sequential_quadratic_instruments)

    condition = product_data['fraction_identified_earnings']>=0.5
    kept_data = product_data.loc[condition].index

    product_data = product_data.loc[kept_data]

    local_instruments = local_instruments.loc[kept_data]
    quadratic_instruments = quadratic_instruments.loc[kept_data]
    blp_instruments = blp_instruments.loc[kept_data]

    product_data.reset_index(drop=True, inplace=True)
    blp_instruments.reset_index(drop=True, inplace=True)
    local_instruments.reset_index(drop=True, inplace=True)
    quadratic_instruments.reset_index(drop=True, inplace=True)

    # Restringiendo la muestra a retailers que tienen 2 o más marcas identificadas. 
    # Keep rows by 'market_ids' if they contain 2 or more samples
    market_counts = product_data['market_ids'].value_counts()
    valid_markets = market_counts[market_counts >= 2].index
    product_data = product_data[product_data['market_ids'].isin(valid_markets)]

    local_instruments = local_instruments[product_data.index]
    quadratic_instruments = quadratic_instruments[product_data.index]
    blp_instruments = blp_instruments[product_data.index]

    product_data.reset_index(drop=True, inplace=True)
    blp_instruments.reset_index(drop=True, inplace=True)
    local_instruments.reset_index(drop=True, inplace=True)
    quadratic_instruments.reset_index(drop=True, inplace=True)

    # Salvando datos
    nivel_de_agregacion = 'retailer'
    product_data.to_csv(f'/oak/stanford/groups/polinsky/Mergers/cigarettes/processed_data/product_data_{nivel_de_agregacion}_{DIRECTORY_NAME}_{datetime.datetime.today()}.csv', index=False)
    blp_instruments.to_csv(f'/oak/stanford/groups/polinsky/Mergers/cigarettes/processed_data/blp_instruments_{nivel_de_agregacion}_{DIRECTORY_NAME}_{datetime.datetime.today()}.csv', index=False)
    local_instruments.to_csv(f'/oak/stanford/groups/polinsky/Mergers/cigarettes/processed_data/local_instruments_{nivel_de_agregacion}_{DIRECTORY_NAME}_{datetime.datetime.today()}.csv', index=False)
    quadratic_instruments.to_csv(f'/oak/stanford/groups/polinsky/Mergers/cigarettes/processed_data/quadratic_instruments_{nivel_de_agregacion}_{DIRECTORY_NAME}_{datetime.datetime.today()}.csv', index=False)
    agent_data.to_csv(f'/oak/stanford/groups/polinsky/Mergers/cigarettes/processed_data/agent_data_{nivel_de_agregacion}_{DIRECTORY_NAME}_{datetime.datetime.today()}.csv', index=False)

    plain_logit(product_data = product_data, inst_data = local_instruments)
    
    rcl_without_demographics(product_data=product_data,
                             blp_inst=blp_instruments,
                             local_inst=local_instruments,
                             quad_inst=quadratic_instruments)
    
    rcl_with_demographics(product_data=product_data,
                             blp_inst=blp_instruments,
                             local_inst=local_instruments,
                             quad_inst=quadratic_instruments)
    
    print('fin')




if __name__=='__main__':
    run()
