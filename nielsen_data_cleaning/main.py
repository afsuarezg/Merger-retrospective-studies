import os
import pandas as pd
import datetime
import pyblp


# from dotenv import load_dotenv

from .descarga_merge import movements_file, stores_file, products_file, extra_attributes_file, retail_market_ids_fips, retail_market_ids_identifier
from .caracteristicas_productos import match_brands_to_characteristics, list_of_files, characteristics
from .empresas import find_company, brands_by_company
# from .filtrar_mercados import *
# from .informacion_poblacional import *
# from .instrumentos import *
from .precios_ingresos_participaciones import total_income, total_units, unitary_price, price, fraccion_ventas_identificadas, prepend_zeros

# # load_dotenv()
# # github_repo_key = os.getenv('github_repo_token')

DIRECTORY_NAME = 'Reynolds_Lorillard'
DEPARTMENT_CODE = 4510 #aka product_group_code
# PRODUCT_MODULE = 7460
# NROWS = 10000000
YEAR = 2014
# WEEKS = [20140125, 20140201]


def run():
    os.chdir(f'/oak/stanford/groups/polinsky/Nielsen_data/Mergers/{DIRECTORY_NAME}/nielsen_extracts/RMS/{YEAR}/Movement_Files/{DEPARTMENT_CODE}_{YEAR}/')

    movements_data = movements_file()
    stores_data = stores_file()
    products_data = products_file()
    extra_attributes_data = extra_attributes_file(movements_data)

    product_data = pd.merge(movements_data, stores_data, on='store_code_uc', how='left')
    product_data = pd.merge(product_data, products_data, on='upc', how='left')
    product_data = pd.merge(product_data, extra_attributes_data, on='upc', how='left')

    product_data['week_end_ID'] = pd.factorize(product_data['week_end'])[0]
    product_data['market_ids'] = product_data.apply(retail_market_ids_identifier, axis=1)
    product_data['market_ids_fips'] = product_data.apply(retail_market_ids_fips, axis=1)
    product_data['firm_ids'] = None

    product_data = product_data[['store_code_uc','market_ids','market_ids_fips','store_zip3','week_end','week_end_ID',#mercado tiempo y espacio
                        'fips_state_code', 'fips_state_descr', 'fips_county_code', 'fips_county_descr',
                        'upc','firm_ids', 'brand_code_uc','brand_descr', ##companía y marca
                        'units', 'multi', 'price', 'prmult', #cantidades y precio 
                        'style_code','style_descr', 'type_code', 'type_descr','strength_code','strength_descr']]# características del producto

    product_data['brand_descr'] = product_data['brand_descr'].fillna('Not_identified')
    product_data['total_income'] = product_data.apply(total_income, axis=1)
    product_data['total_individual_units'] = product_data.apply(total_units, axis=1)
    product_data['unitary_price'] = product_data.apply(unitary_price, axis=1)

    product_data = product_data[['store_code_uc', 'market_ids', 'market_ids_fips',  'store_zip3', 'week_end', 'week_end_ID',
                             'fips_state_code', 'fips_state_descr', 'fips_county_code', 'fips_county_descr',
       'upc', 'firm_ids', 'brand_code_uc', 'brand_descr', 
       'units', 'multi', 'price', 'prmult','unitary_price', 'total_income',
       'total_individual_units',
       'style_code', 'style_descr', 'type_code','type_descr', 'strength_code', 'strength_descr']]
    
    product_data.rename(columns={'store_zip3':'zip'}, inplace=True)

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
    product_data['prices'] = product_data.apply(price, axis=1)

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
    product_data['total_income_market_known_brands'].fillna(0.0, inplace=True)
    product_data['fraction_identified_earnings'] = product_data.apply(fraccion_ventas_identificadas, axis=1)
    total_sold_units_per_marketid = pd.DataFrame(product_data.groupby(by=['market_ids',
                                                                 'store_code_uc'], as_index=False).agg({'units': 'sum'}))
    total_sold_units_per_marketid.rename(columns={'units':'total_units_retailer'}, inplace=True)
    product_data = product_data.merge(total_sold_units_per_marketid, 
                                  how ='left',
                                  on=['market_ids','store_code_uc'])
    product_data = product_data[product_data['brand_code_uc'].notna()]

    fips_pop= pd.read_excel('/oak/stanford/groups/polinsky/Tamaño_mercado/PopulationEstimates.xlsx', skiprows=4)
    fips_pop=fips_pop[['FIPStxt','State','CENSUS_2020_POP']]

    fips_pop['FIPS'] = fips_pop['FIPStxt'].astype('int')
    fips_pop['FIPStxt']=fips_pop['FIPStxt'].astype(str)
    product_data['fip'] = product_data.apply(prepend_zeros, axis=1).astype('int')
    fips_pop = fips_pop.rename(columns={'FIPS': 'fip'})
    product_data=product_data.merge(fips_pop[['CENSUS_2020_POP','fip']], how='left', on='fip')

    product_data = product_data[['market_ids', 'store_code_uc', 'zip','fip', 'week_end', 'week_end_ID',
       'market_ids_fips',  'fips_state_code', 'fips_state_descr', 'fips_county_code', 'fips_county_descr', 
       'firm_ids', 'brand_code_uc','brand_descr', 
       'units',  'prices', 'unitary_price_x_reemplazar','price_x_reemplazar',
       'total_individual_units',  'total_units_retailer',
       'style_code', 'style_descr', 'type_code', 'type_descr', 'strength_code', 'strength_descr',
       'total_income','total_income_market', 'total_income_market_known_brands',
       'fraction_identified_earnings',  
       'CENSUS_2020_POP']]
    product_data.rename(columns={'CENSUS_2020_POP':'poblacion_census_2020'}, inplace=True)



    product_data['firm']=product_data.apply(find_company, axis=1)
    product_data['firm_ids']=(pd.factorize(product_data['firm']))[0]


    product_data['brand_descr']=product_data['brand_descr'].str.lower()
    match_bases_datos = match_brands_to_characteristics(product_data, characteristics, threshold=85)
    characteristics_matches=pd.DataFrame.from_dict(match_bases_datos)
    product_data = product_data.merge(characteristics_matches, how='left', left_on='brand_descr', right_on='from_nielsen')
    product_data = product_data.merge(characteristics, how='left', left_on='from_characteristics', right_on='name')
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
    product_data = product_data[product_data['name'].notna()]

    product_data.rename(columns={'market_ids_fips':'market_ids_string'}, inplace=True)
    product_data['market_ids']=product_data['market_ids_string'].factorize()[0]
    markets_characterization =product_data[['zip',
                          'market_ids_string',
                          'market_ids',
                          'total_income_market',
                          'total_income_market_known_brands',
                          'fraction_identified_earnings']].sort_values(by=['fraction_identified_earnings'],
                                                                                         axis=0,
                                                                                         ascending=False)
    product_data = product_data[(product_data['total_income_market_known_brands'] > 700) & (product_data['fraction_identified_earnings'] >0.4 )].reset_index()
    del product_data['index']
    product_data['market_ids']=product_data['market_ids_string'].factorize()[0]
    product_data['product_ids'] = pd.factorize(product_data['brand_descr'])[0]


    product_data = product_data.dropna(subset=['tar', 'nicotine', 'co',
       'nicotine_mg_per_g', 'nicotine_mg_per_g_dry_weight_basis',
       'nicotine_mg_per_cig'])
    
    formulation = pyblp.Formulation('0 + tar + nicotine + co + nicotine_mg_per_g + nicotine_mg_per_g_dry_weight_basis + nicotine_mg_per_cig')
    blp_instruments = pyblp.build_blp_instruments(formulation, product_data)
    blp_instruments = pd.DataFrame(blp_instruments)
    blp_instruments.rename(columns={i:f'blp_instruments{i}' for i in blp_instruments.columns}, inplace=True)

    local_instruments = pyblp.build_differentiation_instruments(
    formulation,
    product_data)
    local_instruments = pyblp.build_differentiation_instruments(formulation, product_data)
    local_instruments = pd.DataFrame(local_instruments, columns=[f'local_instruments{i}' for i in range(local_instruments.shape[1])])

    quadratic_instruments = pyblp.build_differentiation_instruments(formulation, product_data, version='quadratic')
    quadratic_instruments = pd.DataFrame(quadratic_instruments, columns=[f'quadratic_instruments{i}' for i in range(quadratic_instruments.shape[1])])

    nivel_de_agregacion = 'retailer'
    product_data.to_csv(f'/oak/stanford/groups/polinsky/Mergers/cigarettes/processed_data/product_data_{nivel_de_agregacion}_{DIRECTORY_NAME}_{datetime.datetime.today()}.csv', index=False)

    print('fin')


if __name__=='__main__':
    run()
