import os
import pandas as pd

from dotenv import load_dotenv

from .descarga_merge import movements_file, stores_file, products_file, extra_attributes_file, retail_market_ids_fips, retail_market_ids_identifier
from .caracteristicas_productos import match_brands_to_characteristics, list_of_files
from .empresas import find_company
# from .filtrar_mercados import *
# from .informacion_poblacional import *
# from .instrumentos import *
from .precios_ingresos_participaciones import total_income, total_units, unitary_price, price, fraccion_ventas_identificadas

# load_dotenv()
# github_repo_key = os.getenv('github_repo_token')


def prueba():
    return 'listo'


def run():
    DIRECTORY_NAME = 'Reynolds_Lorillard'
    DEPARTMENT_CODE = 4510 #aka product_group_code
    # PRODUCT_MODULE = 7460
    # NROWS = 10000000
    YEAR = 2014
    # WEEKS = [20140125, 20140201]
    os.chdir(f'/oak/stanford/groups/polinsky/Nielsen_data/Mergers/{DIRECTORY_NAME}/nielsen_extracts/RMS/{YEAR}/Movement_Files/{DEPARTMENT_CODE}_{YEAR}/')

    movements_file = movements_file()
    stores_file = stores_file()
    products_file = products_file()
    extra_attributes_file = extra_attributes_file()

    product_data = pd.merge(movements_file, stores_file, on='store_code_uc', how='left')
    product_data = pd.merge(product_data, products_file, on='upc', how='left')
    product_data = pd.merge(product_data, extra_attributes_file, on='upc', how='left')

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

    nivel_de_agregacion = 'retailer'
    product_data.to_csv(f'/oak/stanford/groups/polinsky/Nielsen_data/Mergers/Reynolds_Lorillard/analisis/1.compiled_{nivel_de_agregacion}_{DIRECTORY_NAME}_{datetime.datetime.today()}.csv', index=False)

    print('fin')


if __name__=='__main__':
    run()
