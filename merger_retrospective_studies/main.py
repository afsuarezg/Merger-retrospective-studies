import os
import pandas as pd
import datetime
import pyblp

import sys
import json
from itertools import chain


# from dotenv import load_dotenv

from .nielsen_data_cleaning.descarga_merge import movements_file, stores_file, products_file, extra_attributes_file, retail_market_ids_fips, retail_market_ids_identifier, filter_row_weeks
from .nielsen_data_cleaning.caracteristicas_productos import match_brands_to_characteristics, list_of_files, characteristics
from .nielsen_data_cleaning.empresas import find_company_pre_merger, find_company_post_merger, brands_by_company_pre_merger, brands_by_company_post_merger
from .nielsen_data_cleaning.consumidores_sociodemograficas import read_file_with_guessed_encoding, process_file, get_random_samples_by_code, KNNImputer, add_random_nodes, creating_agent_data
from .nielsen_data_cleaning.precios_ingresos_participaciones import total_income, total_units, unitary_price, price, fraccion_ventas_identificadas, prepend_zeros, shares_with_outside_good
from .nielsen_data_cleaning.product_data_creation import creating_product_data_for_comparison, creating_product_data_rcl, compile_data, creating_instruments_data
from .estimaciones.plain_logit import plain_logit
from .estimaciones.rcl_without_demographics import rcl_without_demographics, rename_instruments
from .estimaciones.rcl_with_demographics import rcl_with_demographics
from .estimaciones.estimaciones_utils import save_dict_json
from .estimaciones.post_estimation_merger_simulation import predict_prices, original_prices
from .estimaciones.optimal_instruments import results_optimal_instruments
from .nielsen_data_cleaning.utils import create_output_directories, create_agent_data_from_cps, create_agent_data_sample, filter_by_identified_earnings, filter_by_market_size, filter_matching_markets, create_instruments, save_processed_data, create_formulations, check_matrix_collinearity, find_first_non_collinear_matrix, select_product_data_columns, filtering_data_by_identified_sales, filtering_data_by_number_brands, matching_agent_and_product_data, create_directories, save_product_data


DIRECTORY_NAME = 'Reynolds_Lorillard'
DEPARTMENT_CODE = 4510 #aka product_group_code
# PRODUCT_MODULE = 7460
# NROWS = 10000000
YEAR = 2014
# WEEKS = [20140125, 20140201]

#TODO: organizar las funciones de Nielsen_data_cleaning dependiendo de la base que está procesando. Las que se usen en diferentes bases deberían ir en un archivo más general. 


def main():
    date = datetime.datetime.today().strftime("%Y-%m-%d")
    datetime_=datetime.datetime.today()
    year=2014
    first_week=20
    num_weeks=2
    threshold_identified_earnings = 0.35
    optimization_algorithm = 'l-bfgs-b'

    #-----------------------------------------------------------------
    product_data = creating_product_data_rcl(main_dir='/oak/stanford/groups/polinsky/Mergers/Cigarettes',
                                     movements_path=f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/Nielsen_data/{year}/Movement_Files/4510_{year}/7460_{year}.tsv',
                                     stores_path='Nielsen_data/2014/Annual_Files/stores_2014.tsv',
                                     products_path='Nielsen_data/Master_Files/Latest/products.tsv',
                                     extra_attributes_path='Nielsen_data/2014/Annual_Files/products_extra_2014.tsv', 
                                     first_week=first_week,
                                     num_weeks=num_weeks,
                                     lower_threshold_identified_sales=threshold_identified_earnings)
    print('product_data columns: ', product_data.columns)
    #-----------------------------------------------------------------
    # product_data = product_data[product_data['fraction_identified_earnings']>=threshold_identified_earnings]
    # market_counts = product_data['market_ids'].value_counts()
    # valid_markets = market_counts[market_counts >= 2].index
    # product_data = product_data[product_data['market_ids'].isin(valid_markets)]

    #-----------------------------------------------------------------
    product_data = select_product_data_columns(product_data=product_data)
   
    #-------------------------------------------------------------
    # Crea directorios para guardar los datos procesados, las predicciones y los resultados de la optimización.
    week_dir = create_output_directories(product_data=product_data, date=date,optimization_algorithm=optimization_algorithm)

    #-----------------------------------------------------------------
    # Agregando información sociodemográfica 
    pop_agent_data = create_agent_data_from_cps(record_layout_path='/oak/stanford/groups/polinsky/Current_Population_Survey/2014/January_2014_Record_Layout.txt',
                                                agent_data_path='/oak/stanford/groups/polinsky/Current_Population_Survey/2014/apr14pub.dat')

    # socdem_file_structure = process_file('/oak/stanford/groups/polinsky/Current_Population_Survey/2014/January_2014_Record_Layout.txt')
    # agent_data_pop = pd.read_fwf('/oak/stanford/groups/polinsky/Current_Population_Survey/2014/apr14pub.dat', widths= [int(elem) for elem in socdem_file_structure.values()])
    # agent_data_pop.columns  = socdem_file_structure.keys()
    
    # # Elimina observaciones de los consumidores para los que el valor de condado es igual a 0.
    # agent_data_pop=agent_data_pop[agent_data_pop['GTCO']!=0]

    # # Genera la variable FIPS a partir de las variables GESTFIPS y GTCO.
    # agent_data_pop['FIPS'] = agent_data_pop['GESTFIPS']*1000 + agent_data_pop['GTCO']

    # # Resetea el índice del agent_data_pop.
    # agent_data_pop.reset_index(inplace=True, drop=True)

    # product_data=product_data.rename(columns={'fip':'FIPS', 'fips_state_code':'GESTFIPS'})
    


    #-----------------------------------------------------------------
    sample_agent_data = create_agent_data_sample(agent_data_pop=pop_agent_data, product_data=product_data)
    # agent_data_sample = get_random_samples_by_code(agent_data_pop, product_data['GESTFIPS'].unique(), 400)[['FIPS', 'GESTFIPS', 'HEFAMINC', 'PRTAGE', 'HRNUMHOU','PTDTRACE', 'PEEDUCA']]
    # agent_data_sample.replace(-1, np.nan, inplace=True)

    # knn_imputer = KNNImputer(n_neighbors=3)
    # agent_data_sample_knn_imputed = pd.DataFrame(knn_imputer.fit_transform(agent_data_sample[['HEFAMINC', 'PRTAGE', 'HRNUMHOU', 'PTDTRACE', 'PEEDUCA']]), columns=['hefaminc_imputed', 'prtage_imputed', 'hrnumhou_imputed', 'ptdtrace_imputed', 'peeduca_imputed'])

    # grouped = agent_data_sample.groupby('GESTFIPS').size()

    # agent_data_sample['weights'] = agent_data_sample['GESTFIPS'].map(1 / grouped)
    # agent_data_sample = pd.concat([agent_data_sample[['FIPS', 'GESTFIPS','weights']],agent_data_sample_knn_imputed], axis=1)
    # agent_data_sample = add_random_nodes(agent_data_sample)

    # agent_data_sample = agent_data_sample[['FIPS', 'GESTFIPS', 'weights',
    #                                         'nodes0', 'nodes1', 'nodes2', 'nodes3','nodes4',
    #                                         'hefaminc_imputed', 'prtage_imputed','hrnumhou_imputed', 
    #                                         'ptdtrace_imputed', 'peeduca_imputed']]
    
    # agent_data = pd.merge(product_data[['market_ids', 'market_ids_string', 'GESTFIPS']].drop_duplicates(),
    #                                   agent_data_sample, 
    #                                   how='inner', 
    #                                   left_on='GESTFIPS',
    #                                   right_on='GESTFIPS')
    
    #-----------------------------------------------------------------
    ##### Filtrar base a partir de ventas identificadas########
    #TODO: Quitar esta sección dado que la eliminación de retailers con ventas identificadas inferiores a un threshold se hará al interior de la función creating_product_data_rcl
    product_data = filter_by_identified_earnings(product_data, threshold_identified_earnings)
    # condition = product_data['fraction_identified_earnings']>=threshold_identified_earnings
    # kept_data = product_data.loc[condition].index
    # product_data = product_data.loc[kept_data]

    ####### Restringiendo la muestra a retailers que tienen 2 o más marcas identificadas ######## //
    #TODO: La restricción de los mercados a aquellos que tengan 2 o más marcas se debería implementar con anteriordad para evitar procesar información que posteriormente será eliminada. 
    product_data = filter_by_market_size(product_data, min_brands=2)

    # market_counts = product_data['market_ids'].value_counts()
    # valid_markets = market_counts[market_counts >= 2].index
    # product_data = product_data[product_data['market_ids'].isin(valid_markets)]

    #-----------------------------------------------------------------
    ######### Manteniendo la información en agents y data con iguales market_ids ##########
    agent_data, product_data = filter_matching_markets(agent_data, product_data)
    # agent_data = agent_data[agent_data['market_ids'].isin(set(product_data['market_ids']))]
    # product_data = product_data[product_data['market_ids'].isin(agent_data['market_ids'].unique())]


    # product_data.to_csv(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/processed_data/{week_dir}/{date}/product_data_preins_{DIRECTORY_NAME}_{datetime_}.csv', index=False)

    #-----------------------------------------------------------------
    ########## Creación de instrumentos ########## //TODO Revisar si las variables que se usan para crear los instrumentos también deben ser usadas al momento de definir el conjunto de características de los productos a ser analizados.
    formulation = pyblp.Formulation('0 + tar + nicotine + co + nicotine_mg_per_g + nicotine_mg_per_g_dry_weight_basis + nicotine_mg_per_cig')
    blp_instruments, local_instruments, quadratic_instruments = create_instruments(product_data, formulation)

    # formulation = pyblp.Formulation('0 + tar + nicotine + co + nicotine_mg_per_g + nicotine_mg_per_g_dry_weight_basis + nicotine_mg_per_cig')
    # blp_instruments = pyblp.build_blp_instruments(formulation, product_data)
    # blp_instruments = pd.DataFrame(blp_instruments)
    # blp_instruments.rename(columns={i:f'blp_instruments{i}' for i in blp_instruments.columns}, inplace=True)

    # local_instruments = pyblp.build_differentiation_instruments(formulation, product_data, version='local')
    # local_instruments = pd.DataFrame(local_instruments, columns=[f'local_instruments{i}' for i in range(local_instruments.shape[1])])

    # quadratic_instruments = pyblp.build_differentiation_instruments(formulation, product_data, version='quadratic')
    # quadratic_instruments = pd.DataFrame(quadratic_instruments, columns=[f'quadratic_instruments{i}' for i in range(quadratic_instruments.shape[1])])

    # local_instruments = local_instruments.loc[kept_data]
    # quadratic_instruments = quadratic_instruments.loc[kept_data]
    # blp_instruments = blp_instruments.loc[kept_data]

    # product_data.reset_index(drop=True, inplace=True)
    # blp_instruments.reset_index(drop=True, inplace=True)
    # local_instruments.reset_index(drop=True, inplace=True)
    # quadratic_instruments.reset_index(drop=True, inplace=True)


    # local_instruments = local_instruments.loc[product_data.index]
    # quadratic_instruments = quadratic_instruments.loc[product_data.index]
    # blp_instruments = blp_instruments.loc[product_data.index]

    #-----------------------------------------------------------------
    ######### Salvando instrumentos e información de los consumidores ###########

    save_processed_data(product_data=product_data, 
                       blp_instruments=blp_instruments, 
                       local_instruments=local_instruments, 
                       quadratic_instruments=quadratic_instruments, 
                       agent_data=agent_data, 
                       week_dir=week_dir, 
                       date=date, 
                       directory_name=DIRECTORY_NAME, 
                       datetime_str=datetime_)

    # os.makedirs(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/processed_data/{week_dir}/{date}', exist_ok=True)
    # # product_data.to_csv(os.path.join(output_dir, f'product_data_{first_week}_{num_weeks}.csv'), index=False)
    
    # product_data.to_csv(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/processed_data/{week_dir}/{date}/product_data_{DIRECTORY_NAME}_{datetime_}.csv', index=False)
    # blp_instruments.to_csv(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/processed_data/{week_dir}/{date}/blp_instruments_{DIRECTORY_NAME}_{datetime_}.csv', index=False)
    # local_instruments.to_csv(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/processed_data/{week_dir}/{date}/local_instruments_{DIRECTORY_NAME}_{datetime_}.csv', index=False)
    # quadratic_instruments.to_csv(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/processed_data/{week_dir}/{date}/quadratic_instruments_{DIRECTORY_NAME}_{datetime_}.csv', index=False)
    # agent_data.to_csv(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/processed_data/{week_dir}/{date}/agent_data_{DIRECTORY_NAME}_{datetime_}.csv', index=False)

    # Print all the locations where the DataFrames were saved
    # print(f"Product data saved to: /oak/stanford/groups/polinsky/Mergers/Cigarettes/processed_data/{week_dir}/{date}/product_data_{DIRECTORY_NAME}_{datetime_}.csv")
    # print(f"BLP instruments saved to: /oak/stanford/groups/polinsky/Mergers/Cigarettes/processed_data/{week_dir}/{date}/blp_instruments_{DIRECTORY_NAME}_{datetime_}.csv")
    # print(f"Local instruments saved to: /oak/stanford/groups/polinsky/Mergers/Cigarettes/processed_data/{week_dir}/{date}/local_instruments_{DIRECTORY_NAME}_{datetime_}.csv")
    # print(f"Quadratic instruments saved to: /oak/stanford/groups/polinsky/Mergers/Cigarettes/processed_data/{week_dir}/{date}/quadratic_instruments_{DIRECTORY_NAME}_{datetime_}.csv")
    # print(f"Agent data saved to: /oak/stanford/groups/polinsky/Mergers/Cigarettes/processed_data/{week_dir}/{date}/agent_data_{DIRECTORY_NAME}_{datetime_}.csv")
    # print(f"Compiled product data saved to: /oak/stanford/groups/polinsky/Mergers/Cigarettes/processed_data/{week_dir}/{date}/compiled_data_{DIRECTORY_NAME}_{datetime_}.csv")

    # product_data.reset_index(drop=True, inplace=True)
    # blp_instruments.reset_index(drop=True, inplace=True)
    # local_instruments.reset_index(drop=True, inplace=True)
    # quadratic_instruments.reset_index(drop=True, inplace=True)
    # agent_data.reset_index(drop=True, inplace=True)

    #-----------------------------------------------------------------
    product_data = compile_data(product_data = product_data, 
                            blp_inst = blp_instruments, 
                            local_inst = local_instruments, 
                            quad_inst = quadratic_instruments, 
                            agent_data= agent_data)
    
    product_data.to_csv(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/processed_data/{week_dir}/{date}/compiled_data_{DIRECTORY_NAME}_{datetime_}.csv', index=False)

    #-----------------------------------------------------------------
    print(f'empezando optimización {datetime_}')
    iter =  0

    #logit formulation 
    linear_formulation, non_linear_formulation, agent_formulation = create_formulations()

    linear_formulation=pyblp.Formulation('1+ prices', absorb='C(product_ids)')
    non_linear_formulation=pyblp.Formulation('1+ prices + tar')#k2=3
    agent_formulation=pyblp.Formulation('0 + hefaminc_imputed + prtage_imputed + hrnumhou_imputed + ptdtrace_imputed')#v=4

    plain_logit_results=plain_logit(product_data=product_data, formulation=linear_formulation)

    while iter <= 100:
        print('------------------------------')
        print(iter)
        print('------------------------------')
        try:
            results= rcl_with_demographics(product_data=product_data, 
                                           agent_data=agent_data,
                                           linear_formulation=linear_formulation,
                                           non_linear_formulation=non_linear_formulation,
                                           agent_formulation=agent_formulation,
                                           logit_results=plain_logit_results)
        

            results.to_pickle(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/ProblemResults_class/pickle/{week_dir}/{date}/{optimization_algorithm}/iteration_{iter}.pickle')
            
            print(f'------------results {iter}------------------')
            if results.converged == True:
                initial_prices = original_prices(product_data=product_data, results=results)


                #predicting the prices and appending the information to a dataframe
                predicted_prices = predict_prices(product_data = product_data, results = results, merger=[3,0])
                predicted_prices = predicted_prices.tolist()
                price_pred_df = product_data[['market_ids','market_ids_string','store_code_uc', 'week_end', 'product_ids', 'brand_code_uc', 'brand_descr']].copy()
                price_pred_df.loc[:, 'price_prediction'] = predicted_prices 


                #saving the file
                price_pred_df.to_json(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/Predicted/{week_dir}/{date}/{optimization_algorithm}/price_predictions_{iter}.json', index=False)
                print('predictions saved')

            # optimal_results = results_optimal_instruments(results)
        except Exception as e:
            print(e)
            

        iter += 1
        

    print('fin')



if __name__=='__main__': 
    main()
