import datetime
import pyblp

import sys


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
from .estimaciones.utils import save_dict_json
from .estimaciones.post_estimation_merger_simulation import predicted_prices
from .estimaciones.optimal_instruments import results_optimal_instruments
from .nielsen_data_cleaning.utils import create_output_directories, create_agent_data_from_cps, create_agent_data_sample, filter_by_identified_earnings, filter_by_market_size, filtering_agent_data_match_markets, create_instruments, save_processed_data, create_formulations, check_matrix_collinearity, find_first_non_collinear_matrix, select_product_data_columns, filtering_data_by_identified_sales, filtering_data_by_number_brands, matching_agent_and_product_data, create_directories, save_product_data, run_optimization_iterations


DIRECTORY_NAME = 'Reynolds_Lorillard'
DEPARTMENT_CODE = 4510 #aka product_group_code
# PRODUCT_MODULE = 7460
# NROWS = 10000000
YEAR = 2014
# WEEKS = [20140125, 20140201]

#TODO: organizar las funciones de Nielsen_data_cleaning dependiendo de la base que está procesando. Las que se usen en diferentes bases deberían ir en un archivo más general. 


def main(num_iterations:int=1, post_estimation: bool=True, iteration_max_evaluations:int=100000, iteration_atol:float=1e-14):
    date = datetime.datetime.today().strftime("%Y-%m-%d")
    datetime_=datetime.datetime.today()
    year=2014
    first_week=20
    num_weeks=1
    threshold_identified_earnings = 0.35
    optimization_algorithm = 'l-bfgs-b'

    #-----------------------------------------------------------------
    # Crea las formulaciones
    linear_formulation = '1+ prices'
    absorb_formula = 'C(product_ids)'
    non_linear_formulation = '1+ prices + tar'
    agent_formulation_str = '0 + hefaminc_imputed + prtage_imputed + hrnumhou_imputed + ptdtrace_imputed'
    linear_formulation, non_linear_formulation, agent_formulation = create_formulations(linear_formula=linear_formulation, absorb_formula=absorb_formula, nonlinear_formula=non_linear_formulation, agent_formula=agent_formulation_str)

    #-----------------------------------------------------------------
    # Crea la base de datos de productos
    product_data = creating_product_data_rcl(main_dir='/oak/stanford/groups/polinsky/Mergers/Cigarettes',
                                     movements_path=f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/Nielsen_data/{year}/Movement_Files/4510_{year}/7460_{year}.tsv',
                                     stores_path='Nielsen_data/2014/Annual_Files/stores_2014.tsv',
                                     products_path='Nielsen_data/Master_Files/Latest/products.tsv',
                                     extra_attributes_path='Nielsen_data/2014/Annual_Files/products_extra_2014.tsv', 
                                     first_week=first_week,
                                     num_weeks=num_weeks,
                                     lower_threshold_identified_sales=threshold_identified_earnings)
    
    #-----------------------------------------------------------------
    # Selecciona y organizalas columnas necesarias para el análisis
    product_data = select_product_data_columns(product_data=product_data)
   
    #-------------------------------------------------------------
    # Crea directorios para guardar los datos procesados, las predicciones y los resultados de la optimización.
    week_dir = create_output_directories(product_data=product_data, 
                                        date=date,
                                        optimization_algorithm=optimization_algorithm)

    #-----------------------------------------------------------------
    # Agregando información sociodemográfica 
    pop_agent_data = create_agent_data_from_cps(record_layout_path='/oak/stanford/groups/polinsky/Current_Population_Survey/2014/January_2014_Record_Layout.txt',
                                                agent_data_path='/oak/stanford/groups/polinsky/Current_Population_Survey/2014/apr14pub.dat')
    #-----------------------------------------------------------------
    # Crea la base de datos de consumidores
    num_nodes=len([term.strip() for term in agent_formulation_str.split('+') if term.strip() != '0'])
    filtered_sample_agent_data = create_agent_data_sample(agent_data_pop=pop_agent_data, product_data=product_data, num_samples=400, n_neighbors=4, num_nodes=num_nodes)

    #-----------------------------------------------------------------
    ######### Manteniendo la información en agents y data con iguales market_ids ##########
    filtered_agent_data = filtering_agent_data_match_markets(filtered_sample_agent_data, product_data)

    #-----------------------------------------------------------------
    ########## Creación de instrumentos ########## 
    #TODO Revisar si las variables que se usan para crear los instrumentos también deben ser usadas al momento de definir el conjunto de características de los productos a ser analizados.
    formulation = pyblp.Formulation('0 + tar + nicotine + co + nicotine_mg_per_g + nicotine_mg_per_g_dry_weight_basis + nicotine_mg_per_cig')
    blp_instruments, local_instruments, quadratic_instruments = create_instruments(product_data, formulation)
    #-----------------------------------------------------------------
    ######### Salvando instrumentos e información de los consumidores ###########
    # save_processed_data(product_data=product_data, 
    #                    blp_instruments=blp_instruments, 
    #                    local_instruments=local_instruments, 
    #                    quadratic_instruments=quadratic_instruments, 
    #                    agent_data=filtered_agent_data, 
    #                    week_dir=week_dir, 
    #                    date=date, 
    #                    directory_name=DIRECTORY_NAME, 
    #                    datetime_str=datetime_)

    #-----------------------------------------------------------------
    product_data = compile_data(product_data = product_data, 
                            blp_inst = blp_instruments, 
                            local_inst = local_instruments, 
                            quad_inst = quadratic_instruments, 
                            agent_data= filtered_agent_data)
    
    product_data.to_csv(f'/oak/stanford/groups/polinsky/Mergers/Cigarettes/processed_data/{date}/{week_dir}/compiled_data_{DIRECTORY_NAME}_{datetime_}.csv', index=False)

    #-----------------------------------------------------------------

    #logit formulation 
    print('Logit')
    # linear_formulation, non_linear_formulation, agent_formulation = create_formulations()

    plain_logit_results=plain_logit(product_data=product_data, formulation=linear_formulation)

    results_list=[]
    results=run_optimization_iterations(
        product_data=product_data,
        filtered_sample_agent_data=filtered_sample_agent_data,
        week_dir=week_dir,
        date=date,
        optimization_algorithm=optimization_algorithm,
        num_iterations=num_iterations,
        linear_formulation=linear_formulation,
        non_linear_formulation=non_linear_formulation,
        agent_formulation=agent_formulation,
        plain_logit_results=plain_logit_results,
        iteration_max_evaluations=iteration_max_evaluations,
        iteration_atol=iteration_atol
    )            
    results_list.append(results)
    return results_list


if __name__=='__main__': 
    import sys
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        num_iterations = int(sys.argv[1])
    else:
        num_iterations = 1  # default value
    main(num_iterations=num_iterations)
