import pandas as pd
import pyblp
# import statsmodels.formula.api as smf

from .utils import create_instrument_dict
# prompt: create a dict that returns keys equal to the value of the variable and values equal to the number of times the value appears in the variable



def plain_logit(product_data: pd.DataFrame, formulation: pyblp.Formulation):
    problem= pyblp.Problem(product_formulations=formulation, 
                           product_data=product_data)
    
    logit_results=problem.solve()

    return logit_results

    

if __name__ == '__main__':
    plain_logit()
