import pickle
import pandas as pd
import numpy as np

# read in the model
import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file_model = os.path.join(THIS_FOLDER, 'drugs_final_model.p')
with open(my_file_model,'rb') as f:
   my_model = pickle.load(f)

my_file_ref_table = os.path.join(THIS_FOLDER, 'Flask_ref_table.csv')
ref_table = pd.read_csv(my_file_ref_table, index_col = 0)

# create a function to take in user-entered amounts, calculate variables for model and apply the model
def user_likelihood(bokeh_inputs, model = my_model):
    # Inputs from bokeh:
    age = bokeh_inputs[0]
    sex = bokeh_inputs[1]
    country = bokeh_inputs[2]
    edu = bokeh_inputs[3]
    n = bokeh_inputs[4]
    e = bokeh_inputs[5] 
    o = bokeh_inputs[6] 
    c = bokeh_inputs[7]
    a = bokeh_inputs[8]  
    s = bokeh_inputs[9]
    i = bokeh_inputs[10]
  
    # Reference score for age:
    age_dict = {
        0: 18,
        1: 25,
        2: 35,
        3: 45,
        4: 55,
        5: 65
    }
    
    # Reference table for edu:
    edu_dict = {
        'Left before 16 yrs old':15,
        'Left at 16 yrs old':16,
        'Left at 17 yrs old':17,
        'Left at 18 yrs old':18,
        'Some college or university, no certificate or degree':19,
        'Professional certificate/diploma':20,
        'University degree':21,
        'Masters degree':23,
        'Doctorate degree':27
    }    
    
    # Reference tables for impulsivity score:
    i_dict = {
        1: -2.55524,
        2: -1.37983,
        3: -0.71126,
        4: -0.21712,
        5: 0.19268,
        6: 0.52975,
        7: 0.88113,
        8: 1.29221,
        9: 1.86203,
        10: 2.90161
    }

    # Reference table for sensation seeking score:
    s_dict = {
        1: -2.07848,
        2: -1.54858, 
        3: -1.18084,
        4: -0.84637,
        5: -0.52593,
        6: -0.21575,
        7: 0.07987,
        8: 0.40148,
        9: 0.76540,
        10: 1.22470,
        11: 1.92173
    }
    
    # Transform Age
    age = age_dict[age]
    
    # Transform sex to binary
    if sex == 'male':
        sex_bin = 1
    else: sex_bin = 0        
    
    # Transform Edu
    edu = edu_dict[edu]

    # Transform impulsivity
    impulsivity = i_dict[i] # use value of i from Bokeh to look up corresponding value for model input
    
    # Transform sensation seeking
    sensation_seeking = s_dict[s]
    
    # Calculate edu_vs_exp value for model input
    edu_vs_exp = float(edu / (ref_table.loc[ref_table['country']==country]['edu_exp_yr'].iat[0]))
    
    # Calculate edu_vs_mean value for model input
    edu_vs_mean = float(edu / (ref_table.loc[ref_table['country']==country]['edu_mean_yr'].iat[0]))
    
    # Calculate age_vs_exp value for model input (use sex specific life expectancy)
    age_vs_exp = float(age / (np.where(sex_bin == 0, 
                                        float(ref_table.loc[ref_table['country']==country]['female_life'].iat[0]), 
                                        float(ref_table.loc[ref_table['country']==country]['male_life'].iat[0])
                                       )))
    
    # Calculate Big5 Metric
    big5 = float(( (sex_bin + 1)**2 ) * (( (sensation_seeking) + o + n + impulsivity ) / (c + age)))
    
    # Calculate nes_c metric
    nes_c = float((n / 34.0429 * e * 2*sensation_seeking) - 85*(c / 39.5499 )**2)
    
    # Calculate n_ac metric
    n_ac = float( n / (c * a) )
    
    # Calculate no_ metric
    no_ = float( n * o )

    # Calculate gdp_educ metric
    gdp_educ = float( ref_table.loc[ref_table['country']==country]['gdp_pc'] / c)
    
    # inputs for the model
    model_inputs_df = [[
        n,
        a,
        c,
        impulsivity,
        sensation_seeking,
        edu_vs_exp,
        edu_vs_mean,
        age_vs_exp,
        big5,
        nes_c,
        n_ac,
        no_,
        gdp_educ
    ]]
    
    # make a prediction
    prediction = my_model.predict_proba(model_inputs_df)[0][1]*100
    
    # print(prediction)

    return prediction