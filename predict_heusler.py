import os
import pandas as pd
import numpy as np
import joblib
import dill
from mastml.feature_generators import ElementalFeatureGenerator, OneHotGroupGenerator

def get_preds_ebars_domains(df_test):
    d = 'model_heusler'
    scaler = joblib.load(os.path.join(d, 'StandardScaler.pkl'))
    model = joblib.load(os.path.join(d, 'RandomForestRegressor.pkl'))
    df_features = pd.read_csv(os.path.join(d, 'X_train.csv'))
    recal_params = pd.read_csv(os.path.join(d, 'recal_dict.csv'))

    features = df_features.columns.tolist()
    df_test = df_test[features]

    X = scaler.transform(df_test)

    # Make predictions
    preds = model.predict(X)

    # Get ebars and recalibrate them
    errs_list = list()
    a = recal_params['a'][0]
    b = recal_params['b'][0]
    for i, x in X.iterrows():
        preds_list = list()
        for pred in model.model.estimators_:
            preds_list.append(pred.predict(np.array(x).reshape(1, -1))[0])
        errs_list.append(np.std(preds_list))
    ebars = a * np.array(errs_list) + b 

    # Get domains
    with open(os.path.join(d, 'model.dill'), 'rb') as f:
        model_domain = dill.load(f)

    domains = model_domain.predict(X)

    return preds, ebars, domains

def process_data(comp_list):
    X = pd.DataFrame(np.empty((len(comp_list),)))
    y = pd.DataFrame(np.empty((len(comp_list),)))

    df_test = pd.DataFrame({'Material composition': comp_list})

    # Try this both ways depending on mastml version used.
    try:
        X, y = ElementalFeatureGenerator(composition_df=df_test['Material composition'],
                                    feature_types=['composition_avg', 'arithmetic_avg', 'max', 'min','difference'],
                                    remove_constant_columns=False).evaluate(X=X, y=y, savepath=os.getcwd(), make_new_dir=False)
    except:
        X, y = ElementalFeatureGenerator(featurize_df=df_test['Material composition'],
                                         feature_types=['composition_avg', 'arithmetic_avg', 'max', 'min',
                                                        'difference'], remove_constant_columns=False).evaluate(X=X, y=y, savepath=os.getcwd(), make_new_dir=False)

    df_test = pd.concat([df_test, X], axis=1)

    return df_test

def make_predictions(comp_list, numelec_list, heuslertype_list):

    # Process data
    df_test = process_data(comp_list)

    # Process data
    X_train = pd.read_csv('model_heusler/X_train.csv')
    feature_names = X_train.columns.tolist()

    # Convert heusler type encoding to numbers
    pf_0 = list()
    pf_1 = list()
    pf_2 = list()
    for i in heuslertype_list:
        if i == 'Full Heusler':
            pf_0.append(1)
            pf_1.append(0)
            pf_2.append(0)
        elif i == 'Inverse Heusler':
            pf_0.append(0)
            pf_1.append(1)
            pf_2.append(0)
        elif i == 'Half Heusler':
            pf_0.append(0)
            pf_1.append(0)
            pf_2.append(1)
        else:
            raise ValueError('Heusler type must be one of Full Heusler, Half Heusler, or Inverse Heusler')
    
    df_test['heusler type_0'] = pf_0
    df_test['heusler type_1'] = pf_1
    df_test['heusler type_2'] = pf_2
    df_test['num_electron'] = numelec_list

    # Get the ML predicted values
    preds, ebars, domains = get_preds_ebars_domains(df_test)

    pred_dict = {'Predicted magnetization (emu/cm3)': preds,
                 'Ebar magnetization (emu/cm3)': ebars}

    for d in domains.columns.tolist():
        pred_dict[d] = domains[d]

    del pred_dict['y_pred']
    #del pred_dict['d_pred']
    del pred_dict['y_stdu_pred']
    del pred_dict['y_stdc_pred']

    for f in feature_names:
        pred_dict[f] = np.array(df_test[f]).ravel()

    return pd.DataFrame(pred_dict)