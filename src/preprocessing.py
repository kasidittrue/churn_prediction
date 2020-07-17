import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import Pipeline , FeatureUnion
from sklearn.preprocessing import StandardScaler , OneHotEncoder
import joblib 
import os,sys

# Custom class 
# Data frame selection
class Dfselector (BaseEstimator,TransformerMixin):
    def __init__(self,feature,cat=False):
        self.feature= feature
        self.cat =cat
    def fit(self,X):
        return self
    def transform(self,X):
        return X[self.feature]

# Preprocess features
num_feature =['TOT_CALL_OUT_1D', 'TOT_CALL_IN_1D', 'TOT_CALL_OUT_MINS_1D', 'TOT_CALL_IN_MINS_1D', 'TOT_SMS_OUT_1D', 'TOT_SMS_IN_1D', 'TOT_DATA_MB_1D', 'TOT_TOPUP_TRANS_1D', 'TOT_TOPUP_AMT_1D', 'TOT_TOPPING_AMT_1D', 
              'TOT_CHARGE_1D', 'TOT_CALLING_1D', 'TOT_BIGBONUS_1D', 'TOT_CALL_OUT_7D', 'TOT_CALL_IN_7D', 'TOT_SMS_OUT_7D', 'TOT_SMS_IN_7D', 'TOT_DATA_MB_7D', 'TOT_TOPUP_TRANS_7D', 'TOT_TOPUP_AMT_7D', 
              'TOT_TOPPING_AMT_7D', 'TOT_CHARGE_7D', 'TOT_CALLING_7D', 'TOT_BIGBONUS_7D', 'TOT_CALL_OUT_30D', 'TOT_CALL_IN_30D', 'TOT_CALL_OUT_MINS_30D', 'TOT_CALL_IN_MINS_30D','TOT_SMS_OUT_30D', 'TOT_SMS_IN_30D',
              'TOT_DATA_MB_30D', 'TOT_TOPUP_TRANS_30D', 'TOT_TOPUP_AMT_30D', 'TOT_TOPPING_AMT_30D', 'TOT_CHARGE_30D', 'TOT_CALLING_30D', 'TOT_BIGBONUS_30D']
categorical_feature = ['TCARD_DESC','AO_MA_DAY','IMEI_USE_MTH','MAIN_BALANCE','SIM_DAY_LEFT']
no_preprocess_feature = ['NO_ACTIVITY_1D', 'NO_ACTIVITY_7D', 'NO_ACTIVITY_30D','TRU_FLAG', 'TMN_FLAG', 'TID_FLAG']

# Numerical features
num_trans = Pipeline([
    ('Dfselector',Dfselector(num_feature)),
    ('std',StandardScaler())
])
# Categorical features
cat_trans  = Pipeline([
    ('Dfselector',Dfselector(categorical_feature)),
    ('Onhot',OneHotEncoder())
])
# Preprocess pipeline
pre_process =FeatureUnion([
    ('num',num_trans),
    ('cat',cat_trans),
])


class Preprocess_object:
    def __init__(self):
        self.pre_processed_pipeline = None
    def clean(df):
        # Clean
        df['TRU_FLAG'].fillna(0,inplace = True)
        df['TMN_FLAG'].fillna(0,inplace = True)
        df['TID_FLAG'].fillna(0,inplace = True)
        df['IMEI_USE_MTH'].fillna(5.53,inplace = True)
        df['TCARD_DESC'].fillna('Unknown',inplace = True)

        #bin data 
        sim_bins = [-np.inf, 1, 7,30,180,270,365,730,np.inf]
        df.loc[:,'SIM_DAY_LEFT'] = pd.cut(df['SIM_DAY_LEFT'],sim_bins)

        imei_bins = [-np.inf, 1, 3,6,12,18,np.inf]
        df.loc[:,'IMEI_USE_MTH'] = pd.cut(df['IMEI_USE_MTH'],imei_bins)

        main_balance_bins = [-np.inf,0, 10,20,30,50,100,np.inf]
        df.loc[:,'MAIN_BALANCE'] = pd.cut(df['MAIN_BALANCE'],main_balance_bins)

        aos_bins =[-np.inf, 1, 7,30,180,270,365,365*2,365*3,365*4,365*5,np.inf]
        df.loc[:,'AO_MA_DAY'] = pd.cut(df['AO_MA_DAY'],aos_bins)
        return df

    def train(self, datapath,seps, nrows = None):
        # Load data
        if nrows == None:
            df = pd.read_csv(datapath,sep= seps )
        else:
            df = pd.read_csv(datapath,sep= seps,nrows = nrows )
        # Set index
        df.set_index('SUBS_KEY',inplace = True)

        # Clean
        df['TRU_FLAG'].fillna(0,inplace = True)
        df['TMN_FLAG'].fillna(0,inplace = True)
        df['TID_FLAG'].fillna(0,inplace = True)
        df['IMEI_USE_MTH'].fillna(5.53,inplace = True)
        df['TCARD_DESC'].fillna('Unknown',inplace = True)

        #Bin data 
        sim_bins = [-np.inf, 1, 7,30,180,270,365,730,np.inf]
        df.loc[:,'SIM_DAY_LEFT'] = pd.cut(df['SIM_DAY_LEFT'],sim_bins)

        imei_bins = [-np.inf, 1, 3,6,12,18,np.inf]
        df.loc[:,'IMEI_USE_MTH'] = pd.cut(df['IMEI_USE_MTH'],imei_bins)

        main_balance_bins = [-np.inf,0, 10,20,30,50,100,np.inf]
        df.loc[:,'MAIN_BALANCE'] = pd.cut(df['MAIN_BALANCE'],main_balance_bins)

        aos_bins =[-np.inf, 1, 7,30,180,270,365,365*2,365*3,365*4,365*5,np.inf]
        df.loc[:,'AO_MA_DAY'] = pd.cut(df['AO_MA_DAY'],aos_bins)        

        # create pipeline
        pre_processed_pipeline =  pre_process.fit(df)
        print('Pipeline created')
    
    def save(self,pathsave):
        if self.pre_processed_pipeline is not None:
            joblib.dump(self.pre_processed_pipeline, pathsave)
            print('Pipeline saved')
        else:
            raise TypeError("The Pipeline is not trained yet, use .train() before saving")

    def load(pathload):
        return  joblib.load(pathload)
   
# Post processing 
# feature importance
thirtyday_coef = np.array([-1.25943695e-01, -4.06313875e-02, -2.43569344e-02, -3.71426577e-02,
        9.98627627e-03,  9.76230686e-02, -1.05390834e+00, -6.42753957e-02,
        1.04846453e-01, -1.51456143e-01,  5.65674038e-02,  6.34679868e-03,
        3.21092330e-04, -2.67422098e-02, -1.14286597e-01, -1.50634012e-02,
        1.72657143e-01,  1.17978752e-01, -5.08027015e-03, -1.04213583e-01,
       -1.09130074e-01,  3.38323456e-02,  1.16549102e-02,  1.62603139e-03,
        7.30809895e-02, -1.40692570e-01,  1.73227911e-02,  3.31245978e-02,
        3.11701565e-02, -2.03325591e-01, -5.61072128e-02,  1.11760004e-01,
       -1.56714196e-01,  1.48502180e-01, -3.76513921e-02, -2.83532734e-03,
        3.94091944e-02, -1.24767095e-03, -9.89402953e-03, -1.85249529e-01,
       -6.63032029e-04, -6.09845170e-01, -6.87540928e-01, -5.26879458e-01,
        2.58578550e+00, -6.31971905e-01,  1.88082916e-01, -1.52340240e-01,
       -4.47795558e-01, -5.41439378e-02,  1.40181928e-01,  2.08751778e-01,
        1.18960484e-01, -1.06895771e-01,  3.76921765e-02,  3.03256992e-01,
        1.91287577e-01,  4.19429265e-02, -7.86396512e-02, -1.06182300e-02,
       -5.14735838e-01,  7.49199675e-01,  4.54276184e-01, -1.13966963e-01,
       -1.98396249e-01, -3.10059128e-01, -2.71247459e-01, -3.77312283e-01,
        3.24528224e-02,  1.09178470e-01, -5.60110676e-02, -5.27142330e-02,
        3.31429344e-02, -1.13718445e-01, -1.98948351e-02,  5.81303513e-05,
       -6.40834301e-02, -1.02140345e-01, -4.02681157e-02, -3.30869628e-01,
       -1.39395363e+00, -2.17076236e-01])

seven_coef =  np.array([-1.73508772e-01, -4.68912071e-02, -4.08553480e-02, -5.47410414e-02,
        4.86465078e-02, -3.34254223e-02, -1.24993577e+00,  3.66176097e-02,
        6.53890960e-02, -1.66558445e-01,  8.84612833e-02,  4.03993076e-02,
        5.12265875e-03, -2.10747447e-02, -1.27316018e-01, -1.09767917e-02,
        1.77897779e-02,  1.27125384e-01,  2.10754543e-02, -5.45217248e-03,
       -1.21081988e-01,  8.07231944e-02,  2.73403694e-03, -1.21323088e-03,
        8.93168390e-02, -3.06624079e-01,  1.43370899e-02,  5.29144624e-02,
        2.25694224e-02,  3.42248759e-02, -4.23606397e-03,  6.89004055e-02,
       -1.17194245e-01,  1.01242967e-01, -1.52604615e-01, -6.34469487e-03,
        2.86540918e-02, -1.01721204e-03, -6.58163452e-03, -1.77320845e-01,
       -3.65897144e-04, -3.62979757e-01, -3.90138266e-01, -3.55441490e-01,
        2.22265950e+00, -3.95733175e-01,  4.10633413e-01, -1.01013736e-01,
       -3.56682044e-01, -2.08354706e-02,  2.34287831e-01,  1.80936878e-01,
        1.12530152e-01, -1.57977723e-02,  8.90219682e-02,  4.11878009e-01,
        2.40037132e-01,  1.78764870e-01,  3.30541182e-02,  6.34899731e-02,
       -3.94142882e-01,  8.40974034e-01,  5.40845390e-01,  3.35661998e-03,
       -1.36276606e-01, -2.33099973e-01, -2.20108279e-01, -2.62609966e-01,
        7.34069984e-02,  2.35352823e-01,  2.70639214e-01,  1.62773318e-01,
        6.93509487e-02, -1.82451409e-01, -9.69217378e-02,  9.31064340e-04,
        5.38355163e-01, -1.22662774e-01, -7.08652893e-02, -1.77534721e-01,
       -7.75849008e-01, -1.02782164e-01])