from src import jdbapi, model, preprocessing
from src.preprocessing import Dfselector
import pandas as pd
import numpy as np
from joblib import  load
from datetime import datetime

startTime = datetime.now()

df = jdbapi.read_jdbc("""
SELECT 
    *
FROM TMP_POOM2
LIMIT 5000
""")

# df = jdbapi.read_jdbc("""
# SELECT 
#      -- Index
#      ED2.SUBS_KEY, 
#      -- Label
#      STOP_MA_MTE_7DAY, 
#      STOP_MA_MTE_30DAY, 
#      -- Features
# 	 NO_ACTIVITY_1D, 
#      NO_ACTIVITY_7D, 
#      NO_ACTIVITY_30D, 
#      AO_MA_DAY, 
#      TOT_CALL_OUT_1D, 
#      TOT_CALL_IN_1D, 
#      TOT_CALL_OUT_MINS_1D, 
#      TOT_CALL_IN_MINS_1D, 
#      TOT_SMS_OUT_1D, 
#      TOT_SMS_IN_1D, 
#      TOT_DATA_MB_1D, 
#      TOT_TOPUP_TRANS_1D, 
#      TOT_TOPUP_AMT_1D, 
#      TOT_TOPPING_AMT_1D, 
#      TOT_CHARGE_1D, 
#      TOT_CALLING_1D, 
#      TOT_BIGBONUS_1D, 
#      TOT_CALL_OUT_7D, 
#      TOT_CALL_IN_7D, 
#      TOT_SMS_OUT_7D, 
#      TOT_SMS_IN_7D, 
#      TOT_DATA_MB_7D, 
#      TOT_TOPUP_TRANS_7D, 
#      TOT_TOPUP_AMT_7D, 
#      TOT_TOPPING_AMT_7D, 
#      TOT_CHARGE_7D, 
#      TOT_CALLING_7D, 
#      TOT_BIGBONUS_7D, 
#      TOT_CALL_OUT_30D, 
#      TOT_CALL_IN_30D, 
#      TOT_CALL_OUT_MINS_30D, 
#      TOT_CALL_IN_MINS_30D, 
#      TOT_SMS_OUT_30D, 
#      TOT_SMS_IN_30D, 
#      TOT_DATA_MB_30D, 
#      TOT_TOPUP_TRANS_30D, 
#      TOT_TOPUP_AMT_30D, 
#      TOT_TOPPING_AMT_30D, 
#      TOT_CHARGE_30D, 
#      TOT_CALLING_30D, 
#      TOT_BIGBONUS_30D, 
#      TRU_FLAG, TMN_FLAG, TID_FLAG, TCARD_DESC,IMEI_USE_MTH
#    	-- FROM BALANCE
# 	,BAL.BAL_AMT as MAIN_BALANCE
# 	,days_between(BAL.BAL_END_TM_KAY_DAY  ,  to_date(BAL.SNAPSHOT_TM_KEY_DAY, 'YYYYMMDD') ) as SIM_DAY_LEFT
# FROM 
# 	TELCOANAPRD.PERMPOO.TMP_PREPAID_PROFILE_ED2 AS ED2
# 	LEFT JOIN ADMIN_PREP.FCT_SUBS_MAIN_BALANCE AS BAL
# 		ON ED2.SUBS_KEY = BAL.SUBS_KEY AND ED2.MAX_DATE = BAL.SNAPSHOT_TM_KEY_DAY
# WHERE
# 	     MAX_DATE >= 20200101
# 	AND 
#           AO_MA_DAY >= 30
#      AND 
#           MAIN_BALANCE IS NOT NULL 
#      --AND
#           --KEY_DATE = TO_CHAR(current_timestamp,'YYYYMMDD')

# LIMIT 5
# """)

print('Time spent import data: ',datetime.now() - startTime)
print('Total time:',datetime.now() - startTime)
importTime = datetime.now()

# Clean dataset
keep_index = df['SUBS_KEY']
df = preprocessing.Preprocess_object.clean(df)
# Load trained pipeline
process_pipeline = preprocessing.Preprocess_object.load('model/pre_process.joblib')
# Transform dataset
df_processed = process_pipeline.transform(df)
# Concat 
df_processed = np.concatenate((df_processed.todense(),np.array(df[preprocessing.no_preprocess_feature])),axis = 1)
# List of feature used
features_final = np.concatenate((preprocessing.num_feature,process_pipeline.transformer_list[1][1]['Onhot'].get_feature_names(preprocessing.categorical_feature)),axis = 0)
features_final = np.concatenate((features_final,preprocessing.no_preprocess_feature))

print('Time spent preprocess data: ',datetime.now() - importTime)
print('Total time:',datetime.now() - startTime)
preprocessTime = datetime.now()

# Prediction
# Load trained model
model = load('model/rnd_base.joblib')
# model.predict proba
#churn7_30 = model.predict(df_processed)
proba_7,proba_30 = model.predict_proba(df_processed)

print('Time spent on model making prediction: ',datetime.now() - preprocessTime)
print('Total time:',datetime.now() - startTime)
modelTime = datetime.now()

# Get Feature importance by level
thirtyday_factor = pd.DataFrame(df_processed,columns= features_final) * preprocessing.thirtyday_coef
seven_factor = pd.DataFrame(df_processed,columns= features_final) * preprocessing.seven_coef
factors30 = np.c_[pd.DataFrame(thirtyday_factor.apply(lambda x:list(thirtyday_factor.columns[np.array(x).argsort()[::-1][:5]]), axis=1).to_list(),  columns=['CHURN_FACTOR_1_7D', 'CHURN_FACTOR_2_7D', 'CHURN_FACTOR_3_7D','CHURN_FACTOR_4_7D','CHURN_FACTOR_5_7D']),pd.DataFrame(thirtyday_factor.apply(lambda x:list(thirtyday_factor.columns[np.array(x).argsort()[::1][:5]]), axis=1).to_list(),  columns=['NOTCHURN_FACTOR_1_7D', 'NOTCHURN_FACTOR_2_7D', 'NOTCHURN_FACTOR_3_7D','NOTCHURN_FACTOR_4_7D','NOTCHURN_FACTOR_5_7D'])]
factors7 = np.c_[pd.DataFrame(seven_factor.apply(lambda x:list(seven_factor.columns[np.array(x).argsort()[::-1][:5]]), axis=1).to_list(),  columns=['CHURN_FACTOR_1_30D', 'CHURN_FACTOR_2_30D', 'CHURN_FACTOR_3_30D','CHURN_FACTOR_4_30D','CHURN_FACTOR_5_30D']),pd.DataFrame(seven_factor.apply(lambda x:list(seven_factor.columns[np.array(x).argsort()[::1][:5]]), axis=1).to_list(),  columns=['NOTCHURN_FACTOR_1_30D', 'NOTCHURN_FACTOR_2_30D', 'NOTCHURN_FACTOR_3_30D','NOTCHURN_FACTOR_4_30D','NOTCHURN_FACTOR_5_30D'])]
factors = np.c_[factors30,factors7]

# write to csv(Netezza)
prediction = pd.DataFrame(np.c_[np.c_[proba_7,proba_30],factors]
             ,columns=['prob_not_churn_7day'
                         ,'prob_churn_7day'
                         ,'prob_not_churn_30day'
                         ,'prob_churn_30day'
                         ,'CHURN_FACTOR_1_7D', 'CHURN_FACTOR_2_7D', 'CHURN_FACTOR_3_7D','CHURN_FACTOR_4_7D','CHURN_FACTOR_5_7D'
                         ,'NOTCHURN_FACTOR_1_7D', 'NOTCHURN_FACTOR_2_7D', 'NOTCHURN_FACTOR_3_7D','NOTCHURN_FACTOR_4_7D','NOTCHURN_FACTOR_5_7D'
                         ,'CHURN_FACTOR_1_30D', 'CHURN_FACTOR_2_30D', 'CHURN_FACTOR_3_30D','CHURN_FACTOR_4_30D','CHURN_FACTOR_5_30D'
                         ,'NOTCHURN_FACTOR_1_30D', 'NOTCHURN_FACTOR_2_30D', 'NOTCHURN_FACTOR_3_30D','NOTCHURN_FACTOR_4_30D','NOTCHURN_FACTOR_5_30D'])

# add high, medium, low labels
prediction['churn_likely7d'] = prediction['prob_churn_7day'].apply(lambda x: 'H' if x >= .8 else ('M' if x>= 0.6 else 'L'))
prediction['churn_likely30d'] = prediction['prob_churn_30day'].apply(lambda x: 'H' if x >= .8 else ('M' if x>= 0.6 else 'L'))
# add today date
prediction['KEY_DATE'] = datetime.today().strftime('%Y%m%d')
# prediction.set_index(keep_index,inplace = True)
prediction['SUBS_KEY'] = keep_index

print('Time spent add feature importances: ',datetime.now() - modelTime)
print('Total time:',datetime.now() - startTime)
featureTime = datetime.now()

# to Netezza
dest_table = 'TMP_POOM1'
jdbapi.insertBulk(dest_table,prediction)


print('Time spent export to Netezza: ',datetime.now() - featureTime)
print('Total time:',datetime.now() - startTime)
print('Successfully export to {}'.format(dest_table))
# prediction.to_csv('data/prediction.csv')
# print('Successfully prediction to csv')