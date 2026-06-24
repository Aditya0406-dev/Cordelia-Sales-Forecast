# Cordelia Cruise Forecasting Pipeline 
Execute scripts in the following order to reproduce the data layer: 
1. python generate_synthetic_data.py 
2. python build_features.py 
3. python build_indian_holidays.py 
4. python train_all_models.py 
5. streamlit run dashboard.py 
