# TODO - House Price Prediction Refactor

## Step 1: Implement production refactor (main.py)
- [x] Rewrite `main.py` using `Pipeline` + `ColumnTransformer`
- [x] Use `OneHotEncoder(handle_unknown="ignore")` for `city` and `statezip`
- [x] Drop high-cardinality columns (e.g., `street`) and drop constant columns (e.g., `country` when single-valued)
- [x] Convert `date` into `year`, `month`, `day` via a custom transformer
- [x] Train and compare LinearRegression, RandomForestRegressor, GradientBoostingRegressor
- [x] Print MAE, RMSE, R² for each; select best by highest R²
- [x] Save the complete fitted pipeline to `best_model.pkl`
- [x] Preserve all existing plots + ensure feature-importance mapping works with pipeline one-hot outputs


## Step 2: Implement inference refactor (inference.py)
- [x] Replace inference logic to load `best_model.pkl` and call `.predict()` on raw input DataFrame
- [x] Remove any manual dummy column creation


## Step 3: Add missing project docs
- [ ] Create `requirements.txt`
- [ ] Create `README.md`

## Step 4: Validate end-to-end run
- [ ] Run `python main.py` and confirm plots + `best_model.pkl` generation
- [ ] Run `python inference.py` and confirm no feature mismatch errors

