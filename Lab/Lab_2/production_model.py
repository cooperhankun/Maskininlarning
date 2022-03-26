import joblib
import pandas as pd

df = pd.read_csv("test_samples.csv")
LR_deploy_model = joblib.load("LR_deploy_model")

predictions = pd.DataFrame(LR_deploy_model.predict_proba(scaled_X_test), columns=["probability class 0", "probability class 1"])
predictions["prediction"] = LR_deploy_model.predict(scaled_X_test)

predictions.to_csv("prediction.csv", index=False)