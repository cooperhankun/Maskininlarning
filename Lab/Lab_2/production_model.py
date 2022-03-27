import joblib
import pandas as pd

# Load the model and samples
df = pd.read_csv("test_samples.csv")
LR_deploy_model = joblib.load("LR_deploy_model")

# apply the model to the samples
predictions = pd.DataFrame(LR_deploy_model.predict_proba(df), columns=["probability class 0", "probability class 1"])
predictions["prediction"] = LR_deploy_model.predict(df)

predictions.to_csv("prediction.csv", index=False)