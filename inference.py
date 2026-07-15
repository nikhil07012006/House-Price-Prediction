import joblib
import pandas as pd

MODEL_PATH = "best_model.pkl"


def main():
    model = joblib.load(MODEL_PATH)

    # Raw input schema (same as training X = df.drop('price')).
    # No manual dummy columns.
    sample = pd.DataFrame(
        [
            {
                "date": "2014-05-02",
                "bedrooms": 3,
                "bathrooms": 2,
                "sqft_living": 2000,
                "sqft_lot": 5000,
                "floors": 2,
                "waterfront": 0,
                "view": 0,
                "condition": 3,
                "sqft_above": 1800,
                "sqft_basement": 200,
                "yr_built": 2005,
                "yr_renovated": 0,
                "street": "UNKNOWN",
                "city": "UNKNOWN",
                "statezip": "UNKNOWN",
                "country": "USA",
            }
        ]
    )

    prediction = model.predict(sample)
    print("Predicted House Price:", float(prediction[0]))


if __name__ == "__main__":
    main()

