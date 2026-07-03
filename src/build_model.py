from predictor import TennisPredictor

if __name__ == "__main__":
    predictor = TennisPredictor(data_dir="data")
    predictor.save(model_dir="models")
    print("Saved trained model to models/predictor.joblib")
