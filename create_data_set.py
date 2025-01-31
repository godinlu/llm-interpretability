from classif_model import GlobalConfig, TextClassifier, DataProcessor

if __name__ == '__main__':
    # Create a configuration object that contains all parameters for the script
    cfg: GlobalConfig = GlobalConfig(model_name = 'distilbert/distilgpt2')
    # Create data processor object that handles all data formating operations
    data_processor = DataProcessor(cfg)
    #load_and_evaluate("C:/Users/andre/Documents/Cours/SSD/M2/ProjetM2/lightning_logs/version_2/checkpoints/epoch=0-step=497.ckpt",
    #                  data_processor.data_loader_val,
    #                  "evl_output_test.pkl")
    model = TextClassifier(cfg)
    model.train_model(data_processor.data_loader_train, model_name="testing_mod")
    out = model.evaluate(data_processor.data_loader_test, "evl_output_test.pkl")
    print(f"Test Accuracy: {out["accuracy"]:.4f}")
    