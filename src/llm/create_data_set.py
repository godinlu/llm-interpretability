from classif_model import GlobalConfig, TextClassifier, DataProcessor
from pathlib import Path

if __name__ == '__main__':
    # Create a configuration object that contains all parameters for the script
    cfg: GlobalConfig = GlobalConfig(model_name = 'distilbert/distilgpt2',
                                     max_epochs=1,
                                     batch_size_train=124,
                                     path_to_save_cfg = Path("gpt2_L1_pad_cfg_3.pkl"),
                                     model_file_name = "gpt2_L1_pad_3",
                                     L1_lambda = 0.8,
                                     fixed_pading = True,
                                     extract_lm_parameters = True,
                                     extract_clasif_parameters = True,
                                     compress_results_file=False)
    # Create data processor object that handles all data formating operations
    data_processor = DataProcessor(cfg)
    #load_and_evaluate("C:/Users/andre/Documents/Cours/SSD/M2/ProjetM2/lightning_logs/version_2/checkpoints/epoch=0-step=497.ckpt",
    #                  data_processor.data_loader_val,
    #                  "evl_output_test.pkl")
    model = TextClassifier(cfg)
    model.train_model(data_processor.data_loader_train)
    out = model.evaluate(data_processor.data_loader_val)
    print(f"Test Accuracy: {out["accuracy"]:.4f}")
    