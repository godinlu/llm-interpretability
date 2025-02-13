from typing import Dict, Any
from classif_model import GlobalConfig, TextClassifier, DataProcessor, apply_to_attentions_cond, sum_head, div_attentions, max_head, unpack_attentions
from pathlib import Path
from torch.cuda import empty_cache

def format_output(res: Dict[str, Any]) -> Dict[str, Any]:
    res["attentions"] = unpack_attentions(res["attentions"])
    nb_obs = len(res["attentions"])
    # Get mean values
    val_list = list(apply_to_attentions_cond(res["attentions"].copy(), 
                                             sum_head, 
                                             res["predictions"], 
                                             res["targets"]))
    for idx in range(len(val_list)):
        val_list[idx] = div_attentions(val_list[idx], len(val_list[idx]))
    # Get max values
    val_list.extend(list(apply_to_attentions_cond(res["attentions"].copy(), 
                                                  max_head, 
                                                  res["predictions"], 
                                                  res["targets"])))
    res.pop("attentions")
    names = ["att_is_fun_mean", 
             "att_is_unfun_mean", 
             "att_pred_fun_mean", 
             "att_pred_unfun_mean",
             "att_is_fun_max", 
             "att_is_unfun_max", 
             "att_pred_fun_max", 
             "att_pred_unfun_max"]
    for idx, val in enumerate(val_list):
        res.update({names[idx] : val})
    return res

if __name__ == '__main__':
    for i in range(3):
        # Create a configuration object that contains all parameters for the script
        cfg: GlobalConfig = GlobalConfig(model_name = 'distilbert/distilgpt2',
                                     max_epochs = 15,
                                     batch_size_train = 16,
                                     model_file_name = f"gpt2_L1_pad_{i}",
                                     L1_lambda = 1,
                                     fixed_pading = False,
                                     extract_lm_parameters = True,
                                     extract_clasif_parameters = True,
                                     compress_results_file = False,
                                     #formating_output_function=format_output
                                     )
        # Create data processor object that handles all data formating operations
        data_processor: DataProcessor = DataProcessor(cfg)
        model = TextClassifier(cfg)
        model.train_model(data_processor.data_loader_train)
        model.evaluate(data_processor.data_loader_val)
        # Makes sure that vram is flushed properly at each iteration
        model = model.cpu()
        del model, data_processor, cfg
        empty_cache()
    