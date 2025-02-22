from typing import Dict, List, Any, Union, Callable, Iterable
from torch import Tensor, tensor, no_grad, argmax, unsqueeze, sum, abs, zeros, max
from torch.cuda import empty_cache
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lightning import LightningModule, Trainer
from lightning.pytorch.loggers import CSVLogger
from transformers import AutoModel, AutoTokenizer#, BertTokenizer, BertForSequenceClassification, AutoModelForSequenceClassification
#from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from pathlib import Path
from pandas import read_csv, concat
from tqdm import tqdm
from pickle import dump, load
from concurrent.futures import ThreadPoolExecutor
#from pprint import pprint

@dataclass
class GlobalConfig:
    # list of devices on which model is run
    #device: int = 0
    #accelerator: str = "gpu"
    dropout: int = 0.2
    # batch size
    batch_size_train: int = 16
    #batch_size_predict: int = 16
    # learning rate
    lr: float = 5e-5
    lr_scheduler_factor: float = 0.1
    """Learing rate reduction factor (e.g: by how much to reduce the learing rate at each modification)."""
    lr_scheduler_patience: int = 2
    """How many iterations the learing rate sheduler waits with the improvment below the threashold before reducing learing rate."""
    lr_scheduler_threshold: float = 0.01
    """The improvement threashold for the learing rate sheduler."""
    L1_lambda: float = 0.8
    """The amount of regularization to be applied"""
    fixed_pading: bool = True
    """Indicates if the same padding length should be applied to all inputs to the model.
    The max padding is then set to the longest input in the dataloader and all inputs will be padded to that length.
    If true will ignore the max_input_length value."""
    max_input_length: int = 512
    clasif_layer_to_extract: List[str] = field(default_factory=lambda: ["Dropout2", "Dropout5"])
    """List containing the names of all the classifier layer to be extracted"""
    save_model: bool = True
    """Indicates if the model should be saved to a file."""
    extract_lm_parameters: bool = False
    """Indicates if the model output should output the language model parameters. (They take up a lot of space)"""
    extract_clasif_parameters: bool = False
    """Indicates if the model output should output the classification layer parameters. (They take up a lot of space)"""
    formating_output_function: Callable = None
    """The function through which the final output will be passed. Can be used to condense attentions head and activation values 
    in order to save file space."""
    #early_stopping_parm: Dict[str, Any] = field(default_factory= lambda:{"monitor":'val_loss', 
    #                                                                     "mode":'min', 
    #                                                                     "min_delta": 0.001, 
    #                                                                     "patience": 5, 
    #                                                                     "verbose":True})
    # max number of epochs
    max_epochs: int = 10
    # number of workers for the data loaders
    num_workers: int = 8
    model_name: str = "distilbert/distilgpt2"#'distilgpt2'
    """Hugging face reference for the language model to be used."""
    model_file_name: str = "gpt2_L1"
    """Name of the file in which all model traing and evaluation results will be stored."""
    compress_results_file: bool = False
    """If true will compress the resulting file using mgzip. False is default as it takes time to compress."""
    use_lower_case: bool = True
    """Indicates if all text loaded into the data loaders should be set to lower case. (It should be. As most unfuned titles are all in capitals)"""
    path_to_data: Path = Path("Data", "pairs_with_ratings.tsv")
    """Path to the dataset name"""
    logging_root_dir: str = "./logging"
    """Logging root dir: where to save the logger outputs"""
    # Where to save the checkpoints
    #ckpt_root_dir: str = "./ckpt"
    # torch precision
    #torch_precision: str = "medium" #"highest"
    #"""Specifies the precision in float calculation that torch uses. Less precise = faster"""
    #checkpoint_param: Dict[str, Any] = field(default_factory= lambda:{"enable_version_counter":False})

class DataProcessor:

    def __init__(self, cfg: GlobalConfig):
        self.cfg: GlobalConfig = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        data_loader_train, data_loader_val, data_loader_test = self.load_data(cfg)
        self.data_loader_train = data_loader_train
        self.data_loader_val = data_loader_val
        self.data_loader_test = data_loader_test
        # Add pading token if no pading token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_data(self, cfg: GlobalConfig) -> DataLoader:
        pd_train_data = read_csv(cfg.path_to_data.as_posix(), sep=" *\t *", engine='python')
        #training_data = pd_train_data.to_dict('records').copy()
        x = concat([pd_train_data["original_title"],pd_train_data["title"]]).to_list()
        y = [0] * len(pd_train_data["original_title"])  + [1] * len(pd_train_data["title"])
        # Split data into train, validation and test
        x_train, x_validation, y_train, y_validation = train_test_split(x, 
                                                                        y, 
                                                                        random_state=100, 
                                                                        test_size=0.3, 
                                                                        stratify=y)
        x_validation, x_test, y_validation, y_test = train_test_split(x_validation, 
                                                                      y_validation, 
                                                                      random_state=100, 
                                                                      test_size=0.1,
                                                                      stratify=y_validation)
        # Place data into list of dicts
        data_train = [{"title" : title.lower() if cfg.use_lower_case else title, "target" : target} for title, target in zip(x_train, y_train)]
        data_validation = [{"title" : title.lower() if cfg.use_lower_case else title, "target" : target} for title, target in zip(x_validation, y_validation)]
        data_test = [{"title" : title.lower() if cfg.use_lower_case else title, "target" : target} for title, target in zip(x_test, y_test)]
        # If using same padding length for all inputs find the longest token sequence
        if self.cfg.fixed_pading:
            max_len = 0
            for text in data_train:
                tokens_len = len(self.tokenizer.tokenize(text["title"]))
                if tokens_len > max_len:
                    max_len = tokens_len
            for text in data_validation:
                tokens_len = len(self.tokenizer.tokenize(text["title"]))
                if tokens_len > max_len:
                    max_len = tokens_len
            for text in data_test:
                tokens_len = len(self.tokenizer.tokenize(text["title"]))
                if tokens_len > max_len:
                    max_len = tokens_len
            self.cfg.max_input_length = max_len
        # Creating data loaders
        data_loader_train = DataLoader(data_train, 
                                      shuffle=True, 
                                      batch_size=cfg.batch_size_train, 
                                      collate_fn=self.collate_fn, 
                                      num_workers= cfg.num_workers,
                                      persistent_workers=True) 
        data_loader_val = DataLoader(data_validation, 
                                    shuffle=False, 
                                    batch_size=cfg.batch_size_train, 
                                    collate_fn=self.collate_fn, 
                                    num_workers=cfg.num_workers, 
                                    persistent_workers=True)
        data_loader_test = DataLoader(data_test, 
                                    shuffle=False, 
                                    batch_size=cfg.batch_size_train, 
                                    collate_fn=self.collate_fn, 
                                    num_workers=cfg.num_workers, 
                                    persistent_workers=True)
        return data_loader_train, data_loader_val, data_loader_test

    def collate_fn(self, samples: List[Union[str, Dict[str, str]]]) -> Dict[str, Union[Tensor, None]]:
        # inputs
        samples_is_dict = isinstance(samples[0], dict)
        if samples_is_dict:
            # If list of dicts is imputed
            input_texts = [sample["title"] for sample in samples]
            encoded_input = self.tokenizer(input_texts, 
                                           add_special_tokens=True, 
                                           padding='max_length' if self.cfg.fixed_pading else 'longest', 
                                           truncation=True,
                                           max_length=self.cfg.max_input_length, 
                                           return_attention_mask=True,
                                           return_tensors='pt', 
                                           return_offsets_mapping=False, 
                                           return_token_type_ids=False,
                                           verbose=False, )
        else:
            # If simple list of texts is inputed
            encoded_input = self.tokenizer(samples, 
                                           add_special_tokens=True, 
                                           padding='max_length' if self.cfg.fixed_pading else 'longest', 
                                           truncation=True,
                                           max_length=self.cfg.max_input_length, 
                                           return_attention_mask=True,
                                           return_tensors='pt', 
                                           return_offsets_mapping=False, 
                                           return_token_type_ids=False,
                                           verbose=False, )
        # build batch
        batch = {
            'input_ids': encoded_input.input_ids.squeeze(0),
            'attention_mask': encoded_input.attention_mask.squeeze(0),
            'label_ids': tensor([sample["target"] for sample in samples]) if samples_is_dict else None, #encoded_labels,
            'input_texts' : input_texts
        }
        #pprint(batch)
        return batch

    def decode_labels(self, model_prdiction: list[int]) -> List[List[str]]:
        return ["Comedy" if res == 0 else "Serious" for res in model_prdiction]

# First, a classifier component (a simple PyTorch NN module)
class ClassifierComponent(nn.Module):

    def __init__(self, input_dim: int, dropout: float, layer_list: List[str] = None):
        super().__init__()
        k = 2
        self.nn = nn.Sequential(nn.Linear(input_dim, input_dim * k),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(input_dim * k, input_dim),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(input_dim, 2))
        # Find the indexes of the layers that we want to extract
        #self.returned_layer_activation_indexes: List[int] = []
        #"""List of indexes of the layers we wish to extract"""
        #self.include_input: bool = "input" in layer_list
        #"""If activation output should include inputs"""
        #for idx, layer in enumerate(self.nn):
        #    layer_name = layer._get_name() + str(idx)
        #    if layer_list != None:
        #        if layer_name in layer_list:
        #            self.returned_layer_activation_indexes.append(idx)
        #    # If the list is set to None then return all layers
        #    else:
        #        self.returned_layer_activation_indexes.append(idx)

    def forward(self, X):
        """
        :param X:   B x d
        :return:
        """
        activations = {}
        for idx, layer in enumerate(self.nn):
            X = layer(X)
            activations[layer._get_name() + str(idx)] = X
        return X, activations #self.nn(X)

# Define Lightning Module
class TextClassifier(LightningModule):
    def __init__(self, cfg: GlobalConfig, num_labels=2):
        super().__init__()
        self.cfg = cfg
        self.model = AutoModel.from_pretrained(cfg.model_name, output_attentions=True)
        #self.model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, output_attentions=True)
        #self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)
        self.classifier = ClassifierComponent(self.model.config.hidden_size, cfg.dropout)
        self.lr = cfg.lr
        self.criterion = nn.CrossEntropyLoss()
        # Set pad token if missing
        if self.model.config.pad_token_id == None:
            self.model.config.pad_token_id = self.model.config.eos_token_id # Use end of sequence token as padding token
        # Calculate the number of weights in the model use for L1 regularization
        self.nweights = 0
        """Number of none bias weights"""
        for name,weights in self.named_parameters():
            if 'bias' not in name:
                self.nweights = self.nweights + weights.numel()
        #print(f'Total number of weights in the model = {self.nweights}')

    def calc_L1(self) -> Tensor:
        # Calculate L1 term
        L1_term = tensor(0., requires_grad=True)
        for name, weights in self.named_parameters():
            if 'bias' not in name:
                weights_sum = sum(abs(weights))
                L1_term = L1_term + weights_sum
        return L1_term / self.nweights

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids, output_hidden_states=True)
        pooled_output = outputs.last_hidden_state.mean(dim=1)  # Use last token's hidden state
        #print(outputs.last_hidden_state.shape)
        logits, activations = self.classifier(pooled_output)
        return logits, outputs["hidden_states"], activations #, self.attention_outputs

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        logits, _, _= self(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        loss = self.criterion(logits, batch['label_ids'])
        if self.cfg.L1_lambda != None:
            loss = loss - self.calc_L1() * self.cfg.L1_lambda
        self.log("train_loss", loss, on_step=True, on_epoch=True, batch_size=self.cfg.batch_size_train)
        return loss

    def validation_step(self, batch, batch_ix):
        logits, _, _= self(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        loss = self.criterion(logits, batch['label_ids'])
        self.log("val_loss", loss, on_step=False, on_epoch=True, batch_size=self.cfg.batch_size_train)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        out = {'optimizer' : optimizer,
               'lr_scheduler': {'scheduler': ReduceLROnPlateau(optimizer, 
                                                                 mode="min", 
                                                                 factor=self.cfg.lr_scheduler_factor, 
                                                                 patience=self.cfg.lr_scheduler_patience,
                                                                 threshold=self.cfg.lr_scheduler_threshold),
                                'monitor': 'train_loss',
                                'interval': 'epoch',
                                'frequency': 1}}
        return out
    
    def train_model(self, training_data_loader: DataLoader, validation_data_loader: DataLoader = None) -> None:
        self.train()
        logger = CSVLogger(save_dir=self.cfg.logging_root_dir, name=self.cfg.model_file_name)
        trainer = Trainer(max_epochs=self.cfg.max_epochs,
                          logger=logger)
        trainer.fit(self, training_data_loader, validation_data_loader)
        with open(Path(self.cfg.logging_root_dir) / Path(self.cfg.model_file_name) / Path(f"{self.cfg.model_file_name}_cfg"), "wb+") as file:
                dump(self.cfg, file)
                #print(f"Saved cfg object to: {self.cfg.path_to_save_cfg}")

    # Evaluate accuracy
    def evaluate(self, 
                 dataloader: DataLoader,
                 #save_output_file_name: str = None
                 ) -> Dict[str, Any]:
        self.eval().cuda(device=0)
        all_preds, all_labels, all_attentions, all_tokens, all_classifier_activations = [], [], [], [], []
        with no_grad():
            print("Making predictions...")
            # For each batch extract all relevent information
            for batch in tqdm(dataloader):
                logits, attentions, classifier_activations = self(input_ids=batch['input_ids'].cuda(0), 
                                                                  attention_mask=batch['attention_mask'].cuda(0))
                #classifier_activations.extend(logits.clone().detach())
                preds = argmax(logits.cpu(), dim=1).cpu().numpy()
                labels = batch['label_ids'].cpu().numpy()
                #temp = batch['attention_mask'].cpu()
                #del batch["attention_mask"], temp
                all_preds.extend(preds)
                all_labels.extend(labels)
                # The tuple (dimention 1) is the number of attention layers then [batch size, nb attention heads, len(text), len(text)]
                all_attentions.append([att.cpu() for att in attentions])
                all_tokens.extend(batch['input_ids'].cpu())
                all_classifier_activations.append(classifier_activations)
                del logits, attentions, classifier_activations
            del batch
        # Filter out layers that are not tracked
        all_classifier_activations = unpack_activations(all_classifier_activations)
        for idx in range(len(all_classifier_activations)):
            new_dict = {}
            for key in self.cfg.clasif_layer_to_extract:
                new_dict.update({key : all_classifier_activations[idx][key]})
            all_classifier_activations[idx] = new_dict
        out = {"accuracy" : accuracy_score(all_labels, all_preds), 
                "predictions" : all_preds,
                "tokens" : all_tokens,
                "targets" : all_labels,
                "attentions" : all_attentions,
                "classif_activations" : all_classifier_activations}
        # Add model parameter values
        out.update({"parameters" : {"languag_model" : self.model.state_dict() if self.cfg.extract_lm_parameters else None,
                                    "classifier" : self.classifier.nn.state_dict() if self.cfg.extract_clasif_parameters else None}})
        out["parameters"] = unbind_parm_from_vram(out["parameters"])
        # Apply finale transformation on outputed data such as averaging all the activation head values.
        if self.cfg.formating_output_function != None:
            out = self.cfg.formating_output_function(out)
        # Save the output if path specified
        if self.cfg.save_model:
            #if self.cfg.compress_results_file:
            #    with mgzip.open((Path(self.cfg.logging_root_dir) / Path(self.cfg.model_file_name) / f"{self.cfg.model_file_name}_res.bt").as_posix(), "wb") as file:
            #        dump(out, file)
            #        file.close()
            #    print(f"Saved output to: {self.cfg.model_file_name}_res.bt")
            #else:
                with open(Path(self.cfg.logging_root_dir) / Path(self.cfg.model_file_name) / f"{self.cfg.model_file_name}_res.pkl", "wb") as file:
                    dump(out, file)
                    file.close()
                print(f"Saved output to: {self.cfg.model_file_name}_res.pkl")
        empty_cache()
        return out

def unbind_parm_from_vram(param_dict: Dict[str, Dict[str, Tensor]]) -> Dict[str, Dict[str, Tensor]]:
    """Used to remove parameter values from vram memory."""
    for model_key in param_dict:
        for key in param_dict[model_key]:
            param_dict[model_key][key] = param_dict[model_key][key].cpu()
    return param_dict

def load_and_evaluate(path_to_model: str,
                      cfg: GlobalConfig, 
                      dataloader: DataLoader, 
                      save_output_path: str = None) -> Dict[str, Any]:
    model = TextClassifier.load_from_checkpoint(path_to_model, cfg=cfg)
    model.eval()
    all_preds, all_labels, all_attentions, all_tokens, all_classifier_activations = [], [], [], [], []
    with no_grad():
        print("Making predictions...")
        # For each batch extract all relevent information
        for batch in tqdm(dataloader):
            logits, attentions, classifier_activations = model(input_ids=batch['input_ids'].cuda(0), attention_mask=batch['attention_mask'].cuda(0))
            #classifier_activations.extend(logits.clone().detach())
            preds = argmax(logits.cpu(), dim=1).cpu().numpy()
            labels = batch['label_ids'].cpu().numpy()
            #temp = batch['attention_mask'].cpu()
            #del batch["attention_mask"], temp
            all_preds.extend(preds)
            all_labels.extend(labels)
            # The tuple (dimention 1) is the number of attention layers then [batch size, nb attention heads, len(text), len(text)]
            all_attentions.append([att.cpu() for att in attentions])
            all_tokens.extend(batch['input_ids'].cpu())
            all_classifier_activations.append(classifier_activations)
            del logits, attentions, classifier_activations
        del batch
        # Filter out layers that are not tracked
        all_classifier_activations = unpack_activations(all_classifier_activations)
        for idx in range(len(all_classifier_activations)):
            new_dict = {}
            for key in cfg.clasif_layer_to_extract:
                new_dict.update({key : all_classifier_activations[idx][key]})
            all_classifier_activations[idx] = new_dict
        out = {"accuracy" : accuracy_score(all_labels, all_preds), 
                "predictions" : all_preds,
                "tokens" : all_tokens,
                "targets" : all_labels,
                "attentions" : all_attentions,
                "classif_activations" : all_classifier_activations}
        # Add model parameter values
        out.update({"parameters" : {"languag_model" : model.model.state_dict() if model.cfg.extract_lm_parameters else None,
                                    "classifier" : model.classifier.nn.state_dict() if model.cfg.extract_clasif_parameters else None}})
        out["parameters"] = unbind_parm_from_vram(out["parameters"])
        # Apply finale transformation on outputed data such as averaging all the activation head values.
        if cfg.formating_output_function != None:
            out = cfg.formating_output_function(out)
        # Save the output if path specified
        if cfg.save_model:
            #if self.cfg.compress_results_file:
            #    with mgzip.open((Path(self.cfg.logging_root_dir) / Path(self.cfg.model_file_name) / f"{self.cfg.model_file_name}_res.bt").as_posix(), "wb") as file:
            #        dump(out, file)
            #        file.close()
            #    print(f"Saved output to: {self.cfg.model_file_name}_res.bt")
            #else:
                with open(save_output_path, "wb") as file:
                    dump(out, file)
                    file.close()
                print(f"Saved output to: {cfg.model_file_name}_res.pkl")
        empty_cache()
        return out

def unpack_attentions(attentions: List[Tensor], bertviz_compatible: bool = False) -> List[List[Tensor]]:
    """Unpacks batched attentions placing each one into it's own tensor.

    ## Args:
        - attentions (list): The list of batched attentions from model prediction.
        - bertviz_compatible (bool, optional): Indicates if each atention head should be placed in its own tensor to make it compatible with
        the bertviz package

    ## Returns:
        if bertviz_compatible == False:
        - list: List of model attentions.
        out[0][attention head layer][nb of attention head][nb tokens][nb tokens]

        if bertviz_compatible == True:
        - list: List of model attentions.
        out[0][attention head layer][usless dimention][nb of attention head][nb tokens][nb tokens]
    """    
    out = []
    for batch_idx in range(len(attentions)):
        for obs_idx in range(len(attentions[batch_idx][0])):
            if not bertviz_compatible:
                out.append([att[obs_idx].cpu() for att in attentions[batch_idx]])
            else:
                out.append([unsqueeze(att[obs_idx], dim=0).cpu() for att in attentions[batch_idx]])
    return out

def unpack_activations(activations: List[Dict[str, Tensor]]) -> List[Dict[str, Tensor]]:
    """Unpacks batched attentions placing each one into it's own tensor.
    ## Args:
        - activations (list): The list of batched activations from model prediction.
    ## Returns:
        - list: List of dictionaries with each dictionary containing
    """    
    out: List[Dict[str, Tensor]] = []
    keys = list(activations[0].keys())
    batch_size = activations[0][keys[0]].shape[0]
    last_batch_size = activations[-1][keys[0]].shape[0]
    final_nb = batch_size * (len(activations) - 1) + last_batch_size
    for _ in range(final_nb):
        out.append({})
    for layer_key in keys:
        for batch_idx in range(len(activations)):
            for in_batch_idx in range(activations[batch_idx][keys[0]].shape[0]):
                out[in_batch_idx + (batch_size * batch_idx)].update({layer_key : activations[batch_idx][layer_key][in_batch_idx].cpu()})
    return out

def sum_head(tensor1: Tensor, tensor2: Tensor) -> Tensor:
    return tensor1 + tensor2

def max_head(tensor1: Tensor, tensor2: Tensor) -> Tensor:
    return max(tensor1, tensor2)

def apply_to_layers(layer1: List[Tensor], layer2: List[Tensor], fun: Callable) -> List[Tensor]:
    with ThreadPoolExecutor() as executor:
        res = list(executor.map(fun, layer1, layer2))
    return res

def apply_to_attentions(attentions: List[List[Tensor]], fun: Callable) -> List[Tensor]:
    """Given a list of attention head outputs applies the given function to all attention values in the list cumulatively.
    Such as summing or finding the max the values of all the attention heads."""
    out = []
    for idx in range(len(attentions[0])):
        out.append(zeros(attentions[0][0].shape))
    for idx in range(len(attentions)):
        out = apply_to_layers(out, attentions[idx], fun)
    return out

def apply_to_attentions_cond(attentions: List[List[Tensor]], 
                             fun: Callable, 
                             pred_vect: Iterable,
                             target_vect: Iterable) -> List[Tensor]:
    """Given a list of attention head outputs applies the given function to all attention values in the list cumulatively.
    Such as summing or finding the max the values of all the attention heads."""
    att_is_fun = []
    att_is_unfun = []
    att_pred_fun = []
    att_pred_unfun = []
    # debug
    #debug = [0,0,0,0]
    for idx in range(len(attentions[0])):
        att_is_fun.append(zeros(attentions[0][0].shape))
        att_is_unfun.append(zeros(attentions[0][0].shape))
        att_pred_fun.append(zeros(attentions[0][0].shape))
        att_pred_unfun.append(zeros(attentions[0][0].shape))
    for idx in range(len(attentions)):
        if target_vect[idx] == 0:
            att_is_fun = apply_to_layers(att_is_fun, attentions[idx], fun)
            #debug[0] += 1
        else:
            att_is_unfun = apply_to_layers(att_is_unfun, attentions[idx], fun)
            #debug[1] += 1
        if pred_vect[idx] == 0:
            att_pred_fun = apply_to_layers(att_is_unfun, attentions[idx], fun)
            #debug[2] += 1
        else:
            att_pred_unfun = apply_to_layers(att_is_unfun, attentions[idx], fun)
            #debug[3] += 1
    #print(f"debug val: {debug}")
    return att_is_fun, att_is_unfun, att_pred_fun, att_pred_unfun

def div_attentions(attentions: List[Tensor], val: int) -> List[Tensor]:
    """Divides each values in the attention head by the given value"""
    with ThreadPoolExecutor() as executor:
        res = list(executor.map(lambda head: head / val, attentions))
    return res

def apply_to_2_activations(act1: Dict[str, Tensor], act2: Dict[str, Tensor], fun: Callable) -> Dict[str, Tensor]:
    out = {}
    with ThreadPoolExecutor() as executor:
        res = list(executor.map(fun, act1.values(), act2.values()))
    for idx, key in enumerate(act1):
        out.update({key : res[idx]})
    return out

def apply_to_all_activation(activations: List[Dict[str, Tensor]], fun: Callable) -> Dict[str, Tensor]:
    """Apply function to all activations in list. Such as sum all activation values."""
    out = {}
    for key in activations[0]:
        out.update({key : zeros(activations[0][key].shape)})
    for act in activations:
        out = apply_to_2_activations(out, act, fun)
    return out

def format_output(res: Dict[str, Any]) -> Dict[str, Any]:
    res["attentions"] = unpack_attentions(res["attentions"])
    #nb_obs = len(res["attentions"])
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
