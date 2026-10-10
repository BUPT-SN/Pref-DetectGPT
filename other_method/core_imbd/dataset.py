from torch.utils.data import Dataset
import json


class CustomDataset(Dataset):
    def __init__(self,
                 data_path,
                 scoring_tokenizer,
                 reference_tokenizer=None,
                 data_format='MIRAGE'):
        super().__init__()
        self.data = json.load(open(data_path, 'r'))
        self.scoring_tokenizer = scoring_tokenizer
        self.reference_tokenizer = reference_tokenizer
        self.data_format = data_format

    def __getitem__(self, index):
        if self.data_format == 'MIRAGE':
            human_text = self.data[index]['human']
            llm_text = self.data[index]['llm']
        else:
            human_text = self.data['human'][index]
            llm_text = self.data['llm'][index]
        return {
            'human': human_text,
            'llm': llm_text
        }
    
    def collate_fn(self, batch):
        human_texts = [item['human'] for item in batch]
        llm_texts = [item['llm'] for item in batch]

        # Tokenize batches with padding and truncation
        # 限制最大长度为 512 token
        human_tokens_for_scoring_model = self.scoring_tokenizer(
            human_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            return_token_type_ids=False,
        )
        llm_tokens_for_scoring_model = self.scoring_tokenizer(
            llm_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            return_token_type_ids=False,
        )
        human_tokens_for_reference_model = self.reference_tokenizer(
            human_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            return_token_type_ids=False,
        ) if self.reference_tokenizer is not None else {'input_ids': None, 'attention_mask': None}
        llm_tokens_for_reference_model = self.reference_tokenizer(
            llm_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            return_token_type_ids=False,
        ) if self.reference_tokenizer is not None else {'input_ids': None, 'attention_mask': None}

        return {
            'scoring':{
                'human_input_ids': human_tokens_for_scoring_model["input_ids"],
                'human_attention_mask': human_tokens_for_scoring_model["attention_mask"],
                'llm_input_ids': llm_tokens_for_scoring_model["input_ids"],
                'llm_attention_mask': llm_tokens_for_scoring_model["attention_mask"]
            },
            'reference':{
                'human_input_ids': human_tokens_for_reference_model["input_ids"],
                'human_attention_mask': human_tokens_for_reference_model["attention_mask"],
                'llm_input_ids': llm_tokens_for_reference_model["input_ids"],
                'llm_attention_mask': llm_tokens_for_reference_model["attention_mask"]
            }
        }
    
    def __len__(self):
        return len(self.data) if self.data_format == 'MIRAGE' else len(self.data['human'])