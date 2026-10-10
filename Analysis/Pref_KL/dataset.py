from torch.utils.data import Dataset
import json


class CustomDataset(Dataset):
    def __init__(self,
                 data_path,
                 data_format='MIRAGE'):
        super().__init__()
        self.data = json.load(open(data_path, 'r'))
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
        
        return {
            'human': human_texts,
            'llm': llm_texts
        }
    
    def __len__(self):
        return len(self.data) if self.data_format == 'MIRAGE' else len(self.data['human'])


class PerturbedDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        raw_data = json.load(open(data_path, 'r'))
        # data is stored in the 'data' key, matching generate_perturbs.py output
        self.data = raw_data['data']

    def __getitem__(self, index):
        item = self.data[index]
        # Adapt keys from generate_perturbs.py ('llm') to what eval_detect_gpt.py expects ('sampled')
        return {
            'human': item['human'],
            'sampled': item['llm'],
            'perturbed_human': item['perturbed_human'],
            'perturbed_sampled': item['perturbed_llm'],
        }

    def collate_fn(self, batch):
        human_texts = [item['human'] for item in batch]
        sampled_texts = [item['sampled'] for item in batch]
        perturbed_human = [item['perturbed_human'] for item in batch]
        perturbed_sampled = [item['perturbed_sampled'] for item in batch]

        return {
            'human': human_texts,
            'sampled': sampled_texts,
            'perturbed_human': perturbed_human,
            'perturbed_sampled': perturbed_sampled
        }

    def __len__(self):
        return len(self.data)