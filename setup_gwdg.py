from tokenizer import BertTokenizer
from src.models import BertModel

if __name__ == "__main__":
    # Download files
    BertTokenizer.from_pretrained('bert-base-uncased')
    BertModel.from_pretrained('bert-base-uncased')