# Data and models
# https://drive.google.com/drive/u/2/folders/15aQejnRjGDa5OEmzegyd6iG0Nv9Nur3z

from typing import NamedTuple, Sequence, Any, List
import string
from torch.utils.data import Dataset, DataLoader

# Label constants
BEGIN = "B"
INSIDE = "I"
OUTSIDE = "O"
DELIM = "-"
# Where CS means "CONCLUDING STATEMENT"
CLASSES = ['LEAD', 'POSITION', 'CLAIM', 'COUNTERCLAIM', 'REBUTTAL', 'EVIDENCE', 'CS']
PUNCTUATIONS = string.punctuation

class Mention(NamedTuple):
    """An immutable mention with an entity type and start/end indices.

    Like standard slicing operations, the start index is inclusive
    and the end index is inclusive. For example, if the tokens of
    a sentence are ["Brandeis", "University", "is", "awesome"],
    an ORG mention for the first two tokens would have a start
    index of 0 and an end index of 2. Note that the length of the
    mention is simply end - start."""

    entity_type: str
    start: int
    end: int

class AnnotatedDoc:
    """
    Contains a sequence of tokens & a sequence of Mentions of an essay
    """
    def __init__(self, doc_id: str, tokens: Sequence[str], mentions: Sequence[Mention]) -> None:
        if not tokens:
            raise ValueError("No tokens provided")
        # We defensively copy arguments just to be safe
        self.doc_id: str = doc_id
        self.tokens: tuple[str, ...] = tuple(tokens)
        self.mentions: tuple[Mention, ...] = tuple(mentions)

    def __str__(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        return f"AnnotatedDoc({self.doc_id}, {self.tokens}, {self.mentions})"

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, AnnotatedDoc)
            and self.doc_id == other.doc_id
            and self.tokens == other.tokens
            and self.mentions == other.mentions
        )

    # We're intentionally breaking hash behavior because you should not be using it.
    def __hash__(self) -> int:
        raise ValueError("Don't try to hash AnnotatedDoc")
        
        
class StudentWritingDataset(Dataset):
    """
    A class that extends torch.utils.data.Dataset
    Used for making the training, dev, and test test
    """
    def __init__(self, model_input_data):
        self.model_input_data = model_input_data
    
    # must be overwritten
    def __len__(self):
        return len(self.model_input_data)

    # must be overwritten
    def __getitem__(self, index):
        return self.model_input_data[index]

# given a sequence of tokens & mentions, return a list of BIO labels
def encode_bio(tokens: Sequence[str], mentions: Sequence[Mention]) -> List[str]:
    encoded_bio = [OUTSIDE] * len(tokens)
    for mention in mentions:
        encoded_bio[mention.start] = BEGIN + DELIM + mention.entity_type
        if mention.end - mention.start > 1:
            for pos in range(mention.start + 1, mention.end):
                encoded_bio[pos] = INSIDE + DELIM + mention.entity_type
    return encoded_bio

# Given a sequence of BIO labels, return a sequence of Mentions
def decode_bio(labels: List[str]) -> List[Mention]:
    mentions = []
    # normalize labels
    for i in range(len(labels)):
        if labels[i].startswith(INSIDE):
            if i == 0 or labels[i - 1] == OUTSIDE or labels[i - 1].split(DELIM)[1] != labels[i].split(DELIM)[1]:
                labels[i] = BEGIN + DELIM + labels[i].split(DELIM)[1]

    # iterate over the normalized labels
    temp_start = -1
    for i in range(len(labels)):
        # if it is a 'B' label, check if there is an 'I' label that follows it
        # if true, directly add a mention
        # if not, record the position of this 'B' label, as we need the start when creating a mention
        if labels[i].startswith(BEGIN):
            # we reach the last position
            if i + 1 == len(labels):
                mentions.append(Mention(labels[i].split(DELIM)[1], i, i + 1))
            elif i + 1 < len(labels) and labels[i + 1] != INSIDE + DELIM + labels[i].split(DELIM)[1]:
                mentions.append(Mention(labels[i].split(DELIM)[1], i, i + 1))
            else:
                temp_start = i
        # if it is an 'I' label, check if there is an 'I' label that follows it
        # if true, use the recorded temp_start and the position info to create a mention
        elif labels[i].startswith(INSIDE):
            # we reach the last position
            if i + 1 == len(labels):
                mentions.append(Mention(labels[i].split(DELIM)[1], temp_start, i + 1))
            elif i + 1 < len(labels) and labels[i + 1] != labels[i]:
                mentions.append(Mention(labels[i].split(DELIM)[1], temp_start, i + 1))
    return mentions

# Auto-batching
def make_data_loaders(processed_data_file, batch_size):
    open_file = open(processed_data_file, "rb")
    model_input = pickle.load(open_file)
    train_dev_split_point = int(len(model_input) * 0.8)
    dev_test_split_point = int(len(model_input) * 0.9)
    train_data = StudentWritingDataset(model_input[: train_dev_split_point])
    dev_data = StudentWritingDataset(model_input[train_dev_split_point: dev_test_split_point])
    test_data = StudentWritingDataset(model_input[dev_test_split_point: ])
    train_set = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    dev_set = DataLoader(dev_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_set = DataLoader(test_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    
    return train_set, dev_set, test_set

class Trainer:
    """
    Helps in training the model and evaluating its performance on the dev set
    metrics here are [token-level], not [mention-level]
    """
    def __init__(self, config, train_loader, dev_loader, test_loader, save_model_path):
        # pass our training configuration, training loader and dev loader 
        # to a Trainer object
        self.config = config
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.train_loss = []
        self.dev_loss = []
        self.save_model_path = save_model_path
    
    # train the model, save the model, and evaluate its performance on dev set
    def train(self):
        # gpu training
        self.config.model = self.config.model.to(self.config.device) 
        self.config.model.train()
        for i in range(self.config.n_epochs):
            print("Epoch {:} out of {:}".format(i + 1, self.config.n_epochs))
            self.train_for_single_epoch()
            self.run_on_dev_or_test(dataset='dev')
        self.run_on_dev_or_test(dataset='test')
    
    # helper method, train the model for one epoch
    def train_for_single_epoch(self):
        total_loss = 0
        # the two lists are used for recording our predictions / actual labels
        # and computing the metrics 
        predictions_list = []
        labels_list = []
        last_idx = 0
        # batch training
        for idx, batch in enumerate(self.train_loader):
            self.config.optimizer.zero_grad()
            # gpu training
            ids = batch['input_ids'].to(self.config.device, dtype = torch.long)
            mask = batch['attention_mask'].to(self.config.device, dtype = torch.long)
            labels = batch['labels'].to(self.config.device, dtype = torch.long)
            
            loss, logits = self.config.model(input_ids=ids, attention_mask=mask, labels=labels,
                                               return_dict=False)
            
            logits = logits.view(-1, self.config.model.num_labels)
            # find invalid labels and filter them
            invalid_label_mask = labels.view(-1) != -100
            labels = torch.masked_select(labels.view(-1), invalid_label_mask)
            # get the argmax of the logits, which are our predictions
            predictions = torch.masked_select(torch.argmax(logits, axis=1), invalid_label_mask)
            predictions_list.extend(predictions.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
            total_loss += loss.item()
            self.train_loss.append(loss.item())
            loss.backward()
            self.config.optimizer.step()
            last_idx = idx
            
        # print the metrics after processing every 200 batches
        print("Average Training Loss: {}".format(total_loss / (last_idx + 1)))
        print("Training Acc: {}". format(f1_score(labels_list, predictions_list, average='micro')))
        print("Training Macro F1: {}". format(f1_score(labels_list, predictions_list, average='macro')))
    
    # basically the same as training, but no backward propagation & parameter update
    def run_on_dev_or_test(self, dataset):
        assert dataset in ['dev', 'test']
        if dataset == 'test':
            self.config.model.load_state_dict(torch.load(self.save_model_path))
        model = self.config.model
        total_loss = 0
        last_idx = 0
        predictions_list = []
        labels_list = []
        with torch.no_grad():    
            if dataset == 'dev':
                loader = self.dev_loader
            else:
                loader = self.test_loader
            for idx, batch in enumerate(loader):
                ids = batch['input_ids'].to(self.config.device, dtype = torch.long)
                mask = batch['attention_mask'].to(self.config.device, dtype = torch.long)
                labels = batch['labels'].to(self.config.device, dtype = torch.long)
                loss, logits = model(input_ids=ids, attention_mask=mask, labels=labels,
                                     return_dict=False)
                logits = logits.view(-1, model.num_labels)
                # find invalid labels and filter them
                invalid_label_mask = labels.view(-1) != -100
                labels = torch.masked_select(labels.view(-1), invalid_label_mask)
                predictions = torch.masked_select(torch.argmax(logits, axis=1), invalid_label_mask)
                predictions_list.extend(predictions.cpu().numpy())
                labels_list.extend(labels.cpu().numpy())
                total_loss += loss.item()
                self.dev_loss.append(loss.item())
                if dataset == 'dev':
                    if len(self.dev_loss) == 0 or loss.item() < min(self.dev_loss):
                        if self.save_model_path:
                            torch.save(model.state_dict(), self.save_model_path)
                last_idx = idx
            
            if dataset == 'dev':
                print("Average Dev loss : {}".format(total_loss / (last_idx + 1)))
                print("Dev Acc: {}". format(f1_score(labels_list, predictions_list, average='micro')))
                print("Dev Macro F1 : {}". format(f1_score(labels_list, predictions_list, average='macro'))) 
            else:
                print("Average Test loss : {}".format(total_loss / (last_idx + 1)))
                print("Test Acc: {}". format(f1_score(labels_list, predictions_list, average='micro')))
                print("Test Macro F1 : {}". format(f1_score(labels_list, predictions_list, average='macro'))) 