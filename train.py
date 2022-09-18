import sys
import torch
import wandb
import numpy as np
import pandas as pd
import transformers
from pathlib import Path
from collections import defaultdict
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support, classification_report, confusion_matrix
from transformers import RobertaTokenizer, AutoTokenizer, RobertaConfig, RobertaForSequenceClassification, AdamW, Trainer, TrainingArguments, EarlyStoppingCallback
from transformers.trainer_utils import set_seed

"""
To run: python train.py <window_size>
Set data path/preprocessing between lines 35-45
Set path of output model folder/experiment name in lines 150-160
5 model checkpoints will be saved, and results on dev set will be printed after training
"""

set_seed(1)
window = int(sys.argv[1])

def read_files(filenames_file):
    filenames = []
    with open(filenames_file) as namelist:
        for line in namelist:
            filenames.append(line.strip())
    print(filenames, len(filenames))
    utterances = []
    utt_contexts = []
    topics = []
    tasks = []
    utterances_with_contexts = []

    for file in Path('adjudicated/').glob("*.xlsx"):
        if file.name in filenames:
            df = pd.read_excel(file)
            #print(df.columns)
            file_utterances = []
            file_topics = []
            #print(df['Topic (1,0,-1)'].value_counts())
            for i, row in df.iterrows():
                utt = str(row['Utterance'])
                spk = str(row['Speaker'])
                #print(spk)
                utt = 'Speaker: ' + spk + '<-' + utt
                utterances.append(utt)
                file_utterances.append(utt)
                # sanity check for topic being nan
                if row['Topic (1,0,-1)'] == row['Topic (1,0,-1)']:
                    topic = int(row['Topic (1,0,-1)'])
                    topic_adj = int(row['Adj'])
                    topic = topic_adj
                    if topic == -1:
                        topic = 2
                    #print(topic)
                else:
                    topic = 0
                topics.append(topic)
                file_topics.append(topic)
            curr_utt_contexts = get_utt_contexts(file_utterances, file_topics)
            utt_contexts.extend(curr_utt_contexts)

            
    print(len(utt_contexts), len(topics), len(tasks), len(curr_utt_contexts))
    assert len(utt_contexts) == len(topics)

    #adj_topics = get_adjudicated_topics()
    #assert len(adj_topics) == len(topics), len(topics)
    #topics = adj_topics  
    return utt_contexts, topics, tasks


def get_utt_contexts(utterances, topics):
  contexts = []
  topic_cs = []
  task_cs = []
  #window = len(utterances)
  #print('Num utts in file:', len(utterances))
  for i, utt in enumerate(utterances):
    context = utt + '-> Context: '
    topic_context = '-> Topic: '
    for j in range(1, window+1):
      if i >= j:
        prev_utt_j = utterances[i-j]
        context = context + prev_utt_j + '->'
        topic_context = topic_context + str(topics[i-j])
    #context = context + topic_context
    #context = topic_context
    contexts.append(context)
    topic_cs.append(topic_context)
  return contexts

train_utts, train_topic_labels, train_task_labels = read_files('train_names.txt')
topic_dict = defaultdict(int)
for l in train_topic_labels:
  topic_dict[l] += 1
task_dict = defaultdict(int)
for l in train_task_labels:
  task_dict[l] += 1
print(topic_dict, task_dict)
print('Train utterances', len(train_utts), len(train_topic_labels), train_utts[:5])
dev_utts, dev_topic_labels, dev_task_labels = read_files('dev_names.txt')

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
#tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(train_utts, truncation=True, padding=True)
dev_encodings = tokenizer(dev_utts, truncation=True, padding=True)

class TopicDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = TopicDataset(train_encodings, train_topic_labels)
dev_dataset = TopicDataset(dev_encodings, dev_topic_labels)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

config = transformers.AutoConfig.from_pretrained('roberta-base')
#config = transformers.AutoConfig.from_pretrained('benjaminbeilharz/bert-base-uncased-dailydialog-turn-classifier')
#config = transformers.AutoConfig.from_pretrained('DeepPavlov/bert-base-cased-conversational')
config.num_labels = 3
#model = transformers.AutoModelForSequenceClassification.from_pretrained('roberta-base', config=config)
#model = transformers.AutoModelForSequenceClassification.from_pretrained('benjaminbeilharz/bert-base-uncased-dailydialog-turn-classifier')
#model = transformers.AutoModelForSequenceClassification.from_pretrained('DeepPavlov/bert-base-cased-conversational')
model.to(device)
model.train()

def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average='macro')
    precision = precision_score(y_true=labels, y_pred=pred, average='macro')
    f1 = f1_score(y_true=labels, y_pred=pred, average='macro')
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

batch_size = 64
exp_name = 'roberta-topic-adjudicated-context-' + str(window)
training_args = TrainingArguments(
    output_dir='/rc_scratch/anga5835/off-topic/' + exp_name,          # output directory
    num_train_epochs=50,              # total number of training epochs
    per_device_train_batch_size=batch_size,  # batch size per device during training
    per_device_eval_batch_size=batch_size,   # batch size for evaluation
    warmup_steps=50,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=100,
    save_total_limit=5,
    report_to='wandb',
    run_name=exp_name,
    #gradient_accumulation_steps=4,
    metric_for_best_model = 'eval_f1',
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,                         # the instantiated ? Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=dev_dataset, 
    compute_metrics=compute_metrics, 
    #callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
)

trainer.train()
trainer.save_model()
wandb.finish()

model.eval()
dev_loader = DataLoader(dev_dataset, batch_size=batch_size)
criterion = torch.nn.CrossEntropyLoss()
gold_labels = []
total_pred = []
total_gold = []
loss = 0

for i, batch in enumerate(dev_loader):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    #print('Number and type of labels: ', len(labels), labels)
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    logits = outputs.logits

    pred = outputs[1]
    preds = torch.argmax(pred, dim=-1)

    for j, class_label in enumerate(preds):
        predicted = preds[j].item()
        gold = labels[j].item()
        total_pred.append(predicted)
        total_gold.append(gold)

dev_acc = accuracy_score(total_gold, total_pred)

print(dev_acc)
print(classification_report(total_gold, total_pred))
