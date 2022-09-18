import sys
import csv
import json
import copy
import pandas as pd
import transformers
from pathlib import Path
from collections import defaultdict
from transformers import RobertaTokenizer, TextClassificationPipeline

def load_model(model_path):
  # configure label names for predicted ids
  id2label = {0: 'NA', 1: 'On', 2: 'Off'}
  config = transformers.AutoConfig.from_pretrained('roberta-base', id2label=id2label)
  config.num_labels = 3
  model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path, config=config)
  tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
  # initialize huggingface inference pipeline
  pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)
  return pipeline

def classify_topic(pipeline, utterance):
  # returns a dict with "label" and "score"
  scores_dict = pipeline(utterance)
  pred_label = scores_dict[0]["label"]
  return pred_label

def read_files(filenames_file):
    filenames = []
    with open(filenames_file) as namelist:
        for line in namelist:
            filenames.append(line.strip())
    #print(filenames, len(filenames))
    utterances = []
    utt_contexts = []
    topics = []
    tasks = []
    column_names = ['Speaker', 'Time', 'Utterance', 'Ignore1', 'WER', 'Ignore2']
    file_dict = {}

    for file in Path('Sensor-Immersion-Test-Set-ASR-Human-transcript-alignment/').glob("*.tsv"):
            df = pd.read_csv(file, sep='\t', names=column_names)
            print(df.columns)
            file_dict[file.name] = {'Time': [],
                                    'Speaker': [],
                                    'Utterance': [],
                                    'Topic_Contexts': [],
                                    'Task_Contexts': [],
                                    'Topic': [],
                                    'Task': []}
            print(df.head)
            file_utterances = []
            file_topics = []
            for i, row in df.iterrows():
                print(row['Utterance'], row['Ignore2'], row['Ignore1'])
                utt = str(row['Utterance'])
                time = str(row['Time'])
                file_dict[file.name]['Time'].append(time)
                file_dict[file.name]['Utterance'].append(utt)
                spk = str(row['Speaker'])
                file_dict[file.name]['Speaker'].append(spk)
                utterances.append(utt)
                file_utterances.append(utt)
            topic_contexts, task_contexts = get_utt_contexts(file_utterances)
            #utt_contexts.append(curr_utt_contexts)
            file_dict[file.name]['Topic_Contexts'] = topic_contexts
            file_dict[file.name]['Task_Contexts'] = task_contexts

    #print(len(utt_contexts), len(curr_utt_contexts))
    return file_dict

def get_utt_contexts(utterances):
  topic_contexts = []
  task_contexts = []
  for i, utt in enumerate(utterances):
    topic_context = utt + '-> Context: '
    task_context = utt + '-> Context: '
    for j in range(1, topic_window+1):
      if i >= j:
        prev_utt_j = utterances[i-j]
        topic_context = topic_context + prev_utt_j + '->'
    topic_contexts.append(topic_context)
    for j in range(1, task_window+1):
      if i >= j:
        prev_utt_j = utterances[i-j]
        task_context = task_context + prev_utt_j + '->'
    task_contexts.append(task_context)
  return topic_contexts, task_contexts

topic_window = int(sys.argv[1])
task_window = int(sys.argv[2])
all_files_dict = read_files('train_names.txt')
print(all_files_dict.keys(), len(all_files_dict))

topic_model = 'roberta-topic-adjudicated-context-' + str(topic_window)
task_model = 'roberta-task-adjudicated-context' + str(task_window)

topic_model_path = '/rc_scratch/anga5835/off-topic/' + topic_model + '/checkpoint-900/'
task_model_path = '/rc_scratch/anga5835/off-topic/' + task_model + '/checkpoint-900/'

topic_pipeline = load_model(topic_model_path)
task_pipeline = load_model(task_model_path)

for filename, entries in all_files_dict.items():
  #print(entries)
  output_df = pd.DataFrame(columns=['Speaker', 'Time', 'Utterance', 'Topic', 'Task'])
  output_df['Utterance'] = entries['Utterance']
  output_df['Speaker'] = entries['Speaker']
  topic_contexts = entries['Topic_Contexts']
  task_contexts = entries['Task_Contexts']
  output_df['Time'] = entries['Time']
  topics = []
  tasks = []
  for u, s, c1, c2, t in zip(entries['Utterance'], entries['Speaker'], topic_contexts, task_contexts, entries['Time']):
    topic = classify_topic(topic_pipeline, c1)
    task = classify_topic(task_pipeline, c2)
    topics.append(topic)
    tasks.append(task)
    print(u, topic, task)
    #output_df.append([s, t, u, topic])
  output_df['Topic'] = topics
  output_df['Task'] = tasks
  output_filename = filename[:-4] + '.xlsx'
  output_df.to_excel('asr_outputs/'+output_filename)
  print(output_df.head)


 
