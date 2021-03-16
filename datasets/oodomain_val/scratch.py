import pandas as pd
import json

out = ['race', 'relation_extraction', 'duorc']
data_dict = {'question': [], 'context': [], 'id': [], 'answer': []}
for path in out:
    with open(path, 'rb') as f:
        squad_dict = json.load(f)
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                if len(qa['answers']) == 0:
                    data_dict['question'].append(question)
                    data_dict['context'].append(context)
                    data_dict['id'].append(qa['id'])
                else:
                    for answer in qa['answers']:
                        data_dict['question'].append(question)
                        data_dict['context'].append(context)
                        data_dict['id'].append(qa['id'])
                        data_dict['answer'].append(answer['text'])



df = pd.DataFrame(data_dict, columns = ['question','context', 'id', 'answer'])

print (df)

df.to_csv('ground_truth.csv', index=False)
