train = pd.read_csv("/kaggle/input/competitions/nlp-getting-started/train.csv")
test = pd.read_csv("/kaggle/input/competitions/nlp-getting-started/test.csv")

train.head(3)

print(f'train shape:{train.shape}')
print(f'test shape:{test.shape}')
train['target'].value_counts()
print(train.isnull().sum())
print(test.isnull().sum())

col = ['keyword','location']

for i in [train,test]:
    for j in col:
        i[j] = i[j].fillna("")
#缺失值处理

print(f'train unique keyword:{train['keyword'].nunique()}')
print(f'test unique keyword:{test['keyword'].nunique()}')
print(f'train unique location:{train['location'].nunique()}')
print(f'test unique location:{test['location'].nunique()}')
#统计分类

import matplotlib.pyplot as plt
import seaborn as sns

train['target_mean'] = train.groupby('keyword')['target'].transform('mean')

fig = plt.figure(figsize=(10,80),dpi=100)

sort_train = train.sort_values(by='target_mean',ascending=False)

sns.countplot(y=sort_train['keyword'],hue=sort_train['target'])

plt.legend()
#研究每个keyword的灾难情况



train.drop(columns='target_mean',inplace=True)
train['word_count'] = train['text'].apply(lambda x:len(str(x).split()))
test['word_count'] = test['text'].apply(lambda x:len(str(x).split()))

train['unique_word_count'] = train['text'].apply(lambda x:len(set(str(x).split())))
test['unique_word_count'] = test['text'].apply(lambda x:len(set(str(x).split())))

from wordcloud import STOPWORDS
train['stop_word_count'] = train['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))
test['stop_word_count'] = test['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))

train['url_count'] = train['text'].apply(lambda x:len([w for w in str(x).lower().split() if 'http' in w or 'https' in w] ))
test['url_count'] = test['text'].apply(lambda x:len([w for w in str(x).lower().split() if 'http' in w or 'https' in w] ))

train['char_count'] = train['text'].apply(lambda x:len(str(x)))
test['char_count'] = test['text'].apply(lambda x:len(str(x)))

train['hashtag_count'] = train['text'].apply(lambda x:len([w for w in str(x) if w=='#']))
test['hashtag_count'] = test['text'].apply(lambda x:len([w for w in str(x) if w=='#']))

train['mention_count'] = train['text'].apply(lambda x:len([w for w in str(x) if w=='@']))
train['mention_count'] = train['text'].apply(lambda x:len([w for w in str(x) if w=='@']))
train.head(5)
#研究是否有好的特征



 
meta_features = ['word_count', 'unique_word_count', 'stop_word_count', 'url_count', 'char_count', 'hashtag_count','mention_count']
disaster_tweets = train['target'] == 1

fig, axes = plt.subplots(ncols=2, nrows=len(meta_features), figsize=(20,40), dpi=100)

for i, feature in enumerate(meta_features):
    sns.distplot(train.loc[~disaster_tweets][feature], label='Not Disaster', ax=axes[i][0], color='green')
    sns.distplot(train.loc[disaster_tweets][feature], label='Disaster', ax=axes[i][0], color='red')
    sns.distplot(train[feature], label='Training', ax=axes[i][1])
    sns.distplot(test[feature], label='Test', ax=axes[i][1])
    
    for j in range(2):
        axes[i][j].legend()
    
    axes[i][0].set_title(f'{feature} Target Distribution in Training Set')
    axes[i][1].set_title(f'{feature} Training & Test Set Distribution')

plt.show()
#绘图，研究分布




fig = plt.figure(figsize=(8,8))
sns.countplot( x=train['target'],hue=train['target'] )
#检查数据0，1分布


mislabeled = train.groupby(['text']).nunique().sort_values(by='target',ascending=False)
mislabeled = mislabeled[mislabeled['target']>1]['target']
mislabeled.index.tolist()
train = train[~train['text'].isin(mislabeled.index.tolist())]
#去除垃圾数据，影响分类

import random
import torch
from torch.utils.data import TensorDataset,DataLoader,random_split
from transformers import BertTokenizer
from transformers import BertForSequenceClassification

from transformers import get_linear_schedule_with_warmup

random.seed(17)
np.random.seed(17)
torch.manual_seed(17)
torch.cuda.manual_seed_all(17)

device = torch.device('cuda')



from torch.optim import AdamW
import os
os.environ['HF_ENDPOINT'] = 'https:hf-mirror.com'

train['final_text'] = train['keyword']+' '+train['text']
test['final_text'] = test['keyword']+' '+test['text']

text_value = train['final_text'].values
labels = train['target'].values

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)
#转换数字编码


train['lenth'] = train['final_text'].apply(len)
test['lenth'] = test['final_text'].apply(len)

fig ,ax = plt.subplots(1,2)
fig.set_size_inches(15,6)

sns.distplot(train['length'],ax=ax[0])
sns.distplot(test['length'],ax=ax[1])
plt.show()
#研究特征工程后分布

def encode_(text_list):
    all_input_ids=[]
    for text in text_list:
        input_ids = tokenizer.encode(text,add_special_tokens=True,max_length=100,pad_to_max_length=True,return_tensors='pt')
        all_input_ids.append(input_ids)
    all_input_ids = torch.cat(all_input_ids,dim=0)
    return all_input_ids
#编码函数

epochs=4
batch_size = 32

all_input_ids = encode_fn(text_values)
labels = torch.tensor(labels)

dataset = TensorDataset(all_input_ids, labels)
train_size = int(0.80*len(dataset))
val_size = len(dataset) = train_size
train_dataset,val_dataset = random_split(dataset,[train_size,val_size])
#数据准备


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
model = BertForSequenceClassification.from_pretrained('bert_base_uncased',num_lables=2,output_attentions=False,output_hidden_states=False)
model.cuda()
#搬运数据

optimizer = AdamW(model.parameters(),lr=2e-5)
total_steps = len(train_dataloader)*epochs
scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,num_training_steps = total_steps)
#初始化模型

from sklearn.metrics import f1_score,accuracy_score

def flat_aaccuracy(preds,labels):
    pred_flat = np.argmax(preds,axis=1).flatten()
    labels_flat = labels.flatten()

    return accuracy_score(labels_flat, pred_flat)


for epoch in range(epochs):
    model.train()
    total_loss, total_val_loss = 0, 0
    total_eval_accuracy = 0


    for step,batch in enumerate(train_dataloader):
        model.zero_grad()
        loss ,logits = model(batch[0].to(device),token_type_ids=None,attention_mask=(batch[0]>0).to(device),labels=batch[1].to(device))
        total_val_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        scheduler.step()


    model.eval()
    for i, batch in enumerate(val_dataloader):
        with torvh.no_grad():
            loss, logits = model(batch[0].to(device), token_type_ids=None, attention_mask=(batch[0]>0).to(device), labels=batch[1].to(device))
                
            total_val_loss += loss.item()

            logits = logits.datach().cpu().numpy()
            label_ids = batch[1].to('cpu').numpy()
            total_eval_accuracy += flat_accuracy(logits, label_ids)

    avg_train_loss = total_loss / len(train_dataloader)
    avg_val_loss = total_val_loss / len(val_dataloader)
    avg_val_accuracy = total_eval_accuracy / len(val_dataloader)

    print(f'Train loss     : {avg_train_loss}')
    print(f'Validation loss: {avg_val_loss}')
    print(f'Accuracy: {avg_val_accuracy:.2f}')
    print('\n')
        
text_values = test['final_text'].values
all_input_ids = encode_fn(text_values)
pred_data = TensorDataset(all_input_ids)
pred_dataloader = DataLoader(pred_data, batch_size=batch_size, shuffle=False)
model.eval()
preds = []


for i, (batch,) in enumerate(pred_dataloader):
    with torch.no_grad():
        outputs = model(batch.to(device), token_type_ids=None, attention_mask=(batch>0).to(device))
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        preds.append(logits)

final_preds = np.concatenate(preds,axis=0)
final_preds = np.argmax(final_preds,axis=1)

submission = pd.DataFrame()
submission['id'] = df_test['id']
submission['target'] = final_preds
submission.to_csv('submission.csv', index=False)
