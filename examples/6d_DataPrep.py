import pandas as pd

def replace_symbol(x):
    x=x.replace('\r','')
    x=x.replace('\t','')
    x=x.replace('\n','')
    x=x.replace('*','')
    return x


def convert_CoLA_like(in_path, out_path, cols=None):
    '''
    Objective: convert raw_train_data into CoLA_like_data
    '''

    df_org = pd.read_csv(in_path + 'train.csv',
                         lineterminator="\n")  # path = './modelData/paragh_true_train.csv'

    text, label = cols[0], cols[1]
    df = pd.DataFrame(columns=('col_0', 'label', 'col_2', 'text'))
    df['label'] = df_org[text]
    df['text'] = df_org[label]
    df['col_0'] = None
    df['col_2'] = None

    df['text'] = df['text'].apply(replace_symbol)

    df.to_csv(out_path + 'train.tsv', sep='\t', header=False, index=False)
    print(len(df))
    return

def convert_CoLA_like_from_df(df_org, cols=['content','label']):
    text, label = cols[0], cols[1]
    df = pd.DataFrame(columns=('col_0', 'label', 'col_2', 'text'))
    df['label'] = df_org[label]
    df['text'] = df_org[text]
    df['col_0'] = 'None'
    df['col_2'] = 'None'

    df['text'] = df['text'].apply(replace_symbol).dropna(axis=0)
    print(len(df_org))
    print(len(df))
    print(df.tail())
    df.to_csv('/data1/liudi/data/edm_data/dev.tsv', sep='\t', header=False, index=False)
    return df

train_1 = pd.read_pickle('/data1/liudi/data/edm_data/inbox_text_train.pkl')
train_2 = pd.read_pickle('/data1/liudi/data/edm_data/spam_text_train.pkl')
test_1 = pd.read_pickle('/data1/liudi/data/edm_data/inbox_text_test.pkl')
test_2 = pd.read_pickle('/data1/liudi/data/edm_data/spam_text_test.pkl')

train = pd.concat([train_1,train_2]).sample(frac=1).reset_index(drop=True)
test = pd.concat([test_1,test_2]).reset_index(drop=True)
test = test.drop(test.index[4162]).reset_index(drop=True)
test = test.drop(test.index[[864,865]]).reset_index(drop=True)
convert_CoLA_like_from_df(test, cols=['content','label'])

# import csv
# with open('/data1/liudi/data/edm_data/dev.tsv', "r") as f:
#     # lines = []
#     # reader = csv.reader(f, delimiter='\t', quotechar=None)
#     # i = 0
#     # for line in reader:
#     #
#     #     print(i)
#     #     lines.append(line)
#     #     i+=1
#
#     lines = list(csv.reader(f, delimiter='\t', quotechar=None))
