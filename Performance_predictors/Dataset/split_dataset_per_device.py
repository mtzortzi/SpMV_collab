import pandas as pd

# header = pd.read_csv("./data/data_sample_2.csv")
# print(header)

def split_dataset_per_device(prefix, df):
    for System in list(set(df['System'])):
        sys_df = df[df['System']==System]
        print('./data/' + prefix + '/' + prefix + '_' + System + '.csv')
        sys_df.to_csv('./data/' + prefix + '/' + prefix + '_' + System + '.csv', index=False )
        # break

all_format_df  = pd.read_csv('./data/' + 'all_format_runs_March_2023.csv')
split_dataset_per_device('all_format', all_format_df)

best_format_df = pd.read_csv('./data/' + 'best_format_runs_March_2023.csv')
split_dataset_per_device('best_format', best_format_df)
