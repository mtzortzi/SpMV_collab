import pandas as pd

# header = pd.read_csv("./data/data_sample_2.csv")
# print(header)

def split_dataset_per_device(prefix, df):
    for System in list(set(df['System'])):
        sys_df = df[df['System']==System]
        print('./data/' + prefix + '/' + prefix + '_' + System + '.csv')
        sys_df.to_csv('./data/' + prefix + '/' + prefix + '_' + System + '.csv', index=False )

# all_format_df  = pd.read_csv('./data/' + 'all_format_runs_March_2023.csv')
# split_dataset_per_device('all_format', all_format_df)

# best_format_df = pd.read_csv('./data/' + 'best_format_runs_March_2023.csv')
# split_dataset_per_device('best_format', best_format_df)


def split_validation_dataset_per_device(prefix, df):
    for System in list(set(df['System'])):
        sys_df = df[df['System']==System]
        print('./data/validation/' + prefix + '/' + prefix + '_' + System + '.csv')
        sys_df.to_csv('./data/validation/' + prefix + '/' + prefix + '_' + System + '.csv', index=False )

# all_format_df  = pd.read_csv('./data/validation/' + 'all_format_validation_runs_March_2023.csv')
# split_validation_dataset_per_device('all_format', all_format_df)

# best_format_df = pd.read_csv('./data/validation/' + 'best_format_validation_runs_March_2023.csv')
# split_validation_dataset_per_device('best_format', best_format_df)


def split_dataset_per_device_per_implementation(prefix, df):
    for System in list(set(df['System'])):
        sys_df = df[df['System']==System]
        for implementation in list(set(sys_df['implementation'])):
            impl_sys_df = sys_df[sys_df['implementation']==implementation]
            print('./data/' + prefix + '/' + prefix + '_' + System + '_' + implementation + '.csv')
            impl_sys_df.to_csv('./data/' + prefix + '/' + prefix + '_' + System + '_' + implementation + '.csv', index=False )

all_format_df  = pd.read_csv('./data/' + 'all_format_runs_March_2023.csv')
split_dataset_per_device_per_implementation('all_format', all_format_df)

# no need to do this for the 'best_format' dataset...

def split_validation_dataset_per_device_per_implementation(prefix, df):
    for System in list(set(df['System'])):
        sys_df = df[df['System']==System]
        for implementation in list(set(sys_df['implementation'])):
            impl_sys_df = sys_df[sys_df['implementation']==implementation]
            print('./data/validation/' + prefix + '/' + prefix + '_' + System + '_' + implementation + '.csv')
            impl_sys_df.to_csv('./data/validation/' + prefix + '/' + prefix + '_' + System + '_' + implementation + '.csv', index=False )

all_format_df  = pd.read_csv('./data/validation/' + 'all_format_validation_runs_March_2023.csv')
split_validation_dataset_per_device_per_implementation('all_format', all_format_df)

# no need to do this for the 'best_format' dataset...