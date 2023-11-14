import pandas as pd

# header = pd.read_csv("./data/data_sample_2.csv")
# print(header)

def split_dataset_per_device(path_prefix, prefix, df):
    for System in list(set(df['System'])):
        sys_df = df[df['System']==System]
        print(path_prefix + prefix + '/' + prefix + '_' + System + '.csv')
        sys_df.to_csv(path_prefix + prefix + '/' + prefix + '_' + System + '.csv', index=False )

# all_format_df  = pd.read_csv('./data/' + 'all_format_runs_March_2023.csv')
# split_dataset_per_device('all_format', all_format_df)

# best_format_df = pd.read_csv('./data/' + 'best_format_runs_March_2023.csv')
# split_dataset_per_device('best_format', best_format_df)

# validation dataset
# all_format_df  = pd.read_csv('./data/validation/' + 'all_format_validation_runs_March_2023.csv')
# split_dataset_per_device('./data/validation/', 'all_format', all_format_df)

# best_format_df = pd.read_csv('./data/validation/' + 'best_format_validation_runs_March_2023.csv')
# split_dataset_per_device('./data/validation/', 'best_format', best_format_df)


def split_dataset_per_device_per_implementation(path_prefix, prefix, df):
    for System in list(set(df['System'])):
        sys_df = df[df['System']==System]
        for implementation in list(set(sys_df['implementation'])):
            impl_sys_df = sys_df[sys_df['implementation']==implementation]
            print(path_prefix + prefix + '/' + prefix + '_' + System + '_' + implementation + '.csv')
            impl_sys_df.to_csv(path_prefix + prefix + '/' + prefix + '_' + System + '_' + implementation + '.csv', index=False )

# all_format_df  = pd.read_csv('./data/' + 'all_format_runs_March_2023.csv')
# split_dataset_per_device_per_implementation('./data/', 'all_format', all_format_df)

# no need to do this for the 'best_format' dataset...

# validation dataset
# all_format_df  = pd.read_csv('./data/validation/' + 'all_format_validation_runs_March_2023.csv')
# split_dataset_per_device_per_implementation('./data/validation/', 'all_format', all_format_df)

# no need to do this for the 'best_format' dataset...


def split_CPU_dataset_on_cache_threshold(path_prefix, prefix, df, threshold):
    System = list(set(df['System']))[0]

    for implementation in list(set(df['implementation'])):
        impl_df = df[df['implementation']==implementation]
        
        mem_ranges = list(set(impl_df['mem_range']))
        smaller_than = []
        larger_than  = []
        for mem_range in mem_ranges:
            mr_low = int(mem_range.split('-')[0].split('[')[1])
            mr_high = int(mem_range.split('-')[1].split(']')[0])
            if(mr_low>=threshold):
                larger_than.append(mem_range)
            else:
                smaller_than.append(mem_range)

        smaller_than_df = impl_df[impl_df['mem_range'].isin(smaller_than)]
        larger_than_df  = impl_df[impl_df['mem_range'].isin(larger_than)]

        print(path_prefix + prefix + '/' + prefix + '_' + System + '_' + implementation + '_' + 'smaller_than_cache' + '.csv')
        smaller_than_df.to_csv(path_prefix + prefix + '/' + prefix + '_' + System + '_' + implementation + '_' + 'smaller_than_cache' + '.csv', index=False )
        print(path_prefix + prefix + '/' + prefix + '_' + System + '_' + implementation + '_' + 'larger_than_cache' + '.csv')
        larger_than_df.to_csv(path_prefix + prefix + '/' + prefix + '_' + System + '_' + implementation + '_' + 'larger_than_cache' + '.csv',  index=False )

all_format_AMD_EPYC_24_df  = pd.read_csv('./data/' + 'all_format/all_format_AMD-EPYC-24.csv')
split_CPU_dataset_on_cache_threshold('./data/', 'all_format', all_format_AMD_EPYC_24_df, 128)

# validation dataset
all_format_AMD_EPYC_24_df  = pd.read_csv('./data/validation/' + 'all_format/all_format_AMD-EPYC-24.csv')
split_CPU_dataset_on_cache_threshold('./data/validation/', 'all_format', all_format_AMD_EPYC_24_df, 128)
