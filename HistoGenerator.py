import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


def generate_histograms(file_path, output_dir):
    df = pd.read_csv(file_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    df['File'] = df['File'].astype(str)
    df['Thresholding Method'] = df['Thresholding Method'].astype(str)
    
    df['Gini Coefficient'] = pd.to_numeric(df['Gini Coefficient'], errors='coerce')
    df['M20 Index'] = pd.to_numeric(df['M20 Index'], errors='coerce')
    df['Filamentarity'] = pd.to_numeric(df['Filamentarity'], errors='coerce')
    
    df['Gini Std Dev'] = pd.to_numeric(df['Gini Std Dev'], errors='coerce')
    df['M20 Std Dev'] = pd.to_numeric(df['M20 Std Dev'], errors='coerce')
    df['Filamentarity Std Dev'] = pd.to_numeric(df['Filamentarity Std Dev'], errors='coerce')
    
    galaxy_names = df['File'].unique()
    
    for galaxy in galaxy_names:
        galaxy_df = df[df['File'] == galaxy]
        
        methods = galaxy_df['Thresholding Method'].values
        gini_values = galaxy_df['Gini Coefficient'].values
        m20_values = galaxy_df['M20 Index'].values
        filamentarity_values = galaxy_df['Filamentarity'].values
        
        gini_std = galaxy_df['Gini Std Dev'].values
        m20_std = galaxy_df['M20 Std Dev'].values
        filamentarity_std = galaxy_df['Filamentarity Std Dev'].values
        
        gini_std = np.where(np.isnan(gini_std), 0, gini_std)
        m20_std = np.where(np.isnan(m20_std), 0, m20_std)
        filamentarity_std = np.where(np.isnan(filamentarity_std), 0, filamentarity_std)
        
        fig, axs = plt.subplots(3, 1, figsize=(10, 18))
        
        axs[0].bar(methods, gini_values, yerr=gini_std, color='skyblue', capsize=5)
        axs[0].set_xlabel('Thresholding Method')
        axs[0].set_ylabel('Gini Coefficient')
        axs[0].set_title(f'Gini Coefficient for {galaxy}')
        axs[0].tick_params(axis='x', rotation=45)
        
        axs[1].bar(methods, m20_values, yerr=m20_std, color='lightgreen', capsize=5)
        axs[1].set_xlabel('Thresholding Method')
        axs[1].set_ylabel('M20 Index')
        axs[1].set_title(f'M20 Index for {galaxy}')
        axs[1].tick_params(axis='x', rotation=45)
        
        axs[2].bar(methods, filamentarity_values, yerr=filamentarity_std, color='lightcoral', capsize=5)
        axs[2].set_xlabel('Thresholding Method')
        axs[2].set_ylabel('Filamentarity')
        axs[2].set_title(f'Filamentarity for {galaxy}')
        axs[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()

        output_file = os.path.join(output_dir, f'{galaxy}_combined_histograms.png')
        plt.savefig(output_file)
        plt.close()

    print(f'Combined histograms saved in {output_dir}')

csv_file_path = 'Outputs/Values/JWSTf770/gini_m20_filamentarity_coefficients.csv'
output_directory = 'Outputs/Histograms/JWSTf770'
generate_histograms(csv_file_path, output_directory)
