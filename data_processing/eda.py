# data_processing/eda.py
import pandas as pd
import logging,os,sys
from config.config_handler import load_config
import matplotlib.pyplot as plt
import seaborn as sns

cfg = load_config()

data_dir = cfg.data.data_dir
eda_dir = cfg.data.eda_dir
eda_output_path = os.path.join(data_dir,eda_dir)

if not os.path.exists(eda_output_path):
    os.makedirs(eda_output_path)
   
def eda_cat_variable(df,fileName):
    
    cat_columns = cfg.data.cat_columns
    # Create a figure outside the loop
    plt.figure(figsize=(25, 35))
    plt.suptitle("Analysis Of chrun rate per categorical variable", fontweight="bold", fontsize=20)

    # Determine the number of rows and columns based on the number of categorical columns
    num_rows = len(cat_columns) // 2 + len(cat_columns) % 2
    num_cols = 2

    # Set the font size for all text elements in the plot
    sns.set(font_scale=1.8)

    for i in range(len(cat_columns)):
        plt.subplot(num_rows, num_cols, i + 1)
        #plt.gca().set_title(f'{cat_columns[i]}')
        sns.countplot(x=cat_columns[i], hue='Exited', data=df)

        #plt.xlabel(cat_columns[i])  # Use the default fontsize (scaled by sns.set())
        plt.ylabel('Count')  # Use the default fontsize (scaled by sns.set())

    plt.tight_layout()
    
    plt.savefig(os.path.join(eda_output_path, fileName))
    #plt.show()
    
    
    
def eda_num_variable(df,fileName):
    # Create a figure outside the loop
    
    numerical_columns = cfg.data.numerical_columns
    
    plt.figure(figsize=(25, 35))
    plt.suptitle("Analysis Of chrun rate per Numerical variable", fontweight="bold", fontsize=20)

    # Determine the number of rows and columns based on the number of numerical columns
    num_rows = len(numerical_columns) // 2 + len(numerical_columns) % 2
    num_cols = 2

    # Set the font size for all text elements in the plot
    sns.set(font_scale=1.8)

    for i in range(len(numerical_columns)):
        plt.subplot(num_rows, num_cols, i + 1)

        # Plot the histogram for 'Exited' == 0
        sns.histplot(x=numerical_columns[i], data=df[df['Exited'] == 0], color='green', label='Exited=0', alpha=0.7)

        # Plot the histogram for 'Exited' == 1
        sns.histplot(x=numerical_columns[i], data=df[df['Exited'] == 1], color='orange', label='Exited=1', alpha=0.7)

        plt.xlabel(numerical_columns[i])  # Set the x-axis label
        plt.ylabel('Count')  # Use the default fontsize (scaled by sns.set())

        plt.legend()  # Add a legend to distinguish the two categories

    plt.tight_layout()
    plt.savefig(os.path.join(eda_output_path, fileName))
    #plt.show()


def plot_corr_matrix(df,fileName):
    correlation_matrix = df.corr(numeric_only=True)

    # Create the heatmap
    plt.figure(figsize=(10, 8))  # Adjust the size of the figure as per your preference
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", 
                linewidths=0.5 , annot_kws={"size": 10})
    plt.title("Correlation Matrix Heatmap", fontsize=12)

    plt.yticks(fontsize=10)  # Increase the font size of y-axis labels
    plt.xticks(fontsize=10)  # Increase the font size of x-axis labels

    plt.savefig(os.path.join(eda_output_path, fileName))
    #plt.show()
    
 
def perform_eda(df,cat_fileName,num_fileName,corr_fileName):
    logging.info("Perform eda")
    eda_cat_variable(df,cat_fileName)
    eda_num_variable(df,num_fileName)
    plot_corr_matrix(df,corr_fileName)
