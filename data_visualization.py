import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl

def generate_basic_plot(arr, xlabel, ylabel, figName, close=True, title=''):
    x = np.linspace(1, len(arr), num=len(arr))
    plt.plot(x, arr)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    #plt.show()
    plt.savefig(f"{figName}.png")
    if close:
        plt.close()
        return 
    else:
        return plt

def plot_double(arr1, arr2, xlabel, ylabel, label1, label2, figName, title=''):
    '''
    Plot two arrays on the same figure (plot) 
    the length of arr1 will go on the x-axis.
    Args:
        - arr1: The first array to plot.
        - arr2: The second array to plot.
        - xlabel: Title (string) for xaxis.
        - ylabel: Title (string) for yaxis.
        - label1: Label (string) for the legend of arr1.
        - label2: Label (string) for the legend of arr2.
    '''
    x = np.linspace(1, len(arr1), num=len(arr1))
    plt.plot(x, arr1, label=label1)
    plt.plot(x, arr2, label=label2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(f"{figName}.png")
    plt.close()
    return 

def moving_average(arr, window_size):
    '''
    Calculate moving average with given window size.
    
    Args:
        data: List of floats.
        window_size: Integer size of moving window.
    Returns:
        List of rolling averages with given window size.
    Source: www.geeksforgeeks.org
    '''
    i = 0
    # Initialize an empty list to store moving averages
    moving_averages = []
 
    # Loop through the array to consider
    # every window of size 3
    while i < len(arr) - window_size + 1:
   
        # Store elements from i to i+window_size
        # in list to get the current window
        window = arr[i : i + window_size]
 
        # Calculate the average of current window
        window_average = round(sum(window) / window_size, 2)
     
        # Store the average of current
        # window in moving average list
        moving_averages.append(window_average)
         
        # Shift window to right by one position
        i += 1
    return moving_averages

def plot_moving_avg(arr,  xlabel, ylabel, figName, window_size=10, title=''):


    mov_avg = moving_average(arr, window_size)
    x = np.linspace(1, len(mov_avg), num=len(mov_avg))
    plt.plot(x,mov_avg)
    plt.xlabel(f"{xlabel} x{window_size}")
    plt.ylabel(ylabel)
    plt.title(title)
    #plt.show()
    plt.savefig(f"{figName}.png")
    plt.close()
    return

def plot_double_moving_avg(arr1, arr2,  xlabel, ylabel, label1, label2, figName, window_size=10, title=''):
    '''
    The same as plot_moving_avg but for two plots(arrs)
    '''

    mov_avg1 = moving_average(arr1, window_size)
    mov_avg2 = moving_average(arr2, window_size)
    x = np.linspace(1, len(mov_avg1), num=len(mov_avg1))
    plt.plot(x,mov_avg1, label=label1)
    plt.plot(x,mov_avg2, label=label2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    #plt.show()
    plt.savefig(f"{figName}.png")
    plt.close()
    return
