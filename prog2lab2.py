import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

def load_data(file_path):
    return pd.read_excel(file_path,
                         sheet_name=1, 
                         usecols='A:I')

def mean_variance_price(data_frame, col):
    data = data_frame[col].to_numpy()
    return np.mean(data), np.var(data)

def find_mean(data_frame, col):
    data = data_frame[col].to_numpy()
    return sum(data) / data.size

def find_variance(data_frame, col):
    data = data_frame[col].to_numpy()
    mean_data = sum(data) / data.size
    deviation_sum = 0
    for i in data:
        deviation_sum += (i - mean_data) ** 2
    return deviation_sum / data.size

def compare_results(data_frame, col, runs):
    t_np = 0
    for _ in range(runs):
        start = time.time()
        mean_variance_price(data_frame, col)
        t_np += time.time() - start

    t_manual = 0
    for _ in range(runs):
        start = time.time()
        find_mean(data_frame, col)
        find_variance(data_frame, col)
        t_manual += time.time() - start

    return t_np / runs, t_manual / runs

def wednesday_price_data(data_frame):
    mean_wed = data_frame[data_frame['Day'] == 'Wed']['Price'].mean()
    mean_full = data_frame['Price'].mean()
    return mean_wed, mean_full

def april_price_data(data_frame):
    mean_apr = data_frame[data_frame['Month'] == 'Apr']['Price'].mean()
    mean_full = data_frame['Price'].mean()
    return mean_apr, mean_full

def charges_loss_probability(data_frame):
    data = data_frame['Chg%'].to_numpy()
    return (data < 0).sum() / data.size

def profit_wednesday_probability(data_frame):
    wed_data = data_frame[data_frame['Day'] == 'Wed']['Chg%'].to_numpy()
    return (wed_data > 0).sum() / wed_data.size

def conditional_probability_wed(data_frame):
    return profit_wednesday_probability(data_frame)

def scatter_plot_chg_day(data_frame):
    day_map = {'Mon':1, 'Tue':2, 'Wed':3, 'Thu':4, 'Fri':5}
    x = data_frame['Day'].map(day_map)
    y = data_frame['Chg%']

    plt.scatter(x, y)
    plt.xticks([1,2,3,4,5], ['Mon','Tue','Wed','Thu','Fri'])
    plt.xlabel("Day of the Week")
    plt.ylabel("Chg %")
    plt.title("Scatter Plot of Chg% vs Day")
    plt.show()

def main():
    file_path = 'Lab Session Data (1).xlsx'
    df = load_data(file_path)

    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df['Chg%'] = pd.to_numeric(df['Chg%'].astype(str).str.replace('%',''), errors='coerce')

    mean_np, var_np = mean_variance_price(df, 'Price')
    mean_manual = find_mean(df, 'Price')
    var_manual = find_variance(df, 'Price')

    print("Difference in Mean:", abs(mean_np - mean_manual))
    print("Difference in Variance:", abs(var_np - var_manual))

    time_np, time_manual = compare_results(df, 'Price', 10)
    print("Avg Time (NumPy):", time_np)
    print("Avg Time (Manual):", time_manual)

    mean_wed, mean_full = wednesday_price_data(df)
    print("Wednesday Mean:", mean_wed)
    print("Population Mean:", mean_full)

    mean_apr, mean_full = april_price_data(df)
    print("April Mean:", mean_apr)

    print("Probability of Loss:", charges_loss_probability(df))
    print("Probability of Profit on Wednesday:", profit_wednesday_probability(df))
    print("Conditional Probability (Profit | Wednesday):", conditional_probability_wed(df))

    scatter_plot_chg_day(df)

main()
