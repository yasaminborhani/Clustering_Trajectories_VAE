import matplotlib.pyplot as plt

def create_bar_chart(data_dict, title):
    categories = list(data_dict.keys())
    values = list(data_dict.values())

    # Choose a colormap from Matplotlib (e.g., 'viridis', 'plasma', 'tab20', 'Set2', etc.)
    # You can find a list of available colormaps here: https://matplotlib.org/stable/tutorials/colors/colormaps.html
    colormap = 'Set2'

    plt.bar(categories, values, color=plt.cm.get_cmap(colormap)(range(len(categories))))
    plt.xlabel('Categories')
    plt.ylabel('Value')
    plt.title(title)
    plt.xticks(rotation=45)  # Rotate the category labels for better visibility
    plt.grid(axis='y')  # Add grid lines on the y-axis
    plt.show()

def create_grouped_bar_chart(data_dict1, data_dict2, title):
    categories = list(data_dict1.keys())
    values1 = list(data_dict1.values())
    values2 = list(data_dict2.values())

    # Set the width of the bars
    bar_width = 0.4

    # Calculate the positions of the bars on the x-axis
    bar_positions1 = np.arange(len(categories))
    bar_positions2 = bar_positions1 + bar_width

    plt.bar(bar_positions1, values1, width=bar_width, label='Train')
    plt.bar(bar_positions2, values2, width=bar_width, label='Validation')

    plt.xlabel('Categories')
    plt.ylabel('Values')
    plt.title(title)
    plt.xticks(bar_positions1 + bar_width / 2, categories, rotation=45)
    plt.legend()
    plt.grid(axis='y')
    plt.show()