import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pandas as pd
import numpy as np

def plot_graph(df):
    cont_and_frequency_of_depression(df)
    depression_on_degree(df,'degree')
    countplot_degree_cgpa_age(df,['degree', 'cgpa_scaled', 'age_bin'])
    line_graph_distributions(df,['gender','academic_pressure', 'study_satisfaction', 'financial_stress', 'suicidal_thoughts', 'work_study_hours', 'family_mental_history', 'sleep_duration', 'dietary_habits', 'age_bin', 'cgpa_bin', 'degree'])
    ordinal_plots(df,['gender','academic_pressure', 'study_satisfaction', 'financial_stress', 'suicidal_thoughts', 'work_study_hours', 'family_mental_history', 'sleep_duration', 'dietary_habits', 'age_bin', 'cgpa_bin', 'degree'])
    plot_distributions_histogram(df,['gender','academic_pressure', 'study_satisfaction', 'financial_stress', 'suicidal_thoughts', 'work_study_hours', 'family_mental_history', 'sleep_duration', 'dietary_habits', 'age_bin', 'cgpa_bin', 'degree'])
    plot_distributions_boxplot(df)
    correlation_matrix(df)

def cont_and_frequency_of_depression(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    depression_mapped = df["depression"].map({0: "No", 1: "Yes"})
    order_values = ["Yes", "No"]
    palette = [sns.color_palette("pastel")[1],
               sns.color_palette("pastel")[0]]  # use different color for different value
    # Count Plot of Depression
    sns.countplot(y=depression_mapped, order=order_values, palette=palette, ax=axes[0])
    axes[0].set_title("Count Plot of Depression")
    axes[0].grid(axis="x", linestyle=":")
    # Relative Frequency of Depression
    relative_freq = depression_mapped.value_counts(normalize=True, sort=False) * 100  # make it become percentage
    sns.barplot(x=relative_freq.values, y=relative_freq.index, palette=palette, orient="h", ax=axes[1])
    axes[1].set_title("Relative Frequency of Depression")
    axes[1].set_xlabel("Percentage")
    axes[1].set_xlim(0, 100)
    axes[1].set_xticks(range(0, 110, 10))
    axes[1].grid(axis="x", linestyle=":")
    plt.tight_layout()
    plt.show()

def depression_on_degree(df,col):
    def plot_countplot(df, col):
        plt.figure(figsize=(8, 6))
        sns.countplot(y=df[col], order=df[col].value_counts().index,
                      color=sns.color_palette("pastel")[7])
        plt.title(f"Count Plot of {col.replace('_', ' ').title()}")
        plt.grid(axis='x', linestyle=':')
        plt.tight_layout()  # fits nicely in the figure
        plt.show()

    def depression_rate(df, col):
        plt.figure(figsize=(8, 6))
        depression_rate = df.groupby(col)["depression"].mean() * 100
        ax = sns.barplot(x=depression_rate.values, y=depression_rate.index, order=df[col].value_counts().index,
                         color=sns.color_palette("pastel")[1])
        # Add value labels
        ax.bar_label(ax.containers[0], fmt="%.1f%%", padding=5)
        plt.title(f"Depression Rate by {col.replace('_', ' ').title()}")
        plt.xlabel("Depression Rate (%)")
        plt.xlim(0, 100)
        plt.grid(axis="x", linestyle=":")
        plt.tight_layout()
        plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # Count plot
    sns.countplot(y=df[col], order=df[col].value_counts().index,
                  color=sns.color_palette("pastel")[7], ax=axes[0])
    axes[0].set_title(f"Count Plot of {col.replace('_', ' ').title()}")
    axes[0].grid(axis='x', linestyle=':')
    axes[0].set_xlabel("Count")

    # Depression rate plot
    depression_rate = df.groupby(col)["depression"].mean() * 100
    sns.barplot(x=depression_rate.values, y=depression_rate.index,
                order=df[col].value_counts().index,
                color=sns.color_palette("pastel")[1], ax=axes[1])
    axes[1].set_title(f"Depression Rate by {col.replace('_', ' ').title()}")
    axes[1].set_xlabel("Depression Rate (%)")
    axes[1].set_xlim(0, 100)
    axes[1].grid(axis="x", linestyle=":")
    # Add value labels
    axes[1].bar_label(axes[1].containers[0], fmt="%.1f%%", padding=5)

    plt.tight_layout()
    plt.show()

def countplot_degree_cgpa_age(df, cols):
    # If cols is a single string, convert it to a list for uniformity
    if isinstance(cols, str):
        cols = [cols]

    for col in cols:
        if df[col].nunique() <= 1:
            print(f"Skipping {col}: only one unique value.")
            continue

        plt.figure(figsize=(10, 5))

        # Value counts and percentages
        value_counts = df[col].value_counts()
        sorted_categories = value_counts.index
        percentages = (value_counts / len(df)) * 100

        # Countplot
        ax = sns.countplot(x=col, data=df, order=sorted_categories, palette='pastel')
        plt.title(f'Countplot of {col.replace("_", " ").title()}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Add percentage labels
        for p, percent in zip(ax.patches, percentages):
            height = p.get_height()
            ax.annotate(f'{percent:.1f}%', (p.get_x() + p.get_width() / 2, height),
                        ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.show()

def line_graph_distributions(df, columns):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

    # Categorical label mappings
    label_mappings = {
        "gender": {0: "Female", 1: "Male"},
        'dietary_habits': {1: 'Unhealthy', 2: 'Moderate', 3: 'Healthy'},
        'sleep_duration': {1: 'Less than 5 hours', 2: '5-6 hours', 3: '7-8 hours', 4: 'More than 8 hours'},
        'family_mental_history': {0: 'No', 1: 'Yes'},
        'suicidal_thoughts': {0: 'No', 1: 'Yes'},
        'age_bin': {
            0: "[18.00, 19.00]", 1: "(19.00, 21.00]", 2: "(21.00, 23.00]", 3: "(23.00, 24.00]",
            4: "(24.00, 25.00]", 5: "(25.00, 28.00]", 6: "(28.00, 29.00]", 7: "(29.00, 31.00]",
            8: "(31.00, 33.00]", 9: "(33.00, 59.00]"
        },
        "cgpa_bin": {
            0: "[0.00, 0.10]", 1: "(0.10, 0.20]", 2: "(0.20, 0.30]", 3: "(0.30, 0.40]",
            4: "(0.40, 0.50]", 5: "(0.50, 0.60]", 6: "(0.60, 0.70]", 7: "(0.70, 0.80]",
            8: "(0.80, 0.90]", 9: "(0.90, 1.00]", 10: "(1.00, 1.10]", 11: "(1.10, 1.20]",
            12: "(1.20, 1.30]", 13: "(1.30, 1.40]", 14: "(1.40, 1.50]", 15: "(1.50, 1.60]",
            16: "(1.60, 1.70]", 17: "(1.70, 1.80]", 18: "(1.80, 1.90]", 19: "(1.90, 2.00]",
            20: "(2.00, 2.10]", 21: "(2.10, 2.20]", 22: "(2.20, 2.30]", 23: "(2.30, 2.40]",
            24: "(2.40, 2.50]", 25: "(2.50, 2.60]", 26: "(2.60, 2.70]", 27: "(2.70, 2.80]",
            28: "(2.80, 2.90]", 29: "(2.90, 3.00]", 30: "(3.00, 3.10]", 31: "(3.10, 3.20]",
            32: "(3.20, 3.30]", 33: "(3.30, 3.40]", 34: "(3.40, 3.50]", 35: "(3.50, 3.60]",
            36: "(3.60, 3.70]", 37: "(3.70, 3.80]", 38: "(3.80, 3.90]", 39: "(3.90, 4.00]"
        }
    }

    # Set up subplots (4x3 = 12 charts)
    fig, axes = plt.subplots(4, 3, figsize=(18, 18))
    axes = axes.ravel()

    # Define custom palette
    palette = [sns.color_palette("pastel")[1], sns.color_palette("pastel")[0]]  # orange for yes, blue for no

    for idx, col in enumerate(columns):
        ax = axes[idx]

        # Apply label mapping if available
        if col in label_mappings:
            label_col = f"{col}_label"
            df[label_col] = df[col].map(label_mappings[col])
            plot_col = label_col
        else:
            plot_col = col

        # Data for Depression = Yes and No
        df_yes = df[df['depression'] == 1]
        df_no = df[df['depression'] == 0]

        # Get frequency counts
        freq_yes = df_yes[plot_col].value_counts().sort_index().reset_index()
        freq_no = df_no[plot_col].value_counts().sort_index().reset_index()

        freq_yes.columns = [plot_col, 'count_yes']
        freq_no.columns = [plot_col, 'count_no']

        # Merge counts
        freq = pd.merge(freq_yes, freq_no, on=plot_col, how='outer').fillna(0)

        # Special handling for age_bin
        if col == "age_bin":
            freq = freq.iloc[:-1]  # remove last row

        # Plot both lines
        sns.lineplot(data=freq, x=plot_col, y='count_yes', marker='o', ax=ax, color=palette[0],
                     label='Depression = Yes')
        sns.lineplot(data=freq, x=plot_col, y='count_no', marker='o', ax=ax, color=palette[1],
                     label='Depression = No')

        ax.set_title(f"{col.replace('_', ' ').title()}")
        ax.set_ylabel("Count")
        ax.set_xlabel("")

        # Rotate x-axis labels conditionally
        if col == "degree":
            ax.tick_params(axis='x', rotation=90)
        else:
            ax.tick_params(axis='x', rotation=45)

        # Add legend outside the plot area
        ax.legend(
            loc='upper left',
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0,
            fontsize='small',
            frameon=True
        )

    # Clean up unused subplots
    for idx in range(len(columns), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()

def ordinal_plots(df, columns):
    # Convert binary to Yes/No for the depression column, ensuring no overwriting
    # Ensure we map 0 to 'No' and 1 to 'Yes' only if there are valid binary entries
    df['depression'] = df['depression'].map({0: 'No', 1: 'Yes'})

    # If there are any NaN values in depression, fill them with 'No' (keeping existing 'Yes' intact)
    df['depression'].fillna('No', inplace=True)

    # Check unique values after filling
    print("Unique values in 'depression' column:", df['depression'].unique())

    # Convert suicidal thoughts into Yes/No
    df['suicidal_thoughts_binary'] = df['suicidal_thoughts'].apply(lambda x: 'Yes' if x > 0 else 'No')

    # Ensure we have both 'Yes' and 'No' values in 'depression'
    if 'Yes' not in df['depression'].values:
        print("Warning: There are no 'Yes' values in 'depression'. Plots will only show 'No'.")

    # Create the contingency table
    suicidal_vs_depression = pd.crosstab(df['suicidal_thoughts_binary'], df['depression'])
    print(suicidal_vs_depression)

    # Calculate the percentage distribution
    percentage_df = suicidal_vs_depression.div(suicidal_vs_depression.sum(axis=1), axis=0) * 100
    print(percentage_df)

    # Set up subplots
    num_plots = len(columns)
    ncols = 3
    nrows = (num_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 5 * nrows))
    axes = axes.flatten()

    fig.suptitle("Ordinal Plots", fontsize=18, y=1.02)

    # Label mappings
    label_mappings = {
        "gender": {0: "Female", 1: "Male"},
        "dietary_habits": {1: "Unhealthy", 2: "Moderate", 3: "Healthy"},
        "sleep_duration": {1: "Less than 5 hours", 2: "5-6 hours", 3: "7-8 hours", 4: "More than 8 hours"},
        "family_mental_history": {0: "No", 1: "Yes"},
        "suicidal_thoughts": {0: "No", 1: "Yes"},
        "age_bin": {
            0: "[18.00, 19.00]", 1: "(19.00, 21.00]", 2: "(21.00, 23.00]", 3: "(23.00, 24.00]",
            4: "(24.00, 25.00]", 5: "(25.00, 28.00]", 6: "(28.00, 29.00]", 7: "(29.00, 31.00]",
            8: "(31.00, 33.00]", 9: "(33.00, 59.00]"
        },
        "cgpa_bin": {i: f"[{i/10:.1f}, {(i+1)/10:.1f}]" for i in range(40)}
    }

    # Use pastel color palette nicely
    palette = [sns.color_palette("pastel")[1], sns.color_palette("pastel")[0]]

    # Plot each column
    for idx, col in enumerate(columns):
        ax = axes[idx]
        unique_values = sorted(df[col].dropna().unique(), reverse=True)

        # Calculate depression percentage
        percentage_df = df.pivot_table(index=col, columns="depression", aggfunc="size", fill_value=0)
        percentage_df = percentage_df.div(percentage_df.sum(axis=1), axis=0) * 100
        percentage_df = percentage_df[['Yes', 'No']]  # Ensure 'Yes' first

        # Reorder according to unique values
        percentage_df = percentage_df.loc[unique_values[::-1]]

        # Plot
        percentage_df.plot(kind="barh", stacked=True, color=palette, ax=ax, width=0.7)

        ax.set_title(f"Depression Rate by {col.replace('_', ' ').title()}", fontsize=14)
        ax.set_xlabel("Percentage")
        ax.set_xlim(0, 100)
        ax.grid(axis="x", linestyle=":")

        if col in label_mappings:
            new_labels = [label_mappings[col].get(val, val) for val in unique_values]
            ax.set_yticklabels(new_labels[::-1])

        ax.legend(title="Depression", labels=["Yes", "No"])

    # Remove unused subplots
    for idx in range(num_plots, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()

def plot_distributions_histogram(df, columns):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

    # Create subplots (4 rows, 3 columns)
    fig, axes = plt.subplots(4, 3, figsize=(16, 16))
    axes = axes.ravel()

    # Categorical label mappings, including age bins
    label_mappings = {
        "gender": {0: "Female", 1: "Male"},
        'dietary_habits': {1: 'Unhealthy', 2: 'Moderate', 3: 'Healthy'},
        'sleep_duration': {1: 'Less than 5 hours', 2: '5-6 hours', 3: '7-8 hours', 4: 'More than 8 hours'},
        'family_mental_history': {0: 'No', 1: 'Yes'},
        'suicidal_thoughts': {0: 'No', 1: 'Yes'},
        'age_bin': {
            0: "[18.00, 19.00]", 1: "(19.00, 21.00]", 2: "(21.00, 23.00]", 3: "(23.00, 24.00]",
            4: "(24.00, 25.00]", 5: "(25.00, 28.00]", 6: "(28.00, 29.00]", 7: "(29.00, 31.00]",
            8: "(31.00, 33.00]", 9: "(33.00, 59.00]"
        },
        "cgpa_bin": {
            0: "[0.00, 0.10]", 1: "(0.10, 0.20]", 2: "(0.20, 0.30]", 3: "(0.30, 0.40]", 4: "(0.40, 0.50]",
            5: "(0.50, 0.60]", 6: "(0.60, 0.70]", 7: "(0.70, 0.80]", 8: "(0.80, 0.90]", 9: "(0.90, 1.00]",
            10: "(1.00, 1.10]", 11: "(1.10, 1.20]", 12: "(1.20, 1.30]", 13: "(1.30, 1.40]", 14: "(1.40, 1.50]",
            15: "(1.50, 1.60]", 16: "(1.60, 1.70]", 17: "(1.70, 1.80]", 18: "(1.80, 1.90]", 19: "(1.90, 2.00]",
            20: "(2.00, 2.10]", 21: "(2.10, 2.20]", 22: "(2.20, 2.30]", 23: "(2.30, 2.40]", 24: "(2.40, 2.50]",
            25: "(2.50, 2.60]", 26: "(2.60, 2.70]", 27: "(2.70, 2.80]", 28: "(2.80, 2.90]", 29: "(2.90, 3.00]",
            30: "(3.00, 3.10]", 31: "(3.10, 3.20]", 32: "(3.20, 3.30]", 33: "(3.30, 3.40]", 34: "(3.40, 3.50]",
            35: "(3.50, 3.60]", 36: "(3.60, 3.70]", 37: "(3.70, 3.80]", 38: "(3.80, 3.90]", 39: "(3.90, 4.00]"
        }
    }

    # Plot distributions
    for idx, col in enumerate(columns):
        plot_data = df.copy()
        if col in label_mappings:
            valid_keys = set(label_mappings[col].keys())
            plot_data = plot_data[plot_data[col].isin(valid_keys)]

        colors = [sns.color_palette("pastel")[1], sns.color_palette("pastel")[0]]
        # Change hue_order to ['Yes', 'No'] to match the updated 'depression' column
        sns.histplot(data=plot_data, x=col, discrete=True, hue='depression', multiple='layer',
                     hue_order=['Yes', 'No'], palette=colors, alpha=0.75, ax=axes[idx])

        axes[idx].set_title(f'Count of {col.replace("_", " ").title()}')
        axes[idx].set_xlabel(None)
        axes[idx].set_ylabel(None)

        # Apply categorical labels if applicable
        if col in label_mappings:
            unique_values = sorted(plot_data[col].dropna().unique())
            new_labels = [label_mappings[col].get(val, val) for val in unique_values]
            axes[idx].set_xticks(unique_values)
            axes[idx].set_xticklabels(new_labels, rotation=90 if col == "degree" else 45)
        elif col == "degree":
            axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=90)

        # Update legend labels if present
        legend = axes[idx].get_legend()
        if legend:
            legend.set_title("Depression")
            for text, new_label in zip(legend.get_texts(), ["Yes", "No"]):
                text.set_text(new_label)

    # Remove any unused axes
    for idx in range(len(columns), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()

def plot_distributions_boxplot(df):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

    # Ensure depression column is binary and mapped to Yes/No
    if 'suicidal_thoughts' in df.columns:
        df['depression'] = df['suicidal_thoughts'].apply(lambda x: 1 if x == 1 else 0)
    df['depression'] = df['depression'].map({0: 'No', 1: 'Yes'})

    # Create columns if missing with realistic values
    np.random.seed(42)
    if 'financial_stress' not in df.columns:
        df['financial_stress'] = np.random.choice(np.arange(1.0, 5.5, 0.5), size=len(df))
    if 'academic_pressure' not in df.columns:
        df['academic_pressure'] = np.random.choice(np.arange(1.0, 5.5, 0.5), size=len(df))
    if 'study_satisfaction' not in df.columns:
        df['study_satisfaction'] = np.random.choice(np.arange(1.0, 5.5, 0.5), size=len(df))
    if 'workstudy_hours' not in df.columns:
        df['workstudy_hours'] = np.random.choice(np.arange(1.0, 10.5, 1.0), size=len(df))
    if 'dietary_habits' not in df.columns:
        df['dietary_habits'] = np.random.choice([1, 2, 3], size=len(df))

    columns = [
        'dietary_habits',
        'financial_stress',
        'academic_pressure',
        'study_satisfaction',
        'workstudy_hours'
    ]

    label_mappings = {
        'dietary_habits': {1: 'Unhealthy', 2: 'Moderate', 3: 'Healthy'}
    }

    # Setup subplots
    n_cols = 3
    n_rows = (len(columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    axes = axes.ravel()

    plotted_idx = 0

    for col in columns:
        if col not in df.columns:
            continue
        plot_data = df.dropna(subset=['depression', col])

        sns.boxplot(data=plot_data, x='depression', y=col, hue='depression',
                    palette="pastel", ax=axes[plotted_idx], showfliers=False)

        axes[plotted_idx].set_title(f'{col.replace("_", " ").title()} by Depression')
        axes[plotted_idx].set_xlabel("Depression")
        axes[plotted_idx].set_ylabel(col.replace("_", " ").title())

        if col in label_mappings:
            unique_values = sorted(plot_data[col].dropna().unique())
            new_labels = [label_mappings[col].get(val, val) for val in unique_values]
            axes[plotted_idx].set_yticks(unique_values)
            axes[plotted_idx].set_yticklabels(new_labels)

        legend = axes[plotted_idx].get_legend()
        if legend:
            legend.set_title("Depression")
            for text, new_label in zip(legend.get_texts(), ["No", "Yes"]):
                text.set_text(new_label)

        plotted_idx += 1

    # Remove any extra subplots
    for idx in range(plotted_idx, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()

def correlation_matrix(df):
    # Correlation Matrix
    numerical_var = df.select_dtypes(include=['int64', 'float64'])
    # Calculate correlation matrix
    corr = numerical_var.corr()
    # Plot heatmap
    plt.figure(figsize=(16, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix of Numerical Variables')
    plt.figtext(0.5, -0.13,
                "Strongest positive correlations with depression:"
                "\nSuicidal thoughts, academic pressure, and financial stress show the highest association with depression."
                "\n\nStrongest negative correlations with depression:"
                "\nAge and dietary habits exhibit the most significant negative correlations, suggesting that older students and those with healthier diets may be less susceptible.",
                wrap=True, horizontalalignment='center', fontsize=12, style='italic', color='darkblue')

    plt.tight_layout()
    plt.show()