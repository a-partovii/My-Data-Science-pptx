import pandas as pd
from termcolor import colored

def get_excelf(file_path):
    """Reads the Excel file and returns the DataFrame, otherwise returns None."""
    try:
        df = pd.read_excel(file_path)
        if df.shape[1] < 2:
            print("It looks like the Excel file is empty. Please check it and try again.")
            return None
        return df
    except Exception as e:
        print(f"Error reading the Excel file: {e}")
        return None

def get_target_input(df):
    while True:
        label_col = input("Enter the name of the label column: ")
        if label_col not in df.columns:
            print(colored("-" * 55, "light_green"))  # Graphical underline in the terminal
            print("There is no column with this name. Available columns are:")
            print(colored(df.columns.to_list(), "cyan"))
        else:
            break

    while True:
        target = input("Enter your target value: ")
        if target not in df[label_col].astype(str).unique():
            print(colored("-" * 55, "light_green"))
            print("There is no item with this name in that column. Available values are:")
            values = df[label_col].unique()
            print(colored(f"{label_col}: {list(values)}", "cyan"))
        else:
            break
    return label_col, target

def calculate_gini(df, label_col, target):
    """Calculates Gini impurity for each feature column based on selected label column and targer value."""
    gini_result = {}
    for feature in df.columns:
        if feature == label_col:
            continue  # Skip the label column itself

        print(colored(f"\nfor '{feature}'", "light_green"))
        values_impurity = []

        for value in df[feature].unique():
            subset_labels = df[df[feature] == value][label_col]
            total = len(subset_labels)

            class_counts = subset_labels.value_counts()
            prob_pos = 0

            if target in class_counts:
                prob_pos = class_counts[target] / total
            prob_neg = 1 - prob_pos

            impurity = 1 - (prob_pos ** 2 + prob_neg ** 2)

            values_impurity.append(impurity)

            if prob_pos == 1:
                print(colored(f"    Value '{value}':", "yellow"),
                    f"[Gini = {impurity:.3f}],",
                    colored(f"[positive prob = {prob_pos:.3f}]", "red"),
                    f",[negative prob = {prob_neg:.3f}]")
            elif prob_neg == 1:
                print(colored(f"    Value '{value}':", "yellow"),
                    f"[Gini = {impurity:.3f},",
                    f"[positive prob = {prob_pos:.3f}],",
                    colored(f"[negative prob = {prob_neg:.3f}]", "red"))
            else:
                print(colored(f"    Value '{value}':", "yellow"),
                    f"[Gini = {impurity:.3f}],",
                    f"[positive prob = {prob_pos:.3f}],",
                    f"[negative prob = {prob_neg:.3f}]")

        gini_result[feature] = sum(values_impurity) / len(values_impurity) if values_impurity else 0

    print(colored("-" * 55, "light_green"))
    return gini_result

def main():
    file_path = "example-table1.xlsx"
    df = get_excelf(file_path)

    if df is None:
        return

    label_col, target = get_target_input(df)
    gini_scores = calculate_gini(df, label_col, target)

    if gini_scores:
        print("\nFinal Gini Impurity results for features:")
        min_gini = min(gini_scores.values())
        for feature, gini in gini_scores.items():
            if gini == min_gini:
                print("Feature", colored(f" '{feature}'", "light_red", attrs=['bold']),
                      "Gini Impurity =", colored(f" {gini:.4f}", "light_red", attrs=['bold']))
            else:
                print(f"Feature '{feature}': Gini Impurity = {gini:.4f}")
    input(">>>")

while True:
    main()
