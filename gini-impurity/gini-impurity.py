import pandas as pd

def get_excelf(file_path):
    """Reads the Excel file and returns the DataFrame, otherwise returns None."""
    try:
        df = pd.read_excel(file_path)
        if df.empty or df.shape[1] < 2:
            print("It looks the Excel file is empty, Please check it and try again.")
            return None
        # if df.shape[1] < 2:
        #     print("The Excel file must have at least two columns.")
        #     return None
        return df
    except Exception as e:
        print(f"Error reading the Excel file: {e}")
        return None

def calculate_gini(df):
    """Calculates Gini impurity for each feature column in the DataFrame."""
    label_col = df.columns[-1]
    gini_result = {}

    for feature in df.columns[:-1]:
        print("\033[1;32m"f"\nfor '{feature}'\033[0m")

        impurity_values = []
        for value in df[feature].unique():
            subset_labels = df[df[feature] == value][label_col]
            total = len(subset_labels)

            class_counts = subset_labels.value_counts()
            prob_yes = 0
            if "yes" in class_counts:
                prob_yes = class_counts["yes"] / total
            prob_no = 1 - prob_yes

            impurity = 1 - (prob_yes ** 2 + prob_no ** 2)
            impurity_values.append(impurity)

            print("\033[1;33m"f"    Value '{value}':""\033[0m" f" [Gini = {impurity:.4f}], [Prob = {prob_yes:.3f}]")

        gini_result[feature] = sum(impurity_values) / len(impurity_values)

    print("\033[0;32m""-" * 55 + "\033[0m")
    return gini_result

def main():
    file_path = "example-table1.xlsx"
    df = get_excelf(file_path)

    if df is None:
        return

    gini_scores = calculate_gini(df)

    if gini_scores:
        print("\nFinal Gini Impurity results for features:")
        min_gini = min(gini_scores.values())
        for feature, gini in gini_scores.items():
            if gini == min_gini:
                print("Feature""\033[1;31m" f" '{feature}'""\033[0m"" : Gini Impurity =""\033[1;31m" f" {gini:.4f}" + "\033[0m")
            else:
                print(f"Feature '{feature}': Gini Impurity = {gini:.4f}")
    input(">>>")

main()