from sklearn.metrics import classification_report
import os
import pandas as pd

with open("D3_pseudo_scores.out","w",encoding="utf-8") as resultsFile:
    report = ""
    for output in os.listdir("../pseudo_outputs"):
        outputs = pd.read_csv("../pseudo_outputs/" + output)
        true_labels = outputs["true_labels"]
        predictions = outputs["predictions"]
        if output == "binary_output":
            targets = ['not sexist', 'sexist']
        elif output == "4_way_output":
            targets = ['1. threats, plans to harm and incitement', '2. derogation', '3. animosity', '4. prejudiced discussions']
        else:
            targets = ['1.1 threats of harm', '1.2 incitement and encouragement of harm', '2.1 descriptive attacks', '2.2 aggressive and emotive attacks', '2.3 dehumanising attacks & overt sexual objectification', '3.1 casual use of gendered slurs, profanities, and insults', '3.2 immutable gender differences and gender stereotypes', '3.3 backhanded gendered compliments', '3.4 condescending explanations or unwelcome advice', '4.1 supporting mistreatment of individual women', '4.2 supporting systemic discrimination against women as a group']
        report += classification_report(true_labels, predictions, target_names=targets)
        report += "\n"
    resultsFile.write(report)

