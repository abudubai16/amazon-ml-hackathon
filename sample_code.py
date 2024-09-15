import os
import random
import pandas as pd
from src.model import final_model

def predictor(image_link, category_id, entity_name):
    return final_model(image_url=image_link, entity_name=entity_name)
    # return "" if random.random() > 0.5 else "10 inch"

if __name__ == "__main__":
    DATASET_FOLDER = '../dataset/'
    PATH = r'C:\Users\faroo\OneDrive\Documents\66e31d6ee96cd_student_resource_3\student_resource 3\dataset\test.csv'
    # test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
    test = pd.read_csv(PATH)
    rows = len(test)
    test = test[(int)(rows*0.5):rows]

    test['prediction'] = test.apply(
        lambda row: predictor(row['image_link'], row['group_id'], row['entity_name']), axis=1,)
    

    output_filename = PATH
    test[['index', 'prediction']].to_csv(output_filename, index=False)