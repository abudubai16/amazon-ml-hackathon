import os
import time
import pandas as pd
from src.model import final_model

def predictor(image_link, category_id, entity_name):
    return final_model(image_url=image_link, entity_name=entity_name)
    # return "" if random.random() > 0.5 else "10 inch"

# Function to save progress to a CSV
def save_progress(df, output_filename):
    if not os.path.exists(output_filename):
        df[['index', 'prediction']].to_csv(output_filename, index=False)
    else:
        df[['index', 'prediction']].to_csv(output_filename, mode='a', header=False, index=False)

# Load existing progress if available
def load_existing_progress(output_filename):
    if os.path.exists(output_filename):
        return pd.read_csv(output_filename)
    return pd.DataFrame()

if __name__ == "__main__":
    PATH = r'D:\Amazon ML hackathon\dataset\test.csv'
    OUTPUT_PATH = r'D:\Amazon ML hackathon\dataset\output.csv'  # Save progress here
    
    while True:  # Infinite loop to continue running even after crashes
        try:
            # Load the test data
            test = pd.read_csv(PATH)

            # Load previously saved progress if it exists
            existing_data = load_existing_progress(OUTPUT_PATH)

            # Identify rows that have not been processed yet
            if not existing_data.empty:
                processed_indices = existing_data['index'].tolist()
                test = test[~test['index'].isin(processed_indices)]  # Skip already processed rows

            rows = len(test)
            test = test[:(int)(rows * 0.3)]  # Process 50% of the dataset

            # Initialize an empty list to store the predictions
            predictions = []

            # Iterate through the rows and make predictions
            for index, row in test.iterrows():
                # Generate a prediction
                prediction = predictor(row['image_link'], row['group_id'], row['entity_name'])
                predictions.append((row['index'], prediction))

                # Periodically save progress (every 100 rows, for example)
                if len(predictions) % 100 == 0:
                    # Convert predictions to a DataFrame and save progress
                    progress_df = pd.DataFrame(predictions, columns=['index', 'prediction'])
                    save_progress(progress_df, OUTPUT_PATH)
                    predictions.clear()  # Clear the list after saving

            # Save any remaining predictions after the loop ends
            if predictions:
                progress_df = pd.DataFrame(predictions, columns=['index', 'prediction'])
                save_progress(progress_df, OUTPUT_PATH)
            
            # Exit loop if all rows have been processed
            print("Processing complete.")
            break

        except Exception as e:
            print(f"An error occurred: {e}")
            print("Retrying in 10 seconds...")
            time.sleep(10)  # Wait for 10 seconds before retrying