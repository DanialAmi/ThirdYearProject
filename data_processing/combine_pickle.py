import os
import pickle

def combine_pickle(pickle_dir="pickle_files"):
    combined_data = {
        'embeddings': [],
        'case_ids': [],
        'paragraphs':[]
    }
    
    for filename in os.listdir(pickle_dir):
        if filename.endswith('.pkl'):
            filepath = os.path.join(pickle_dir, filename)
            with open(filepath, 'rb') as file:
                data = pickle.load(file)
                combined_data['embeddings'].extend(data['embeddings'])
                combined_data['case_ids'].extend(data['case_ids'])
                combined_data['paragraphs'].extend(data['paragraphs'])
    
    print("successfully combined")
    print(len(combined_data['case_ids']))
    print(len(combined_data['embeddings']))
    print(len(combined_data['paragraphs']))
    return combined_data