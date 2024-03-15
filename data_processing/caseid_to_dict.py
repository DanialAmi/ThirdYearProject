import os
import pickle

def create_caseid_dictionary(source_folder):
    caseid_to_document = {}

    for folder_name in os.listdir(source_folder):
        folder_path = os.path.join(source_folder, folder_name)
        if os.path.isdir(folder_path):  # Check if it's a directory
            for filename in os.listdir(folder_path):
                if filename.endswith('.txt'):
                    category = folder_name  # Assuming the category is the folder name
                    case_id = filename.split('.')[0]
                    full_case_id = category + "_" + case_id
                    with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                        text = file.read()
                        caseid_to_document[full_case_id] = text
                        
    return caseid_to_document

folder_path = "legal_case"
caseid_document_dict = create_caseid_dictionary(folder_path)

print(len(caseid_document_dict.keys()))
print(len(caseid_document_dict.values()))

output_file_name = "caseid_to_document.pkl"

# Use a with statement to open a file for writing in binary mode
with open(output_file_name, 'wb') as fOut:
    # Use pickle.dump() to serialize your dictionary and write it to the file
    pickle.dump(caseid_document_dict, fOut)

print(f"Serialized dictionary saved to {output_file_name}")