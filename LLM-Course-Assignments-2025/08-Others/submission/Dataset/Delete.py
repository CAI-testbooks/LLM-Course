import os


def process_all_subjects(base_path):
    if not os.path.exists(base_path):
        print(f"Base path does not exist: {base_path}")
        return

    for subject in os.listdir(base_path):
        subject_path = os.path.join(base_path, subject)

        if os.path.isdir(subject_path):
            print(f"Processing subject: {subject}")
            for root, dirs, files in os.walk(subject_path):
                for file in files:
                    if file.endswith(".tsv") or file.endswith(".csv"):
                        file_path = os.path.join(root, file)
                        try:
                            os.remove(file_path)
                            print(f"Deleted file: {file_path}")
                        except Exception as e:
                            print(f"Error deleting file: {file_path}, Error: {e}")



base_path =
process_all_subjects(base_path)
