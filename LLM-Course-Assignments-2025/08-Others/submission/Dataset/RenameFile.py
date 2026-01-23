import os

def clean_all_task_folder_names(root_path):
    if not os.path.exists(root_path):
        print(f"Path does not exist: {root_path}")
        return

    for subject in os.listdir(root_path):
        subject_path = os.path.join(root_path, subject)
        if not os.path.isdir(subject_path):
            continue

        print(f"Processing subject: {subject}")
        for dir_name in os.listdir(subject_path):
            task_path = os.path.join(subject_path, dir_name)
            if not os.path.isdir(task_path):  # 跳过非文件夹
                continue

            # 修改任务文件夹名称
            new_name = dir_name.replace("task-", "").replace("_space-fsaverage6", "")
            if new_name != dir_name:  # 如果有改动
                new_path = os.path.join(subject_path, new_name)
                try:
                    os.rename(task_path, new_path)
                    print(f"Renamed: {task_path} -> {new_path}")
                except Exception as e:
                    print(f"Error renaming {task_path}: {e}")

# 指定路径
root_path =
clean_all_task_folder_names(root_path)
