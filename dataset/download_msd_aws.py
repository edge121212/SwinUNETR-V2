import os
import urllib.request
import tarfile
import argparse

def download_and_extract(url, target_path):
    print(f"開始下載 {target_path} ... (檔案較大，請耐心等候)")
    urllib.request.urlretrieve(url, target_path)
    
    print(f"下載完成，正在解壓縮 {target_path} ...")
    with tarfile.open(target_path) as tar:
        def is_within_directory(directory, target):
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
            prefix = os.path.commonprefix([abs_directory, abs_target])
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        safe_extract(tar)
    print(f"{target_path} 解壓縮完畢！\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download MSD datasets")
    parser.add_argument("--task", type=str, default="All", 
                        choices=["Prostate", "Lung", "Pancreas", "All"],
                        help="Choose which dataset to download: Prostate, Lung, Pancreas, or All")
    args = parser.parse_args()

    all_tasks = {
        "Prostate": "Task05_Prostate.tar",
        "Lung": "Task06_Lung.tar",
        "Pancreas": "Task07_Pancreas.tar"
    }

    if args.task == "All":
        tasks_to_download = list(all_tasks.values())
    else:
        tasks_to_download = [all_tasks[args.task]]

    base_url = "https://msd-for-monai.s3.amazonaws.com/"

    for task in tasks_to_download:
        url = base_url + task
        target_path = task
        extracted_folder = task.replace('.tar', '')
        
        if not os.path.exists(extracted_folder):
            download_and_extract(url, target_path)
        else:
            print(f"{extracted_folder} 已經存在，跳過下載。")

    print("所有指定資料集處理完畢！")
