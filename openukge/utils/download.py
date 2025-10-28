import os
import requests
import zipfile
from typing import Dict

DATASET_URLS: Dict[str, Dict[str, str]] = {
    # Regular datasets
    "cn15k": {
        "github": "https://raw.githubusercontent.com/Chienking1997/OpenUKGE/dataset-branch/data/cn15k.zip",
        "gitee": "https://gitee.com/Chienking/OpenUKGE/raw/dataset-branch/data/cn15k.zip"
    },
    "nl27k": {
        "github": "https://raw.githubusercontent.com/Chienking1997/OpenUKGE/dataset-branch/data/nl27k.zip",
        "gitee": "https://gitee.com/Chienking/OpenUKGE/raw/dataset-branch/data/nl27k.zip"
    },
    "ppi5k": {
        "github": "https://raw.githubusercontent.com/Chienking1997/OpenUKGE/dataset-branch/data/ppi5k.zip",
        "gitee": "https://gitee.com/Chienking/OpenUKGE/raw/dataset-branch/data/ppi5k.zip"
    },
    "onet20k": {
        "github": "https://raw.githubusercontent.com/Chienking1997/OpenUKGE/dataset-branch/data/onet20k.zip",
        "gitee": "https://gitee.com/Chienking/OpenUKGE/raw/dataset-branch/data/onet20k.zip"
    },

    # Few-shot datasets
    "cn15k-few-shot": {
        "github": "https://raw.githubusercontent.com/Chienking1997/OpenUKGE/dataset-branch/few-shot-data/cn15k.zip",
        "gitee": "https://gitee.com/Chienking/OpenUKGE/raw/dataset-branch/few-shot-data/cn15k.zip"
    },
    "nl27k-few-shot": {
        "github": "https://raw.githubusercontent.com/Chienking1997/OpenUKGE/dataset-branch/few-shot-data/nl27k.zip",
        "gitee": "https://gitee.com/Chienking/OpenUKGE/raw/dataset-branch/few-shot-data/nl27k.zip"
    }
}


def download_file(url: str, save_path: str) -> None:
    print(f"Downloading from {url} ...")
    response = requests.get(url, stream=True, timeout=10)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"✅ File downloaded successfully: {save_path}")
    else:
        raise Exception(f"Failed to download file. HTTP status code: {response.status_code}")


def extract_zip(file_path: str, extract_to: str) -> None:
    print(f"Extracting ZIP file: {file_path}")
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"✅ Extraction complete. Files extracted to: {extract_to}")


def dataset_exists(dataset_name: str, dataset_extract_path: str) -> bool:
    """
    Custom existence check:
    - If downloading few-shot: check <base_path>/few-shot/
    - If downloading normal dataset: check <base_path>/ (excluding few-shot/)
    """
    if dataset_name.endswith("-few-shot"):
        few_shot_path = os.path.join(dataset_extract_path, "few-shot")
        return os.path.exists(few_shot_path)
    else:
        if not os.path.exists(dataset_extract_path):
            return False
        # check whether there's any non-few-shot content
        sub_dirs = [d for d in os.listdir(dataset_extract_path)
                   if os.path.isdir(os.path.join(dataset_extract_path, d))]
        files = [f for f in os.listdir(dataset_extract_path)
                 if os.path.isfile(os.path.join(dataset_extract_path, f))]
        # If there are files or subfolders other than few-shot, assume dataset exists
        if files or any(d != "few-shot" for d in sub_dirs):
            return True
        return False


def download_dataset(dataset_name: str, download_path: str = "datasets") -> str:
    """
    Download and extract a dataset.
    - Normal dataset → datasets/{dataset_name}/
    - Few-shot dataset → datasets/{base_name}/few-shot/
    """
    if dataset_name not in DATASET_URLS:
        raise ValueError(f"Dataset '{dataset_name}' is not available in DATASET_URLS.")

    os.makedirs(download_path, exist_ok=True)
    dataset_zip_path = os.path.join(download_path, f"{dataset_name}.zip")

    # Determine final extraction path
    if dataset_name.endswith("-few-shot"):
        base_name = dataset_name.replace("-few-shot", "")
        dataset_extract_path = os.path.join(download_path, base_name)
        target_subpath = os.path.join(dataset_extract_path, "few-shot")
    else:
        dataset_extract_path = os.path.join(download_path, dataset_name)
        target_subpath = dataset_extract_path

    # Custom existence check
    if dataset_exists(dataset_name, dataset_extract_path):
        print(f"ℹ️ Dataset '{dataset_name}' already exists at {target_subpath}. Skipping download.")
        return target_subpath

    urls = DATASET_URLS[dataset_name]

    # Try GitHub first, fallback to Gitee
    try:
        print(f"Attempting to download '{dataset_name}' from GitHub...")
        download_file(urls["github"], dataset_zip_path)
    except Exception as e:
        print(f"⚠️ GitHub download failed: {e}")
        print(f"Trying to download '{dataset_name}' from Gitee...")
        try:
            download_file(urls["gitee"], dataset_zip_path)
        except Exception as e2:
            raise Exception(
                f"❌ Failed to download dataset '{dataset_name}' from both GitHub and Gitee.\n"
                f"GitHub error: {e}\nGitee error: {e2}"
            )

    # Ensure target folder exists before extraction
    if dataset_name.endswith("-few-shot"):
        os.makedirs(dataset_extract_path, exist_ok=True)
        extract_zip(dataset_zip_path, dataset_extract_path)
    else:
        os.makedirs(target_subpath, exist_ok=True)
        extract_zip(dataset_zip_path, target_subpath)

    print(f"📦 Dataset ready at: {target_subpath}")
    return target_subpath


if __name__ == '__main__':
    # Example usage
    download_dataset('nl27k', 'datasets')
    download_dataset('nl27k-few-shot', 'datasets')
