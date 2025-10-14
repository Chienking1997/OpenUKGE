import os
import requests
import zipfile
from typing import Dict

DATASET_URLS: Dict[str, Dict[str, str]] = {
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
    }
}


def download_file(url: str, save_path: str) -> None:
    """
    Download a file from the specified URL and save it locally.

    Args:
        url (str): The download URL.
        save_path (str): The path to save the downloaded file.

    Raises:
        Exception: If the download fails or returns a non-200 HTTP status.
    """
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
    """
    Extract a ZIP file to a target directory.

    Args:
        file_path (str): Path to the ZIP file.
        extract_to (str): Directory to extract the contents into.

    Raises:
        zipfile.BadZipFile: If the file is not a valid ZIP archive.
    """
    print(f"Extracting ZIP file: {file_path}")
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"✅ Extraction complete. Files extracted to: {extract_to}")


def download_dataset(dataset_name: str, download_path: str = "datasets") -> str:
    """
    Download and extract a dataset.
    Priority: GitHub → Gitee (as a fallback).

    Args:
        dataset_name (str): Name of the dataset (e.g. 'cn15k', 'nl27k').
        download_path (str): Directory to store the downloaded and extracted data. Default is 'datasets'.

    Returns:
        str: Path to the extracted dataset folder.

    Raises:
        ValueError: If the dataset name does not exist in DATASET_URLS.
        Exception: If both GitHub and Gitee downloads fail.
    """
    if dataset_name not in DATASET_URLS:
        raise ValueError(f"Dataset '{dataset_name}' is not available in DATASET_URLS.")

    # Ensure the download directory exists
    os.makedirs(download_path, exist_ok=True)

    dataset_zip_path = os.path.join(download_path, f"{dataset_name}.zip")
    dataset_extract_path = os.path.join(download_path, dataset_name)

    # Check if dataset is already downloaded and extracted
    if os.path.exists(dataset_extract_path):
        print(f"ℹ️ Dataset '{dataset_name}' already exists at {dataset_extract_path}. Skipping download.")
        return dataset_extract_path

    urls = DATASET_URLS[dataset_name]

    # Attempt GitHub download first
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

    # Extract the downloaded ZIP file
    extract_zip(dataset_zip_path, dataset_extract_path)

    return dataset_extract_path


if __name__ == '__main__':
    download_dataset('onet20k', 'datasets') # debug