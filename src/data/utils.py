"""
This script contains all the utility functions that are not specific to a particular kind of dataset.
These are mainly used for explorations, testing, and demonstrations.
"""

# import argparse
# import subprocess
# import rasterio
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from google.cloud import storage

# HOME = str(Path.home())


def download_data_from_bucket(
    filenames: List[str],
    destination_dir: str,
    # ml_split: str = "train",
    bucket_id: Optional[str] = None,
) -> None:
    """Function to download data from the bucket to a local destination directory.
    This function differs from the save_file_from_bucket() function in that 
    it takes as input a list of filenames to be downloaded compared to save_file_from_bucket()
    which deals with only a single file. 
    Wraps around the save_file_from_bucket() function to download the list of files.

    Args:
        filenames (List[str]): List of filenames to be downloaded from the bucket.
        destination_dir (str): Path of the destination directory.
        bucket_id (str, optional): Name of the bucket being used to download the files. 
            Defaults to None.
    """    
    
    for ifile in filenames:
        bucket_id = str(Path(ifile).parts[0])

        file_name = str(Path(*Path(ifile).parts[1:]))
        save_file_from_bucket(
            bucket_id, file_name=file_name, destination_file_path=destination_dir
        )


def check_file_in_bucket_exists(
    bucket_name: str, filename_full_path: str, **kwargs
) -> bool:
    """
    Function to check if the file in the bucket exist utilizing Google Cloud Storage
    (GCP) blobs.

    Args:
      bucket_name (str): a string corresponding to the name of the GCP bucket.
      filename_full_path (str): a string containing the full path from bucket to file.

    Returns:
      A boolean value corresponding to the existence of the file in the bucket.
    """
    # initialize client
    client = storage.Client(**kwargs)
    # get bucket
    bucket = client.get_bucket(bucket_name)
    # get blob
    blob = bucket.blob(filename_full_path)
    # check if it exists
    return blob.exists()


def load_json_from_bucket(bucket_name: str, filename: str, **kwargs) -> Dict:
    """
    Function to load the json data for the WorldFloods bucket using the filename
    corresponding to the image file name. The filename corresponds to the full
    path following the bucket name through intermediate directories to the final
    json file name.

    Args:
      bucket_name (str): the name of the Google Cloud Storage (GCP) bucket.
      filename (str): the full path following the bucket_name to the json file.

    Returns:
      The unpacked json data formatted to a dictionary.
    """
    # initialize client
    client = storage.Client(**kwargs)
    # get bucket
    bucket = client.get_bucket(bucket_name)
    # get blob
    blob = bucket.blob(filename)
    # check if it exists
    # TODO: wrap this within a context
    return json.loads(blob.download_as_string(client=None))


# def generate_list_of_files(bucket_id: str, file_path: str):
#     """Generate a list of files within the mentioned filepath from the bucket."""

#     return None


def save_file_from_bucket(bucket_id: str, file_name: str, destination_file_path: str):
    """Function to save a file from a bucket to the mentioned destination file path.

    Args:
        bucket_id (str): the name of the bucket
        file_name (str): the name of the file in bucket (include the directory)
        destination_file_path (str): the directory of where you want to save the
            data locally (not including the filename)

    Returns:
        None: Returns nothing.

    Examples:
        >>> bucket_id = sample_bucket
        >>> file_name = 'path/to/file/and/file.csv'
        >>> dest = 'path/in/bucket/'
        >>> save_file_from_bucket(
        ...     bucket_id=bucket_id,
        ...     file_name=file_name,
        ...     destination_file_path=dest
        ... )
    """

    client = storage.Client()

    bucket = client.get_bucket(bucket_id)
    # get blob
    blob = bucket.get_blob(file_name)

    # create directory if needed
    create_folder(destination_file_path)

    # get full path
    destination_file_name = Path(destination_file_path).joinpath(
        file_name.split("/")[-1]
    )
    # download data
    blob.download_to_filename(str(destination_file_name))

    return None


def open_file_from_bucket(target_directory: str):
    """Function to open a file directly from the bucket.

    Args:
        target_directory (str): Complete filepath of the file to be opened
            within the session.

    Returns:
        google.cloud.storage.blob.Blob: Returns the blob of file 
            that is read into memory within the current session.
    
    Example:
        >>> target_directory = 'path/to/file/and/file.pkl'
        >>> open_file_from_bucket(target_directory)
    """    

    bucket_id, file_path, file_name = parse_gcp_path(target_directory)

    file_path = str(Path(file_path).joinpath(file_name))[1:]
    client = storage.Client()

    bucket = client.get_bucket(bucket_id)
    # get blob
    blob = bucket.get_blob(file_path)

    # download data
    blob = blob.download_as_string()

    return blob


def save_file_to_bucket(target_directory: str, source_directory: str):
    """Function to save file to a bucket.

    Args:
        target_directory (str): Destination file path.
        source_directory (str): Source file path

    Returns:
        None: Returns nothing.

    Examples:
        >>> target_directory = 'target/path/to/file/.pkl'
        >>> source_directory = 'source/path/to/file/.pkl'
        >>> save_file_to_bucket(target_directory)
    """

    client = storage.Client()

    bucket_id, _, _ = parse_gcp_path(target_directory)
    file_path = target_directory.split(bucket_id)[1][1:]

    bucket = client.get_bucket(bucket_id)

    # get blob
    blob = bucket.blob(file_path)

    # upload data
    blob.upload_from_filename(source_directory)

    return None


def check_path_exists(path: str) -> None:
    """Checks if the given exists.

    Args:
        path (str): Input file path

    Raises:
        ValueError: Raises an error in case the file path does not exist

    Returns:
        None: Returns nothing.
    """    
    if not Path(path).is_dir():
        raise ValueError(f"Unrecognized path: {str(Path(path))}")
    return None


def create_folder(directory: str) -> None:
    """Function to create directory if it doesn't exist.

    Args:
        directory (str): directory to be created if it doesn't already exist.
    
    Example:
        >>> directory = "./temp"
        >>> create_folder(directory)
    """

    try:
        Path(directory).mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"Folder '{directory}' Is Already There.")
    else:
        print(f"Folder '{directory}' is created.")


def get_files_in_directory(directory: str, suffix: str) -> List[str]:
    """Function to return the list of files within a given directory.

    Args:
        directory (str): Directory path to get the file list from.
        suffix (str): file extension to be listed

    Returns:
        List[str]: Returns the list of files that match the given extension
            within the given directory.
    """
    p = Path(directory).glob(f"*{suffix}")
    files = [str(x) for x in p if x.is_file()]
    return files


# TODO: This is a redundant function. 
# Refactor all the code to use only one of 
# these two functions and get rid of the redundant function.
def get_filenames_in_directory(directory: str, suffix: str) -> List[str]:
    """Function to return the list of files within a given directory.

    Args:
        directory (str): Directory path to get the file list from.
        suffix (str): file extension to be listed

    Returns:
        List[str]: Returns the list of files that match the given extension
            within the given directory.
    """
    p = Path(directory).glob(f"*{suffix}")
    files = [str(x.name) for x in p if x.is_file()]
    return files


# def get_files_in_bucket_directory(
#     bucket_id: str, directory: str, suffix: str
# ) -> List[str]:
#     p = Path(directory).glob(f"*{suffix}")
#     files = [str(x) for x in p if x.is_file()]
#     return files


def get_files_in_bucket_directory(
    bucket_id: str, directory: str, suffix: str, **kwargs
) -> List[str]:
    """Function to return a list of files in bucket directory
    Args:
        bucket_id (str): the bucket name to query
        directory (str): the directory within the bucket to query
        suffix (str): the filename suffix, e.g. '.tif'
        full_path (bool): whether to add the full path to filenames or not
    Returns:
        files (List[str]): a list of filenames with the fullpaths
    """

    # initialize client
    client = storage.Client(**kwargs)
    # get bucket
    bucket = client.get_bucket(bucket_id)
    # get blob
    blobs = bucket.list_blobs(prefix=directory)
    # check if it exists

    files = [str(x.name) for x in blobs if str(Path(x.name).suffix) == suffix]
    return files


def parse_gcp_path(full_path: str) -> Tuple[str]:
    """Function to parse a GCP bucket file path into smaller components.

    Args:
        full_path (str): Full path of a file within a GCP bucket.

    Returns:
        Tuple[str]: Returns a tuple of substrings from the full_path that include
            bucket_id, filepath, and filename.
    """    
    # parse the components
    bucket_id = str(Path(full_path.split("gs://")[1]).parts[0])
    file_path = str(Path(full_path.split(bucket_id)[1]).parent)
    file_name = str(Path(full_path).name)

    return bucket_id, file_path, file_name