'''
Script aiming at managing different types of documents.
    Input:
        File path.
    Output:
        File is treated (e.g. unzipped and content put in a folder).
'''


def unzip_to_folder(file_path, extract_folder):
    """
    Unzip zip file to a folder with the same name inside the extract_folder.

    Parameters
    ----------
    file_path: string
        Full path of zip file.
    extract_folder: string
        Folder where to extract the uncompressed content.

    """
    if not os.path.exists(extract_folder):
        os.makedirs(extract_folder)

    # Create folder for specific zip file with the same name
    path, file = os.path.split(file_path)
    zip_folder = f"{extract_folder}/{file.replace('.zip', '')}"

    if not os.path.exists(zip_folder):
        os.makedirs(zip_folder)
        try:
            with zipfile.ZipFile(file_path) as zip_file:
                zip_file.extractall(zip_folder)
        except zipfile.BadZipFile:
            print(f"{file} is not a zip file!")
    else:
        print(f"{file_path} had been already unzipped.")
