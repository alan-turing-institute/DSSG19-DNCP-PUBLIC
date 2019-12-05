'''
Script aiming at extracting information from documents.
    Input:
        Files.
    Output:
        File information.
'''

# Import Scripts
from pathlib import Path
import os
import sys
source_path = str(Path(os.path.abspath(__file__)).parent.parent)
pipeline_path = str(Path(os.path.abspath(__file__)).parent)
sys.path = [i for i in sys.path if i != pipeline_path]

if source_path not in sys.path:
    sys.path.insert(0, source_path)

import csv
import datetime as dt
import fire
import logging
import pandas as pd
import os
from pdfminer.pdfparser import PDFParser, PDFSyntaxError
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfdevice import PDFDevice
from pdfminer.pdfpage import PDFTextExtractionNotAllowed
from pdfminer.layout import LAParams, LTTextBox, LTTextLine
from pdfminer.converter import PDFPageAggregator
from PyPDF2 import PdfFileReader
from tika import parser
from tqdm import tqdm

from utils.utils import connect_to_database, load_pandas_to_db

logging.basicConfig(level=logging.INFO)


# Settings
first_pbc_path = '/data/original_data/first_pbc'
all_pbc_path = '/data/original_data/docs'
more_pbc_path = '/data/original_data/docs_0207'
txt_folder_template = '/data/shared_data/data/raw/txt_extracted/{method}/'
output_template = '/data/shared_data/data/output/{date}_documents_extractability_{method}.csv'
header = "filename,is_extractable,number_of_pages\n"


def check_or_create_output_file(file_path):
    """
    Given a PDF file complete path, the function parses the file, counts the number of pages and checks if
    it is text-extractable.

    Parameters
    ----------
    file_path: String
        Complete path to output file

    Return
    ------
    List of filenames in the output file, meaning that they have already been checked.
    """

    if os.path.isfile(file_path):
        output_df = pd.read_csv(file_path)
        return [file for file in output_df['filename']]
    else:
        with open(file_path, 'w') as f:
            f.write(header)
        logging.info(f"Creating output file {file_path}")
        return []


def parse_pdf(file_path, method='tika'):
    """
    Given a PDF file complete path, the function parses the file, counts the number of pages and checks if
    it is text-extractable.

    Parameters
    ----------
    file_path: string
        Complete path to output file.
    method: string
        Method used to extract the text: 'pdfminer', 'pypdf', 'tika'.

    Return
    ------
    extracted_text: string
        Text extracted from the document.
    number_of_pages:
        Number of pages of the document.
    """

    if method == 'pdfminer':
        with open(file_path, "rb") as fp:
            # Create parser object to parse the pdf content
            pdf_parser = PDFParser(fp)

            # Store the parsed content in PDFDocument object
            document = PDFDocument(pdf_parser)

            # Check if document is text-extractable or not
            is_extractable = document.is_extractable

            # Check if document is extractable, if not abort
            if not is_extractable:
                raise PDFTextExtractionNotAllowed

            # Create PDFResourceManager object that stores shared resources such as fonts or images
            rsrcmgr = PDFResourceManager()

            # Set parameters for analysis
            laparams = LAParams()

            # Extract the decive to page aggregator to get LT object elements
            device = PDFPageAggregator(rsrcmgr, laparams=laparams)

            # Create interpreter object to process page content from PDFDocument
            interpreter = PDFPageInterpreter(rsrcmgr, device)

            extracted_text = ""
            number_of_pages = 0

            # Process PDF document page by page
            for page in PDFPage.create_pages(document):
                number_of_pages = number_of_pages + 1
                extracted_text += f"[Page {number_of_pages}]\n"

                # As the interpreter processes the page stored in PDFDocument object
                interpreter.process_page(page)

                # The device renders the layout from interpreter
                layout = device.get_result()

                # Out of the many LT objects within layout, we are interested in LTTextBox and LTTextLine
                for lt_obj in layout:
                    if isinstance(lt_obj, LTTextBox) or isinstance(lt_obj, LTTextLine):
                        extracted_text += lt_obj.get_text()

    if method == 'pypdf':
        with open(file_path, 'rb') as f:
            pdf = PdfFileReader(f)
            number_of_pages = pdf.getNumPages()
            extracted_text = ''.join([f'[Page {i}]\n' + pdf.getPage(i).extractText() for i in range(number_of_pages)])

    if method == 'tika':
        raw = parser.from_file(file_path)
        extracted_text = raw['content']
        number_of_pages = int(raw['metadata']['xmpTPg:NPages'])

    else:
        logging.error(f'Text extractor method {method} not found')

    return extracted_text, number_of_pages


def check_extractability(text):
    """
    Given a text, check if it has some content.

    Parameters
    ----------
    text: string
        Text.

    Return
    ------
    True or False.
    """

    if text == "" or text is None:
        return False
    return True


def save_extracted_text(text, txt_filename, txt_folder):
    """
    Given a text, save it to a given filename in a folder.

    Parameters
    ----------
    text: string
        Text.
    txt_filename: string
        Name of the file to output the text.
    txt_folder: string
        Name of the folder where to save the txt file.
    """

    with open(f"{txt_folder}/{txt_filename}.txt", "w") as f:
        f.write(text)


def extract_text_from_pdf(file_path, txt_filename, txt_folder, method, save_txt=True):
    """
    Given a PDF file, extract the text and get some file characteristics.
    If set, save extracted text to txt file.

    Parameters
    ----------
    file_path: string
        Full path of the file where to extract the text from.
    txt_filename: string
        Name of the file to output the text.
    txt_folder: string
        Name of the folder where to save the txt file.
    method: string
        Method used to extract the text: 'pdfminer', 'pypdf', 'tika'.
    save_txt: boolean
        Whether to save the extracted text in a txt file or not.

    Returns
    -------
    is_extractable: boolean
        Whether the file is text extractable or not.
    number_of_pages: integer
        Number of pages of the file.
    """

    text, number_of_pages = parse_pdf(file_path=file_path, method=method)

    is_extractable = check_extractability(text)

    if is_extractable and save_txt:
        save_extracted_text(text=text, txt_filename=txt_filename, txt_folder=txt_folder)

    return is_extractable, number_of_pages


def list_documents_to_extract(output_file):
    """
    Get the list of documents (complete path) from which to extract text and characteristics.

    Parameters
    ----------
    output_file: string
        Output file where previous runs are stored.

    Returns
    -------
    docs_to_check: list
        List of documents.
    """

    docs_to_check = []
    checked_files = set(check_or_create_output_file(output_file))

    con = connect_to_database()

    # Retrieve all documents from DB
    query = f"select id_llamado, nombre_archivo as filename" \
        f" from raw.pbc_adenda" \
        f" where lower(tipo_documento) != 'adenda'" \
        f" and right(lower(nombre_archivo), 3) = 'pdf'"
    all_documents = pd.read_sql_query(query, con)

    # List document folders
    first_pbc = sorted([file for file in os.listdir(first_pbc_path) if file.lower().endswith('.pdf')])
    all_pbc = sorted([file for file in os.listdir(all_pbc_path) if file.lower().endswith('.pdf')])
    all_pbc_2 = sorted([file for file in os.listdir(more_pbc_path) if file.lower().endswith('.pdf')])

    set_first_pbc = set(first_pbc)
    set_all_pbc = set(all_pbc + all_pbc_2)

    ids = all_documents['id_llamado'].unique()
    for i in tqdm(range(len(ids))):
        id = ids[i]
        docs_id = all_documents[all_documents['id_llamado'] == id].filename.tolist()
        set_docs = set(docs_id)

        # First, check if there are documents in the first PBC folder
        intersect_first = list(set_docs & set_first_pbc)
        len_intersect = len(intersect_first)

        # If there is more than one document in first PBC, there is a problem
        if len_intersect > 1:
            logging.error(f'{id} has {len_intersect} documents in first PBC.')
            continue

        # If there is one document in first PBC, we add it to the output list
        elif len_intersect == 1:
            curr_doc = intersect_first[0]
            if curr_doc in checked_files:
                continue
            else:
                docs_to_check.append(f"{first_pbc_path}/{curr_doc}")

        else:
            intersect_all = list(set_docs & set_all_pbc)

            # If there is no document in first PBC, but only one document existing, we add it
            if len(intersect_all) == 1:
                curr_doc = intersect_all[0]
                if curr_doc in checked_files:
                    continue
                else:
                    if curr_doc in all_pbc:
                        docs_to_check.append(f"{all_pbc_path}/{curr_doc}")
                    else:
                        docs_to_check.append(f"{more_pbc_path}/{curr_doc}")
            # If there is no document in first PBC and more than one in the table, we have a problem
            else:
                logging.error(f'{id} has {len(intersect_all)} documents in PBC folders.')
                continue

    return sorted(list(set(docs_to_check)))


def extract_text_from_pdfs_list(docs_list, method, output_file):
    """
    Extract text from a list of PDFs, using the method specified.
    The characteristics of the file (is_extractable, number_of_pages) are written in an output csv file,
    and the extracted text is saved in a txt file with the same name as the document.

    Parameters
    ----------
    docs_list: list

    method: string
        Method used to extract the text: 'pdfminer', 'pypdf', 'tika'.
    output_file: string

    """

    f = open(output_file, 'a')

    for i in tqdm(range(len(docs_list))):
        file_path = docs_list[i]
        filename = file_path.split('/')[-1]

        is_extractable, n_pages = False, 0  # Default

        try:
            is_extractable, n_pages = extract_text_from_pdf(file_path=file_path,
                                                            txt_filename=filename.lower().replace('.pdf', ''),
                                                            txt_folder=txt_folder_template.format(method=method),
                                                            method=method)
        except PDFSyntaxError:
            logging.error(f"PDF {file_path} could not be processed", exc_info=True)
        except Exception as e:
            logging.error(f"PDF {file_path} could not be processed", exc_info=True)

        f.write(f"{filename},{is_extractable},{n_pages}\n")

        if not i%100:
            f.flush()

    f.close()


def load_documents_to_database(file_path, method, table, schema, how='append'):
    """
    Load csv file with text characteristics into DB.
    The csv is expected to be the output of text extraction, with the columns:
        filename, is_extractable, number_of_pages

    Parameters
    ----------
    file_path: string
        Complete path to csv file to load.
    method: string
        Method used to extract the text: 'pdfminer', 'pypdf', 'tika'.
    table: string
        Name of the target table to upload data.
    schema: string
        Name of the schema where the target table is.
    how: string
        In case the table already exists, what should happen: 'fail', 'replace', 'append' (default).
    """

    con = connect_to_database()

    documents_df = pd.read_csv(file_path, delimiter=',')
    documents_df['method'] = method

    load_df = join_filename_with_id(df=documents_df)

    load_pandas_to_db(df=load_df, table=table, schema=schema, how=how)


def join_filename_with_id(df, filename_col='filename'):
    """
    Get corresponding id_llamado's to filenames.

    Parameters
    ----------
    df: pandas DataFrame
        DataFrame with (at least) a filename column.
    filename_col: string
        Name of the filename column. Default is 'filename'.

    Returns
    -------
    Same input DataFrame with the id column added.
    """

    df = df.drop_duplicates()

    docs_list = "', '".join(df[filename_col].tolist())

    query = f"select id_llamado, nombre_archivo as filename" \
        f" from raw.pbc_adenda" \
        f" where nombre_archivo in ('{docs_list}')"
    relation_df = pd.read_sql_query(query, con)

    return df.merge(relation_df, on='filename')


def extract_text_from_pbc_documents(method):
    """
    Extract text from all PBC documents that have not been check.
    First, we need to calculate the list of files that still need to be checked.
    Note: This function assumes that the paths of all PBC folders are defined:
        - first_pbc_path: Folder with first version of PBC
        - all_pbc_path: Folder with a first set of PBCs
        - all_pbc_path2: Folder with a second set of PBCs

    Parameters
    ----------
    method: String
        Method used to extract the text: 'pdfminer', 'pypdf', 'tika'.
    """

    output_file = output_template.format(date=dt.date.today(), method=method)

    logging.info('Listing files for text extraction!\n')
    docs_list = list_documents_to_extract(output_file)

    logging.info('Extracting texts!\n')
    extract_text_from_pdfs_list(docs_list=docs_list, output_file=output_file, method=method)

    logging.info('Loading results to DB!\n')
    load_documents_to_database(file_path=output_file, method=method, table='documents', schema='semantic')


if __name__ == '__main__':
    # python src/documents/extract_information.py extract_text_from_pbc_documents --method='tika
    fire.Fire()
