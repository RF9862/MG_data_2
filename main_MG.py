# Description:
# This script is an object-oriented approach to table extraction from scaned PDFs or image.

import os
import shutil
import datetime
from MG_parser import Document
from pdfrw import PdfReader, PdfWriter
import fitz                    
from MG_post_proc import post_processing

def clear_contents(dir_path):
    '''
    Deletes the contents of the given filepath. Useful for testing runs.
    '''
    filelist = os.listdir(dir_path)
    if filelist:

        for f in filelist:
            if os.path.isdir(os.path.join(dir_path, f)):
                shutil.rmtree(os.path.join(dir_path, f))
            else:
                os.remove(os.path.join(dir_path, f))
    return None

def parse_document(data_dir, output_dir, doc_path):
    '''
    This is a separate function to facilitate parallelization.
    Returns a dictionary in case of an error, else None.
    '''
    
    pdf_doc = Document(doc_path, data_dir, output_dir)

    return pdf_doc.parse_doc() 

def makedir(dir):
    try:
        os.mkdir(dir)
    except:
        pass
def txtFileSave(path, values):
    with open(path, 'w') as output:
        for row in values:
            output.write(str(row) + '\n')

def mainMG(main_self, data_dir, output_dir, err_dir):
    '''
    Main control flow:
        1. Checks if required folders exist; if not, creates them
        2. Loops over each PDF file in data_path and calls parse_doc().
        3. Output xlsx files are written to output_path.
    '''
    # Check if organizing folders exist
    main_self.total2 = 100
    main_self.cnt2 = 0    
    for i in [data_dir, output_dir]:
        try:
            if i == data_dir and not os.path.exists(data_dir):
                raise Exception("Data folder is missing or not assigned.")
            else:
                os.mkdir(i)
        except FileExistsError:
            continue

    succ_dir = output_dir + "/Success"
    # err_dir = output_dir + "/failed"

    attachErrDir = os.path.join(err_dir, 'attachErrDir')
    damageDir = os.path.join(err_dir, 'Damaged')
    # Clear output folder
    clear_contents(output_dir)
    # clear_contents(err_dir)
    makedir(os.path.join(output_dir, 'temp'))
    makedir(os.path.join(output_dir, 'check'))
    makedir(succ_dir)
    makedir(err_dir)
    makedir(attachErrDir)
    makedir(damageDir)

    # Get list of pdfs to parse
    pdf_list = [f for f in os.listdir(data_dir) if (f.split('.')[-1].lower() in ['pdf'])]
    pdf_list.sort()
    print(f"{len(pdf_list)} file(s) detected.")
    Main_prop, Attatch_info, succFiles = [], [], []
    start = datetime.datetime.now()
    # Loop over PDF files, create Document objects, call Document.parse()
    cnt = 0
    for i in pdf_list:
        cnt = cnt + 1
        print(f"Parsing file_{cnt}/{len(pdf_list)}: {os.path.join(data_dir, i)}")
        pdf_doc = Document(i, data_dir, output_dir)
        main_prop, attatch_info, checking = pdf_doc.parse_doc()
        main_self.cnt2 = int(main_self.total2/len(pdf_list) * (cnt)) 
        main_self.single_done2.emit()            

        pdf_full_name = os.path.join(data_dir, i)
        if checking == 0:
            # failFiles.append(i)
            shutil.copyfile(pdf_full_name, os.path.join(damageDir,i))
        elif checking == 1:
            shutil.copyfile(pdf_full_name, os.path.join(attachErrDir,i))
        else:
            succFiles.append(i)
            
            shutil.copyfile(pdf_full_name, os.path.join(succ_dir, i))
            Main_prop.append(main_prop)
            Attatch_info.append(attatch_info) 

    save_path = os.path.join(output_dir, 'main.xlsx')
    post_processing(Main_prop, Attatch_info, save_path, succFiles)       

    try:
        shutil.rmtree("results/temp")
    except:
        pass
    duration = datetime.datetime.now() - start
    # print(f"Success: {len(success_pdfs)}, Failed: {len(pdf_list)-len(success_pdfs)}")
    print(f"Time taken: {duration}")

    return None
# if __name__ == "__main__":

#     # Key paths and parameters
#     DATA_DIR = "inputs"
#     OUTPUT_DIR = "results"
#     # Initialize logger
#     if os.path.exists('parse_table.log'):
#         os.remove('parse_table.log')

#     # Run main control flow    
#     mainMG(DATA_DIR, OUTPUT_DIR)