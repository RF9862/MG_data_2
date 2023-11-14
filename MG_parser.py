# Description:
# This script defines the Document and Page classes to streamline the flow of information through the script.
import os, sys, traceback
import statistics
import numpy as np
from io import BytesIO
import cv2
import fitz
import pytesseract
from pytesseract import Output
import re 
from PIL import Image
from bs4 import BeautifulSoup
import pikepdf
from xfaTools import XfaObj
import datetime
from cascade_detection_mmdet import cascade_mmdet
import easyocr
reader = easyocr.Reader(['en'], detector=True, recognizer=True, model_storage_directory="model/") # this needs to run only once to load the model into memory Using CPU. Note: This module is much faster with a GPU.

# Image.MAX_IMAGE_PIXELS = 1000000000 
# Record
# Global variables
medi_val = [40, 20]
index = []
head_cols = []
digit_zoom = 1
page_hor_ths = 70 
# page_digit = False
# class
class Document:
    def __init__(self, img_name, doc_dir, output_dir):
        # Initialize key attributes and filepaths
        self.img_name = img_name
        self.doc_name = '.'.join(img_name.split('.')[:-1])
        if self.doc_name == '':
            self.doc_name = img_name
        self.doc_dir = doc_dir
        self.output_dir = output_dir
        self.pages = []
        self.temp_dir = os.path.join(output_dir, "temp")
        self.digit_doc = None
        global digit_zoom
        digit_zoom = 1
        self.head_check = False
    def pil_to_cv2(self, image):
        open_cv_image = np.array(image)
        return open_cv_image[:, :, ::-1].copy()
    def rotationTransform(self, rot, rect):
        if rot == 0:
            x0, x1, y0, y1 = rect[0], rect[2], rect[1], rect[3]
        else:
        # elif rot == 90:
            x0, x1, y0, y1 = rect[1], rect[3], rect[0], rect[2]
        return x0, x1, y0, y1
                       
 
    def open_file(self,filename, password, show=False, pdf=True):
        """Open and authenticate a document."""
        doc = fitz.open(filename)
        if not doc.is_pdf and pdf is True:
            sys.exit("this command supports PDF files only")
        rc = -1
        if not doc.needs_pass:
            return doc
        if password:
            rc = doc.authenticate(password)
            if not rc:
                sys.exit("authentication unsuccessful")
            if show is True:
                print("authenticated as %s" % "owner" if rc > 2 else "user")
        else:
            sys.exit("'%s' requires a password" % doc.name)
        return doc
     
    def embedded_get(self, doc, name, password=None, output=None):
        """Retrieve contents of an embedded file."""
        # doc = self.open_file(input, password, pdf=True)
        try:
            stream = doc.embfile_get(name)
            d = doc.embfile_info(name)
        except ValueError:
            sys.exit("no such embedded file '%s'" % name)
        filename = output if output else d["filename"]
        output = open(filename, "wb")
        output.write(stream)
        output.close()
        print("saved entry '%s' as '%s'" % (name, filename))
    
    def split_pages(self):
        '''
        1. Splits the input pdf into pages
        2. Writes a temporary image for each page to a byte buffer
        3. Loads the image as a numpy array using cv2.imread()
        4. Appends the page image/array to self.pages
        please consider following url for checking embeded file in pdf 
        https://pymupdf.readthedocs.io/en/latest/document.html#Document.embfile_names

        Notes:
        PyMuPDF's get_pixmap() has a default output of 96dpi, while the desired
        resolution is 300dpi, hence the zoom factor of 300/96 = 3.125 ~ 3.
        https://stackoverflow.com/questions/52448560/how-to-fill-pdf-forms-using-python

        '''

        if (self.img_name.split('.')[-1]).lower() == 'pdf':  
            print("Splitting PDF into pages")
            pdf_full_name = os.path.join(self.doc_dir, self.doc_name + ".pdf")
            # doc = fitz.open(stream=mem_area, filetype="pdf")
            # with fitz.open(pdf_full_name) as doc:
            self.digit_doc = fitz.open(pdf_full_name)
            ### get embedded file ###
            nm = self.digit_doc.embfile_names()
            if len(nm) == 1:
                share_f = [[nm[0], self.digit_doc.embfile_info(nm[0])['filename']]]
            else:
                share_f = [[v, self.digit_doc.embfile_info(v)['filename']] for v in nm \
                        if 'share' in self.digit_doc.embfile_info(v)['filename'].lower() or \
                            'shre' in self.digit_doc.embfile_info(v)['filename'].lower()]
            if len(share_f) == 0 or len(self.digit_doc) == 0:
                return "01"
            else:
                self.embedded_get(self.digit_doc, share_f[0][0], output=os.path.join(self.temp_dir, share_f[0][1]))
                return share_f[0]
    def pdf2img(self,doc):
        self.att_pages = []
        for i in range(len(doc)):
            # Load page and get pixmap
            zoom_factor = 3
            page = doc.load_page(i)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom_factor, zoom_factor))
            # Initialize bytes buffer and write PNG image to buffer
            buffer = BytesIO()
            buffer.write(pixmap.tobytes())
            buffer.seek(0)
            # Load image from buffer as array, append to self.pages, close buffer
            img_array = np.asarray(bytearray(buffer.read()), dtype=np.uint8)
            page_img = cv2.imdecode(img_array, 1)
            self.att_pages.append(page_img)
            buffer.close()
        return None
    def border_set(self, img_, coor, tk, color):
        '''
        coor: [x0, x1, y0, y1] - this denotes border locations.
        tk: border thickness, color: border color.
        '''
        img = img_.copy()
        if coor[0] != None:
            img[:, coor[0]:coor[0]+tk] = color # left vertical
        if coor[1] != None:
            img[:, coor[1]-tk:coor[1]] = color # right vertical
        if coor[2] != None:                    
            img[coor[2]:coor[2]+tk,:] = color # up horizontal
        if coor[3] != None:
            img[coor[3]-tk:coor[3],:] = color # down horizontal          

        return img  

    def getting_textdata(self, img, conf, zoom_fac=1):
        '''
        img: soucr image to process.
        conf: tesseract conf (--psm xx)
        '''
        d = pytesseract.image_to_data(img, output_type=Output.DICT, config=conf)
        text_ori = d['text']
        left_coor, top_coor, wid, hei, conf = d['left'], d['top'], d['width'], d['height'], d['conf']        
        ### removing None element from text ###
        text, left, top, w, h, accu, xc, yc= [], [], [], [], [], [], [], []
        for cnt, te in enumerate(text_ori):
            if te.strip() != '' and wid[cnt] > 10 and hei[cnt] > 10:
                text.append(te)
                left.append(int(left_coor[cnt]/zoom_fac))
                top.append(int(top_coor[cnt]/zoom_fac))
                w.append(int(wid[cnt]/zoom_fac))
                h.append(int(hei[cnt]/zoom_fac))
                accu.append(conf[cnt])    
                xc.append(int(left_coor[cnt]+wid[cnt]/2/zoom_fac))
                yc.append(int((top_coor[cnt]+hei[cnt]/2)/zoom_fac))
        return text, left, top, w, h, accu, xc, yc
    def text_region(self, read_img, temp_img):
        '''
        read_img: main_image, temp_img: binary image
        This function removes points and lines noises, then gets exact text range.
        1. Set 4 node(node_size=6) of temp_img into 255
        2. Get only text regions in temp_img. (condition: h < 40 and w > self.tk and h > 8), save the image as temp
        3. Noise remove
        4. Get range including all texts from read_img

        '''
        img_h, img_w = temp_img.shape

        temp_img = self.border_set(temp_img, [0, img_w, 0, img_h], 1, 255) 
        # # Set 4 node(node_size=6) of temp_img into 255
        # node_noise = 6
        # temp_img[0:node_noise,0:node_noise] = 255
        # temp_img[0:node_noise,img_w-node_noise:img_w] = 255
        # temp_img[img_h-node_noise:img_h,img_w-node_noise:img_w] = 255
        # temp_img[img_h-node_noise:img_h,0:node_noise] = 255
        # # Get only text regions in temp_img. (condition: h < 40 and w > self.tk and h > 8)
        # # temp_img = self.border_set(temp_img, [0, img_w, 0, img_h], 1, 255)   
        cnt, _ = cv2.findContours(temp_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        temp = np.zeros_like(temp_img)+255
        for c in cnt:
            x, y, w, h = cv2.boundingRect(c)
            if h < min(temp_img.shape[0]*0.9, 35) and w > 4 and h > 10:# and w < 60:# and h >15:
                # cv2.rectangle(xx, (x, y), (x + w, y + h), (0, 255, 0),1)   
                temp[y:y+h-1, x:x+w-1] = 0
        
        
        def xyRegion(temp):
            # Get range including all texts from read_img          
            kernel_hor = cv2.getStructuringElement(cv2.MORPH_RECT, (img_w, 1)) # vertical
            kernel_ver = cv2.getStructuringElement(cv2.MORPH_RECT, (1, img_h)) # vertical
            hor_temp = cv2.erode(temp, kernel_hor, iterations=2)     
            ver_temp = cv2.erode(temp, kernel_ver, iterations=2)
            img_vh = cv2.addWeighted(ver_temp, 0.5, hor_temp, 0.5, 0.0)
            _, img_vh = cv2.threshold(img_vh, 50, 255, cv2.THRESH_BINARY)
            img_vh = self.border_set(img_vh, [0, img_w, 0, img_h], 2, 255)
            contours, _ = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            x1, x2, y1, y2  = img_w, 0, img_h, 0
            for c in contours:
                x, y, w, h = cv2.boundingRect(c) 
                if w < img_w and h < img_h:
                    if x < x1: x1 = x
                    if y < y1: y1 = y
                    if x+w > x2: x2 = x+w
                    if y+h > y2: y2 = y+h
            return x1,x2,y1,y2    

        x01,x02,y01,y02 = xyRegion(temp)            
        erod_size = 10
        temp = cv2.erode(temp, np.ones((2,erod_size)), iterations=1) # 10 means letter space.
        temp = self.border_set(temp, [0, img_w, 0, img_h], 1, 255) 
        
        # noise remove     
        w_30 = False
        cnt, _ = cv2.findContours(temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        ch_w = 15
        for c in cnt:
            x, y, w, h = cv2.boundingRect(c)
            if w > ch_w:
                w_30 = True 
                break
        if w_30:
            for c in cnt:
                x, y, w, h = cv2.boundingRect(c)
                if w < ch_w or h < 15: temp[y:y+h, x:x+w] = 255            

        # img_bin_ver = cv2.erode(img_bin, np.ones((1,erod_size)), iterations=1)

        x1,x2,y1,y2 = xyRegion(temp)

        # ad = 0
        # x1, y1 = max(0, x1-ad), max(0, y1-ad)
        # x2, y2 = min(img_w, x2+ad), min(img_h, y2+ad)
        if x1 > 2: x1 = x1 + int(erod_size/2)
        if x2 < img_w -2: x2 = x2 - int(erod_size/2)
        x1, x2 = max(x1, x01), min(x2, x02)
        y1, y2 = max(y1, y01), min(y2, y02)

        # img = np.zeros_like(read_img) + 255
        # img_bin = np.zeros_like(temp_img) + 255
        img = read_img[y1:y2, x1:x2]
        # temp_img = temp_img[y1:y2, x1:x2]
        # contours, _ = cv2.findContours(temp_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # for c in contours:
        #     x,y,w,h = cv2.boundingRect(c)
        #     if w<4 and h<4:     
        #         print(x,y,w,h)
        #         img[y:y+h, x:x+w] = [255, 255, 255]
        pad = 10
        try: img = np.pad(img, ((pad, pad), (0,0)),mode='constant', constant_values=255) 
        except: img = np.pad(img, ((pad, pad), (pad,pad), (0,0)),mode='constant', constant_values=255) 

        # img_bin = temp_img[y1:y2, x1:x2]
        # erod_size = 5
        # img_bin_ver = cv2.erode(img_bin, np.ones((1,erod_size)), iterations=1)
        # img_bin_hor = cv2.erode(img_bin, np.ones((erod_size, 1)), iterations=1)

        # ### LAST noise removal ###
        # contours, _ = cv2.findContours(img_bin_ver, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # for c in contours:
        #     x,y,w,h = cv2.boundingRect(c)
        #     # if (h > 40 and w < 20) or (h < 8 and w > 40):
        #     # if w < 5 and (x < 5 or x > img_w-5):
        #     if (w < 5 and (x < 5 or x > img_w-5)) or (h > 40 and w < 20):
        #         # print(w,h)
        #         img[y:y+h, x:x+w] = [255,255,255]
        
        return img        

    def box_detection(self, img_vh):

        '''
        Here gets boxes and texts
        Boxes and texts are corresponding each other
        '''
        img_height, img_width = img_vh.shape[0:2]
        image = self.MainImgTemp.copy()

        # image = cv2.cvtColor(self.MainImg, cv2.COLOR_GRAY2RGB)
        box= []
        contours, _ = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # temp = cv2.cvtColor(self.img_removedByline, cv2.COLOR_RGB2GRAY)
        # _, temp_img = cv2.threshold(temp, 230, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if  15<w < img_width*0.8 and h>15:
                image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                box.append([y, x, h, w])
        box.sort()
        box = np.array(box)
        rows = np.unique(box[:,0]) # values of y all box((y1,x1,h1,w1), (y2,x2,h2,w2),...)
        rows_len = len(rows)
        cols_len = len(box)//rows_len
        box = box.reshape((rows_len, cols_len, 4))

        return box
    def box_text_detection(self, boxes, new_bin_img, tableNo):
        '''
        Here gets boxes and texts
        Boxes and texts are corresponding each other
        '''
        # img_height, img_width = img_vh.shape[0:2]
        image = self.MainImgTemp.copy()

        # image = cv2.cvtColor(self.MainImg, cv2.COLOR_GRAY2RGB)
        box, text = [], []
        # contours, _ = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        custom_config = '--psm 6'
        # # temp = cv2.cvtColor(self.img_removedByline, cv2.COLOR_RGB2GRAY)
        # # _, temp_img = cv2.threshold(temp, 230, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # cnt = 0   

        # for c in contours:
        #     x, y, w, h = cv2.boundingRect(c)
        #     if  15<w < img_width*0.8 and h>15:

        for box in boxes:
            rowText = []
            for bo in box:
                y, x, h, w = bo
                image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                # In case of scanned pdf
                # if (new_bin_img[y+5:y+int(h*0.8), x+int(w*0.05):x+int(w*0.8)] == 0).sum() > 10:
                temp_read = new_bin_img[y:y+h+3, x:x+w]
                img_to_read = self.MainImgTemp[y:y+h+3, x:x+w]
                # cv2.imwrite(f"results/img_to_read_{cnt}.png", img_to_read)  
                if len(np.unique(temp_read)) == 1: 
                    te = ''
                    rowText.append(te)
                    continue
                
                img_to_read = self.text_region(img_to_read,  temp_read)
                if len(np.unique(img_to_read)) > 1:
                    
                    te = pytesseract.image_to_string(img_to_read, config=custom_config)
                    if(len(te) == 0):
                        te = pytesseract.image_to_string(img_to_read, config='--psm 10')
                    ## Modification of text ##   
                    strp_chars = "|^#;$`'-_=*\/‘:¢ \n"
                    te = re.sub('\n+', '\n', te)
                    te = te.replace(':', '')
                    te = te.replace('*', '')
                    # te = re.sub('(:|*|#)', '', te)
                    te = te.replace('\n|\n ', ' ')
                    checkwords, repwords =('{', '}', '!'), ('(', ')', 'I')
                    for check, rep in zip(checkwords, repwords):
                        te = te.replace(check, rep)
                    te = te.strip(strp_chars)
                    while 1:
                        if (te[0:2] in ["l ", "i ", "l\n", "i\n", "| ", "|\n"]):
                            te = te[2:]
                        elif (te[-2:] in [" l", " i", "\nl", "\ni", " |", "\n|"]):
                            te = te[0:-2]
                        else: break
                else:
                    te = ''
                rowText.append(te)
            text.append(rowText)
        if tableNo == 0:
            cv2.imwrite(os.path.join(self.output_dir,'check', self.doc_name+'.jpg'), image)
        else:
            cv2.imwrite(os.path.join(self.output_dir,'check', self.doc_name+'_'+str(tableNo)+'.jpg'), image)
        ## post processing (including reshape) ##
        text = np.array(text, dtype=str)

        return text
    def line_remove(self, image):
        result = image.copy()
        try:
            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        except: 
            thresh = 255 - image
        # Remove horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
        remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            try: cv2.drawContours(result, [c], -1, (255,255,255), 5) # for rgb image.
            except: cv2.drawContours(result, [c], -1, 255, 5) # for binary image

        # Remove vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,35))
        remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            try: cv2.drawContours(result, [c], -1, (255,255,255), 5) # for rgb image.
            except: cv2.drawContours(result, [c], -1, 255, 5) # for binary image

        return result   

    def subset(self, set, lim, loc):
        '''
        set: one or multi list or array, lim: size, loc:location(small, medi, large)
        This function reconstructs set according to size of lim in location of loc.
        '''
        cnt, len_set = 0, len(set)        
        v_coor_y1, index_ = [], []
        pop = []
        for i in range(len_set):
            if i < len_set-1:
                try:
                    condition = int(set[i+1][0]) - int(set[i][0])
                except:
                    condition = int(set[i+1]) - int(set[i])
                if condition < lim:
                    cnt = cnt + 1
                    pop.append(set[i])
                else:
                    cnt = cnt + 1
                    pop.append(set[i])
                    pop = np.asarray(pop)
                    try:
                        if loc == "small": v_coor_y1.append([min(pop[:, 0]), min(pop[:, 1]), max(pop[:, 2])])
                        elif loc == "medi": v_coor_y1.append([int(np.median(pop[:, 0])), min(pop[:, 1]), max(pop[:, 2])])
                        else: v_coor_y1.append([max(pop[:, 0]), min(pop[:, 1]), max(pop[:, 2])])
                    except:
                        if loc == "small": v_coor_y1.append(min(pop))
                        elif loc == "medi": v_coor_y1.append(int(np.median(pop)))
                        else: v_coor_y1.append(max(pop))  
                    index_.append(cnt)
                    cnt = 0
                    pop = []
            else:
                cnt += 1
                pop.append(set[i])
                pop = np.asarray(pop)
                try:
                    if loc == "small": v_coor_y1.append([min(pop[:, 0]), min(pop[:, 1]), max(pop[:, 2])])
                    elif loc == "medi": v_coor_y1.append([int(np.median(pop[:, 0])), min(pop[:, 1]), max(pop[:, 2])])
                    else: v_coor_y1.append([max(pop[:, 0]), min(pop[:, 1]), max(pop[:, 2])])
                except:
                    if loc == "small": v_coor_y1.append(min(pop))
                    elif loc == "medi": v_coor_y1.append(int(np.median(pop)))
                    else: v_coor_y1.append(max(pop))                    
                index_.append(cnt)

        return v_coor_y1, index_            
    def line_detector(self,image, prop):
        # Convert color image to grayscale
        # img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # # Binarize image using thresholding
        # _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        img = image.copy()
        dilate_size = 50
        values = []
        if prop == 'hor':
            # ind = 0
            erode_size = 6
            addi = (int(dilate_size-erode_size)/2)
            img = cv2.erode(img, np.ones((1,erode_size)), iterations=1)
            img = cv2.dilate(img, np.ones((1,dilate_size)), iterations=1)    
            img = self.border_set(img, [0,img.shape[1], 0, img.shape[0]], 1, 255)        
            cnt, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
            W = []
            for c in cnt:
                x, y, w, h = cv2.boundingRect(c) 
                if w > 150 and h < 25:
                    W.append(w)
                    values.append([int(y+h/2), max(0, x-addi), x+w+addi]) 
            try:
                W, values = zip(*sorted(zip(W, values)))
                wUniq, wCnt = self.subset(W, 30, 'medi')
                wCnt = [sum([wCnt[j] for j in range(i, len(wCnt))]) for i in range(len(wCnt))]
                wUniq, wCnt = zip(*sorted(zip(wUniq, wCnt), reverse=True))
                for i, wU in enumerate(wUniq):
                    if wCnt[i] > 2 and wU > 50:
                        k = 0.4 if wU > 150 else 0.7
                        values = [v for j, v in enumerate(values) if W[j] >= wU*k]
                        values.sort()
                        return values
            except: pass
                    

            # for c in cnt:
            #     x, y, w, h = cv2.boundingRect(c) 
            #     if w > 250 and h < 12:
            #         values.append([int(y+h/2), max(0, x-addi), x+w+addi]) 
            # if len(values) <4:
            #     values = []
            #     for c in cnt:
            #         x, y, w, h = cv2.boundingRect(c) 
            #         if h > 150 and w < 20:
            #             values.append([int(y+h/2), max(0, x-addi), x+w+addi])                                  
        else: 
            # ind = 1
            erode_size = 5
            addi = (int(dilate_size-erode_size)/2)            
            img = cv2.erode(img, np.ones((erode_size, 1)), iterations=1)
            img = cv2.dilate(img, np.ones((dilate_size, 1)), iterations=1)  
            img = self.border_set(img, [0,img.shape[1], 0, img.shape[0]], 1, 255)
            cnt, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            H = []
            xx = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            values = []
            for c in cnt:
                x, y, w, h = cv2.boundingRect(c) 
                if h > 50 and w < 30:
                    H.append(h)
                    values.append([int(x+w/2), max(0, y-addi), y+h+addi])
                    cv2.rectangle(xx, (x, y), (x + w, y + h), (0, 255, 0),2)   
            try:
                H, values = zip(*sorted(zip(H, values)))
                hUniq, hCnt = self.subset(H, 50, 'medi')
                hCnt = [sum([hCnt[j] for j in range(i, len(hCnt))]) for i in range(len(hCnt))]
                hUniq, hCnt = zip(*sorted(zip(hUniq, hCnt), reverse=True))
                for i, hU in enumerate(hUniq):
                    if hCnt[i] > 3 and hU > 50:
                        k = 0.5 if hU > 200 else 0.7
                        values = [v for j, v in enumerate(values) if H[j] >= hU*k]
                        values.sort()
                        return values
            except: pass

        return []      
    def getting_table(self):
        '''
        1. Remove unnecessary columns
        2. Get all horizontal lines
        3. Get upper limit and under limit of image,  and self.rows
        4. Make self.table image using self.cols and self.rows
        '''
        # try:
        #     gray = cv2.cvtColor(self.MainImg, cv2.COLOR_RGB2GRAY)
        #     _, img = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # except: img = self.MainImg.copy()
        img = self.imgTemp.copy()
        # try:
        #     gray = cv2.cvtColor(self.MainImg, cv2.COLOR_RGB2GRAY)
        #     _, new_bin_img = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # except: pass
        img = self.border_set(img, [0, img.shape[1], 0, img.shape[0]], 2, 255) 
        cnt, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnt:
            x, y, w, h = cv2.boundingRect(c) 
            if w > img.shape[1]*0.4 and h > 15 and w < 80 or w<80 and h > img.shape[0]*0.3 and w >15:
                img[y:y+h,x:x+w] = 255
        # new_bin_img = self.border_set(new_bin_img, [0, new_bin_img.shape[1], None, None], 50, 255)     
        rows = self.line_detector(img, 'hor')
        cols = self.line_detector(img, 'ver')
        rows, cols = np.array(rows), np.array(cols)
        try: con = len(self.subset(np.sort(cols[:, 0]), 20, 'medi')[0]) < 4
        except: con = True
        if con or len(rows) < 3:     
            print("NoBorder!!!")   

            for c in cnt:
                x, y, w, h = cv2.boundingRect(c) 
                if h < 10:
                    img[y:y+h,x:x+w] = 255

            img = self.line_remove(img)
            er_hor_size = 18
            consider_img = cv2.erode(img, np.ones((1,er_hor_size)), iterations=1)
            ver_bin = cv2.erode(consider_img, np.ones((img.shape[0], 1)), iterations=2)
            consider_img = cv2.erode(img, np.ones((5, 1)), iterations=1)
            hor_bin = cv2.erode(consider_img, np.ones((1, img.shape[1])), iterations=2)
            img_vh = cv2.addWeighted(hor_bin, 0.5, ver_bin, 0.5, 0.0)
            _, img_vh = cv2.threshold(img_vh, 50, 255, cv2.THRESH_BINARY)
            img_vh = self.border_set(img_vh, [0, img_vh.shape[1], 0, img_vh.shape[0]], 1, 255)              
        else:
            print("Border!!!")
            # ver_coor_y1, index1 = self.subset(np.sort(cols[:, 1]), 20, 'medi')
            # ver_coor_y2, index2 = self.subset(np.sort(cols[:, 2]), 20, 'medi')
            # hor_coor_y1, hor_ind1 = self.subset(np.sort(rows[:, 1]), 20, 'medi')
            # hor_coor_y2, hor_ind2 = self.subset(np.sort(rows[:, 2]), 20, 'medi')  
            cols, _ = self.subset(np.sort(cols[:, 0]), 20, 'medi')          
            rows, _ = self.subset(np.sort(rows[:, 0]), 20, 'medi')  
            # min_y = ver_coor_y1[index1.index(max(index1))]-2
            
            # min_y = min([ver_coor_y1[i] for i, x in enumerate(index1) if x == max(index1)]) - 2
            # max_y = max([ver_coor_y2[i] for i, x in enumerate(index2) if x == max(index2)]) + 2
            # min_x = min([hor_coor_y1[i] for i, x in enumerate(hor_ind1) if x == max(hor_ind1)]) - 2
            # max_x = max([hor_coor_y2[i] for i, x in enumerate(hor_ind2) if x == max(hor_ind2)]) + 2
            
            # new_img = img[max(0, min_y):max_y+3, min_x:max_x]
            # new_bin_img = img[max(min_y,0):max_y+3, min_x:max_x]

            vertical_lines = np.zeros_like(img)
            horizontal_lines = np.zeros_like(img)
            # Make self.table image using self.cols and self.rows
            # rows = [v for v in rows if v[1]-v[0] > 15]
            for col in cols:
                vertical_lines[:, col-3:col+3] = 255
            for row in rows:
                if row >= 0:
                    horizontal_lines[max(0, row-3):row+3] = 255
            img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
            _, img_vh = cv2.threshold(img_vh, 50, 255, cv2.THRESH_BINARY)

            # if min_y < 100: min_y = 0
            # img_vh = img_vh[min_y:max_y, min_x:max_x]
            # self.MainImgTemp = self.MainImgTemp[min_y:max_y, min_x:max_x]
            # img = img[min_y:max_y, min_x:max_x]


            # self.img_removedByline = new_bin_img[min_y:max_y, min_x:max_x]
            # img_vh[:, 0:max(0, min_x)] = 0
            # img_vh[:, max_x:] = 0
            
            img_vh = self.border_set(img_vh, [0, img_vh.shape[1], 0, img_vh.shape[0]], 1, 255)


        return img_vh, img
        
    def lines_extraction(self, img, direct, setting, alpa):
        '''
        1. Convert source image into inv grayscal.
        2. Strenthen color contrast of image.
        3. Considering broken lines or faint images, run cv2.filter2D()
        4. Get all lines
        5. Extract lines satisfied some conditions 
        '''
        ################################
        
        # Convert source image into inv grayscal.
        img_gry = 255 - img
        # Strenthen color contrast of image.
        # try:
        #     if page_digit: img_gry = (img_gry>70)*(np.zeros_like(img_gry)+255) + (img_gry<=70)*np.zeros_like(img_gry)
        #     else: img_gry = (img_gry>50)*(np.zeros_like(img_gry)+255) + (img_gry<=50)*np.zeros_like(img_gry)
        # except:
        #     img_gry = (img_gry>50)*(np.zeros_like(img_gry)+255) + (img_gry<=50)*np.zeros_like(img_gry)
        # if direct == "ver": kernel = np.ones((alpa,2),np.float32)/2*alpa           
        # else: kernel = np.ones((1,2),np.float32)/2
        # # Considering broken lines or faint images, run cv2.filter2D()
        # new = cv2.filter2D(img_gry,-1,kernel)
        # th, bin_img = cv2.threshold(img_gry, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # img_cny = cv2.Canny(img_gry, 50, 200)
        #######################################################
        # Get all lines
        fld = cv2.ximgproc.createFastLineDetector()
        lns = fld.detect(img)
        # xxx = fld.drawSegments(img.copy(), lns) # for drawing
        img_cpy = img.copy()
        lim, ver_con, hor_con = 8, setting[0], setting[1]
        lines = []
        # Extract lines satisfied some conditions 
        if lns is None: pass
        else:
            if direct == "ver": 
                for ln in lns:
                    x1, y1, x2, y2 = int(ln[0][0]), int(ln[0][1]), int(ln[0][2]), int(ln[0][3])                
                    if abs(x1-x2) < lim and abs(y1-y2) > ver_con:
                        lines.append([int(x1/2+x2/2), min(y1, y2), max(y1, y2)])
                        # cv2.line(img_cpy, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=5)  
                lines.sort()
                lines, _ = self.subset(lines, 10, 'medi')                    
            elif direct == "hor":
                for ln in lns:
                    x1, y1, x2, y2 = int(ln[0][0]), int(ln[0][1]), int(ln[0][2]), int(ln[0][3])                   
                    if abs(y1-y2) < lim and abs(x1-x2) > hor_con:
                        lines.append([int(y1/2+y2/2), min(x1, x2), max(x1, x2)])
                        # cv2.line(img_cpy, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=5)      
                lines.sort()
                lines, _ = self.subset(lines, 10, 'medi')              
            elif direct == "angle_ver":
                # This part is used for deskew of image.
                def angle_ver(lns, ver_con):   
                    lines = [] 
                    angle_lim = 0.0873 # Rad of . if use Deg, lim = 5 Deg. 
                    for ln in lns:
                        x1, y1, x2, y2 = int(ln[0][0]), int(ln[0][1]), int(ln[0][2]), int(ln[0][3])                     
                        if abs(y1-y2) > ver_con:
                            lim = abs(y1-y2) * np.tan(angle_lim)    
                            if abs(x1-x2) < lim:
                                if y1 > y2: ang = np.arctan((x1-x2)/(y1-y2))*180/np.pi
                                else: ang = np.arctan((x2-x1)/(y2-y1))*180/np.pi
                                if ang < 0: ang = 90 + ang
                                lines.append(ang)
                                cv2.line(img_cpy, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=1) 
                    return lines
                while 1:
                    lines = angle_ver(lns, ver_con)
                    ver_con = ver_con -100
                    if len(lines) > 4 or ver_con  < 80: break


                # if len(lines) < 5: 
                #     lines = angle_ver(lns, 400)
                #     if len(lines) < 5: 
                #         lines = angle_ver(lns, 200)
                #         if len(lines) < 5: lines = angle_ver(lns, 100)
                return lines
            elif direct == "angle_hor":
                self.segment_hor = []
                # This part is used for deskew of image.
                def angle_hor(lns, hor_con):   
                    lines = [] 
                    angle_lim = 0.15 # Rad of . if use Deg, lim = 5 Deg.
                    for ln in lns: 
                        x1, y1, x2, y2 = int(ln[0][0]), int(ln[0][1]), int(ln[0][2]), int(ln[0][3])                           
                        if abs(x1-x2) > hor_con:
                            lim = abs(x1-x2) * np.tan(angle_lim)    
                            if abs(y1-y2) < lim:
                                if x1 > x2: ang = np.arctan((y1-y2)/(x1-x2))*180/np.pi
                                else: ang = np.arctan((y2-y1)/(x2-x1))*180/np.pi
                                if ang < 0: ang = 90 + ang
                                lines.append(ang)     
                        # if abs(x1-x2) > 30 and abs(y1-y2) < 20:
                                if abs(x1-x2) > 300: self.segment_hor.append(abs(x1-x2))   
                    return lines
                while 1:
                    lines = angle_hor(lns, hor_con)
                    hor_con = hor_con -150       
                    if len(lines) > 4 or hor_con  < 80: break
        return lines

    def deter_angle(self,angle_list):
        angle_list1, angle_list2 = [], []
        angle = 0
        if len(angle_list) > 0:
    
            # All angles are split into two sets. 
            for ang in angle_list:
                if ang < 100 and ang > 80:
                    angle_list1.append(ang)
                elif ang < 10 and ang > -10:
                    angle_list2.append(ang)
            # Select one set with more frequently angles.
            if len(angle_list1) > len(angle_list2):
                angle_list = angle_list1
            else:
                angle_list = angle_list2
            # Get median value from angle set.
            try:
                angle = statistics.median(angle_list)
            except statistics.StatisticsError:
                angle = 0
            if angle < -45:
                angle += 90
            elif angle > 45:
                angle -= 90   
        return angle     
    def AdaptiveThreshold(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,4)                            
        return img
    def preprocess_image(self, MainImg, img):
        '''
        1. Gets all angles of horizontral lines from function lines_extraction()
        2. All angles are in range of 80deg~100deg or -10deg~10deg. All angles are split into two sets. 
        3. Select one set with more frequently angles.
        4. Find out median value of selected set.
        5. Rotate image according to the valuel.
        '''
        angle_list = self.lines_extraction(img, "angle_hor", [600, 1000], None)
        # if len(angle_list) < 5: angle_list = self.lines_extraction(self.img, "angle_hor", [50, 70], None)
        angle = self.deter_angle(angle_list)
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        # Rotate image by gotten value 
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        MainImg = cv2.warpAffine(MainImg,
                             M,
                             (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
        img = self.AdaptiveThreshold(MainImg)
        
        return MainImg, img

    def infoFromText(self, text, inds, nameAddr):
        attatch_info = [[],[],[]]
        if inds[0] != -1 and len(text[0]) > 0:
            ## initail remove work empty row
            cnt = 0
            for i, row in enumerate(text):
                if np.count_nonzero(row=='') == text.shape[1]: 
                    text = np.delete(text, i-cnt, axis=0)  
                    cnt += 1
            delK = 1
            for i in range(delK):
                text = np.delete(text, 0, axis=0) 
            
            if type(inds[0]) != list:
                # text = text[:, [v for v in inds if v > 0]]
                if inds[0] != -1:
                    text = text.tolist()
                    newText = []
                    for i in range(len(text)):
                        if text[i][inds[0]+1] != '': 
                            try:  newText.append(list(map('\n'.join, zip(*arr_list))))
                            except: pass
                            arr_list = [text[i]]
                        else: arr_list.append(text[i])
                    try: newText.append(list(map('\n'.join, zip(*arr_list))))
                    except: pass
                    text = np.array(newText)
            
            for i, ind in enumerate(inds):
                if ind != -1:
                    if i == 0 and nameAddr:
                        cc = text[:, ind]
                        try:
                            name_addr = [[cx.strip().split('\n')[0], '\n'.join(cx.strip().split('\n')[1:])] for cx in cc]
                            attatch_info[0] = list(np.array(name_addr)[:, 0])
                            attatch_info[1] = list(np.array(name_addr)[:, 1])
                        except: pass
                    elif i == 0: 
                        try: attatch_info[i] = [re.sub(r'\d+', '', v.strip()) for v in text[:,ind]]
                        except: 
                            cc = text[:, ind]
                            share_name = cc[:, 0]
                            for j in range(1, cc.shape[1]):
                                share_name = list(map('_'.join, zip(share_name, cc[:, j])))   
                                                
                            attatch_info[i] = share_name
                    elif i == 2:
                        attatch_info[i] = [v.strip().split(' ')[0] for v in text[:,ind]] # re.sub('[^0-9.,]','', v)
                    else:
                        attatch_info[i] = list(text[:,ind])
                    
        return attatch_info        
    def getVal(self, val):
        name = val.name
        val = val.string
        
        try:
            tempVal = str(float(val)).strip('0')
            tempVal = tempVal.strip('.')
            if tempVal == "": output = 0
            else:
                try: output = str(int(tempVal))
                except: output = str(float(tempVal))
            
        except: 
            if 'date' in name:
                try:
                    date_obj = datetime.datetime.strptime(val, "%Y-%m-%d")
                    output = date_obj.strftime("%d/%m/%Y")  
                except: 
                    output = "" 
            else:             
                output = val  
        
        return output 
    def mainProp(self):
        '''
        https://github.com/AF-VCD/pdf-xfa-tools
        '''
        CIN_no, share_his_pc, total_share = '-', '-', '-'
        pdf_full_name = os.path.join(self.doc_dir, self.doc_name + ".pdf")
        with pikepdf.Pdf.open(pdf_full_name) as pdfData:
            xfaDict = XfaObj(pdfData)   
            data = xfaDict['datasets'] 
        soup = BeautifulSoup(data)
        output = {}
        output["Cin"] = soup.find_all('cin')
        output["Email"] = soup.find_all('email_id_company')
        output["Phone_Number"] = soup.find_all('phone_number')
        output["PAN"] = soup.find_all('it_pan_of_compny')
        output["Turnover"] = soup.find_all('tot_turnover')
        output["Website"] = soup.find_all('website')
        output["NetWorth"] = soup.find_all('net_worth_comp')
        output["Agm_Date"] = soup.find_all('date_agm')
        output["Financial_From_Date"] = soup.find_all('fy_from_date')
        output["Financial_End_Date"] = soup.find_all('fy_end_date')
        output["Main_Act_Grp_Code"] = soup.find_all('main_act_grp_cod')#11
        output["Des_Business_Act"] = soup.find_all('des_business_act')#12
        output["Business_Act_Code"] = soup.find_all('business_act_cod')#13
        output["Percent_Turn_Ovr"] = soup.find_all('percent_turn_ovr')#14
        
        output["Holding_Company_Name"] = soup.find_all('name_company')#15
        output["Hold_Sub_Asso_Cin"] = soup.find_all('cin_fcrn')# 16
        output["HOLDING_SUBSIDIARY_ASSOCIATE"] = soup.find_all('hold_sub_assoc') #17
        output["PERCENT_SHARE"] = soup.find_all('percent_share') #18
        
        output["Total_Numer_equity_Authorized_Capital"] = soup.find_all('tot_no_es_a_cap') #19- Total_number_equity_Authorized_Capital
        output["Authorized_Capital_Nominal_Value"] = soup.find_all('no_es_a_cap') #20- e_Authorized_Capital_Nominal_Value
        output["Total_Amount_equity_Authorized_Capital"] = soup.find_all('tot_amt_es_a_cap') #21- Total_amount_equity_Authorized_Capital-
        
        output["Total_number_preference_Authorized_Capital"] = soup.find_all('tot_no_ps_a_cap') #22- Total_number_preference_Authorized_Capital
        output["preference_Authorized_Capital_Nominal_Value"] = soup.find_all('nom_val_ps_a_cap') #23- preference_Authorized_Capital_Nominal_Value
        output["Total_amount_preference_Authorized_Capital"] = soup.find_all('tot_amt_ps_a_cap') #24- Total_amount_preference_Authorized_Capital
        
        output["Total_number_equity_Issued_Capital"] = soup.find_all('tot_no_es_i_cap') #25- Total_number_equity_Issued_Capital
        output["Issued_Capital_Nominal_Value"] = soup.find_all('no_es_i_cap') #26- Issued_Capital_Nominal_Value
        output["Total_amount_equity_Issued_Capital"] = soup.find_all('tot_amt_es_i_cap') #27- Total_amount_equity_Issued_Capital
        output["Total_number_preference_Issued_Capital"] = soup.find_all('tot_no_ps_i_cap') #28- Total_number_preference_Issued_Capital
        output["preference_Issued_Capital_Nominal_Value"] = soup.find_all('nom_val_ps_i_cap') #29- preference_Issued_Capital_Nominal_Value
        
        output["Total_amount_preference_Issued_Capital"] = soup.find_all('tot_amt_ps_i_cap') #30- Total_amount_preference_Issued_Capital
        output["Total_number_equity_Subscribed_Capital"] = soup.find_all('tot_no_es_s_cap') #31- Total_number_equity_Subscribed_Capital
        output["Subscribed_Capital_Nominal_Value"] = soup.find_all('no_es_s_cap') #32- Subscribed_Capital_Nominal_Value
        output["Total_amount_equity_Subscribed_Capital"] = soup.find_all('tot_amt_es_s_cap') #33.Total_amount_equity_Subscribed_Capital
        output["Total_number_preference_Subscribed_Capital"] = soup.find_all('tot_no_ps_s_cap') #34- Total_number_preference_Subscribed_Capital
        output["preference_Subscribed_Capital_Nominal_Value"] = soup.find_all('nom_val_ps_s_cap') #35- preference_Subscribed_Capital_Nominal_Value
        output["Total_amount_preference_Subscribed_Capital"] = soup.find_all('tot_amt_ps_s_cap') #36- Total_amount_preference_Subscribed_Capital-
        
        output["Total_number_equity_Paid_Up_Capital"] = soup.find_all('tot_no_es_p_cap') #37- Total_number_equity_Paid_Up_Capital
        output["Paid_Up_Capital_Nominal_Value"] = soup.find_all('nom_val_es_p_cap') #38- Paid_Up_Capital_Nominal_Value
        output["Total_amount_equity_Paid_Up_Capital"] = soup.find_all('tot_amt_es_p_cap') #39- Total_amount_equity_Paid_Up_Capital
        output["Total_number_preference_Paid_Up_Capital"] = soup.find_all('tot_no_ps_p_cap') #40- Total_number_preference_Paid_Up_Capital
        output["preference_Paid_Up_Capital_Nominal_Value"] = soup.find_all('nom_val_ps_p_cap') #41- preference_Paid_Up_Capital_Nominal_Value
        output["Total_amount_preference_Paid_Up_Capital"] = soup.find_all('tot_amt_ps_p_cap') #42- Total_amount_preference_Paid_Up_Capital
        output["NO_COMPANIES"] = soup.find_all('no_companies') #43- NO_COMPANIES
        output["NO_BUSINESS_ACTIVITY"] = soup.find_all('no_business_act') # 44- NO_BUSINESS_ACTIVITY
        output["File_Name"] = self.img_name
        
        # direction check
        for cin in output["Cin"]: 
            try:
                if len(cin.string) > 10: # CIN length is 21
                    output["Cin"] = cin.string
                    break
            except: pass
        newOutput = {}
        direction = False
        for key, val in output.items():
            if key == 'Cin' or key == "File_Name": continue
            if len(val)>1 and val[0].string != None: 
                direction = True
                break
        for key, val in output.items():
            if key == 'Cin' or key == "File_Name": 
                newOutput[key] = val
                continue
            if direction: newOutput[key] = self.getVal(val[0])
            else: newOutput[key] = self.getVal(val[-1])
        
        result = []
        businessTwoCheck, companyTwoCheck = False, False
        if len(output["Business_Act_Code"]) > 2: businessTwoCheck = True                
        if len(output["Holding_Company_Name"]) > 2: companyTwoCheck = True   
        

        if companyTwoCheck:
            companyTwoCheck = True
            newOutputCpy = newOutput.copy()
            if direction: S, k = 0, -1
            else: S, k = len(output["Holding_Company_Name"])-1, 1        
            for i in range(0, len(output["Holding_Company_Name"])-1):
                newOutputCpy["Holding_Company_Name"] = self.getVal(output["Holding_Company_Name"][S-k*i])
                newOutputCpy["Hold_Sub_Asso_Cin"] = self.getVal(output["Hold_Sub_Asso_Cin"][S-k*i])
                newOutputCpy["HOLDING_SUBSIDIARY_ASSOCIATE"] = self.getVal(output["HOLDING_SUBSIDIARY_ASSOCIATE"][S-k*i])
                newOutputCpy["PERCENT_SHARE"] = self.getVal(output["PERCENT_SHARE"][S-k*i])
                if businessTwoCheck:
                    newOutputCpy["Main_Act_Grp_Code"] = ''
                    newOutputCpy["Des_Business_Act"] = ''
                    newOutputCpy["Business_Act_Code"] = ''
                    newOutputCpy["Percent_Turn_Ovr"] = ''
                
                result.append(newOutputCpy) 
        if businessTwoCheck:
            newOutputCpy = newOutput.copy()
            if direction: S, k = 0, -1
            else: S, k = len(output["Business_Act_Code"])-1, 1                
            for i in range(0, len(output["Business_Act_Code"])-1):
                newOutputCpy["Main_Act_Grp_Code"] = self.getVal(output["Main_Act_Grp_Code"][S-k*i])
                newOutputCpy["Des_Business_Act"] = self.getVal(output["Des_Business_Act"][S-k*i])
                newOutputCpy["Business_Act_Code"] = self.getVal(output["Business_Act_Code"][S-k*i])
                newOutputCpy["Percent_Turn_Ovr"] = self.getVal(output["Percent_Turn_Ovr"][S-k*i])
                newOutputCpy
                if companyTwoCheck:
                    newOutputCpy["Holding_Company_Name"] = ''
                    newOutputCpy["Hold_Sub_Asso_Cin"] = ''
                    newOutputCpy["HOLDING_SUBSIDIARY_ASSOCIATE"] = ''
                    newOutputCpy["PERCENT_SHARE"] = ''
                result.append(newOutputCpy)  
        if not (companyTwoCheck or  businessTwoCheck):
            result.append(newOutput)
                
        return result

        # return [CIN_no, share_his_pc, total_share]

    def attachProcessing(self):
        pdf_full_name = os.path.join(self.doc_dir, self.doc_name + ".pdf")
        self.digit_doc = fitz.open(pdf_full_name)
        ### get embedded file ###
        nm = self.digit_doc.embfile_names()
        if len(nm) == 1:
            share_f = [[nm[0], self.digit_doc.embfile_info(nm[0])['filename']]]
        else:
            share_f = [[v, self.digit_doc.embfile_info(v)['filename']] for v in nm \
                    if 'sh' in self.digit_doc.embfile_info(v)['filename'].lower() or 'los' in self.digit_doc.embfile_info(v)['filename'].lower()]# or \
                        # 'shre' in self.digit_doc.embfile_info(v)['filename'].lower() or 'list of sh' \
                        #     in self.digit_doc.embfile_info(v)['filename'].lower()]
        if len(share_f) == 0 or len(self.digit_doc) == 0:
            return "01"
        elif len(share_f) == 1:
            self.embedded_get(self.digit_doc, share_f[0][0], output=os.path.join(self.temp_dir, self.doc_name + ".pdf"))
            return self.doc_name + ".pdf"
        else:
            result = [v for v in share_f if 'share' in v[1].lower()]
            if len(result) == 0: result = share_f
            self.embedded_get(self.digit_doc, result[0][0], output=os.path.join(self.temp_dir, self.doc_name + ".pdf"))
            
            return self.doc_name + ".pdf"

    def check_scan_or_digit(self):

        '''
        Check if pdf is digital or scanned.
        '''
        
        digit = [False]*len(self.attach_doc)
        for i in range(len(self.attach_doc)):
            d = self.attach_doc[i].get_text_words()
            if len(d) > 10:# and digit_flag:
                # digit = self.get_digit(d) 
                digit[i] = True
        return digit     
    def getAttatchImg(self, share_f):
        att_name = share_f
        attach_full_name = os.path.join(self.temp_dir, att_name)
        self.attach_doc = fitz.open(attach_full_name)   
        self.pdf2img(self.attach_doc)  
        # self.digit = self.check_scan_or_digit() 
        self.attach_doc.close()  

    def getTitrow(self, colNums, boxes, new_bin_img, ocrProp):

        title_row = []
        for i in range(colNums):
            y0, x0, y1, x1 = boxes[0][i][0], boxes[0][i][1], boxes[1][i][0]+boxes[1][i][2], boxes[0][i][1]+boxes[0][i][3]
            temp_read = new_bin_img[y0:y1, x0:x1]
            img_to_read = self.MainImgTemp[y0:y1, x0:x1]
            if len(np.unique(temp_read)) == 1: 
                te = ''
                title_row.append(te)
                continue
            img_to_read = self.text_region(img_to_read,  temp_read)
            if ocrProp == "tess":
                te = pytesseract.image_to_string(img_to_read, config='--psm 6')
            else:
                te = ' '.join([v[1] for v in reader.readtext(img_to_read)])
            title_row.append(te)
        return title_row

    def getInds(self, inds, title_row, nameAddr):
        # name, number, address = setting                  
        for j, txt in enumerate(title_row):
            txt = txt.lower().strip()
            if ("name" in txt.split('\n') or "shareholders" in txt.split('\n') or ("ame" in txt and "full" in txt) \
                or ("ame" in txt and "share" in txt) or "name" in txt and "addr" in txt or 'particular' in txt \
                    or ('name' in txt and 'father' in txt)) and inds[0] == -1:
                inds[0], name = j, False
                if "addr" in txt: nameAddr = True
                # tit_row_num.append(i)
            elif "address" in txt and inds[1] == -1 and inds[0] != -1:
                inds[1], address = j, False
                # tit_row_num.append(i)
            elif (txt.strip() == "number" or txt.strip() == "no" or txt.strip() == "nos" or (("numb" in txt or "no" in txt) and "share" in txt)) and inds[2] == -1:
                inds[2], number = j, False
                # tit_row_num.append(i)
        if inds[0] ==-1 and inds[2] != -1:
            for j, txt in enumerate(title_row):
                txt = txt.lower()
                if 'shareholder' in txt: 
                    inds[0] = j
                    break
        if inds[0] == -1:
            fml_name_ind = []
            for i, txt in enumerate(title_row):
                txt = txt.lower()
                if ("irst" in txt) or ("middl" in txt) or ("last" in txt):
                    fml_name_ind.append(i)
            if fml_name_ind != []:
                inds[0] = fml_name_ind

        return inds, nameAddr

    def getExactBoxes(self, boxes, new_bin_img, refInds, nameAddr):

        if refInds[0] == -1:
            
            colNums = boxes.shape[1]
            title_row = self.getTitrow(colNums, boxes, new_bin_img, 'tess')
            nameAddr = False
            refInds, nameAddr = self.getInds(refInds, title_row, nameAddr)
            if refInds[0] == -1 or refInds[2] == -1:
                title_row = self.getTitrow(colNums, boxes, new_bin_img, 'easy')
                refInds, nameAddr = self.getInds(refInds, title_row, nameAddr)
        
        inds = refInds.copy()
        maxInd = -1
        newInds = []
        for k, ind in enumerate(inds):
            if ind != -1:
                if type(ind) == list:
                    newInds += ind
                    inds[k] = [i for i in range(len(ind))]
                    maxInd = len(ind)-1
                else:
                    newInds.append(ind)
                    inds[k] = maxInd + 1
                    maxInd = inds[k]
                    
        boxes = boxes[:, newInds]

        return boxes, inds, refInds, nameAddr      
    def parse_doc(self):
        '''
        In a document, main process is done for all pages 
        '''
        # Split and convert pages to images
        blackImg = np.zeros((512,512))
        cv2.imwrite(os.path.join(self.output_dir,'check', self.doc_name+'.jpg'), blackImg) # damaged file
        checking = 0
        main_prop, attatch_info = attatch_info = [], [[], [], []]
        try:
            main_prop = self.mainProp()
            cv2.imwrite(os.path.join(self.output_dir,'check', self.doc_name+'.jpg'), blackImg+150) # attatch file issue
            checking = 1
            share_f = self.attachProcessing()
            self.digit_doc.close()
            
            if share_f == "01":
                err = "PDF file is damaged or PDF file hasn't attatchment"
                print(err)
                '''
                https://github.com/pymupdf/PyMuPDF/discussions/776
                '''
            # for idx, page in enumerate(self.digit_doc):
            else:
                self.getAttatchImg(share_f)
                '''
                self.MainImg, self.img, self.img_removedByline : important images.
                '''
                attatch_infos = []
                cnt = 0
                for k, im in enumerate(self.att_pages):
                    if k > 1: break
                    self.MainImg = im.copy()
                    tables = cascade_mmdet(self.MainImg)
                    for table in tables:
                        self.MainImgTemp = self.MainImg[max(0,table[1]-25):table[3], max(0,table[0]-20):table[2]]
                        self.imgTemp = self.AdaptiveThreshold(self.MainImgTemp) # self.img is gotten in this function
                        self.MainImgTemp, self.imgTemp = self.preprocess_image(self.MainImgTemp, self.imgTemp)
                        img_vh, new_bin_img = self.getting_table()
                        ## find necessary columns ##
                        boxes = self.box_detection(img_vh)
                        try:
                            boxes, inds, refInds, nameAddr = self.getExactBoxes(boxes, new_bin_img, refInds, nameAddr)
                        except: 
                            boxes, inds, refInds, nameAddr = self.getExactBoxes(boxes, new_bin_img, [-1, -1, -1], False)
                        # if cnt == 0: inds = tempInds.copy()
                        text = self.box_text_detection(boxes, new_bin_img, cnt)
                        attatch_infos.append(self.infoFromText(text, inds, nameAddr))
                        cnt += 1
                        
                attatch_info = [[],[],[]]
                for atta in attatch_infos:
                    attatch_info[0] +=  atta[0]
                    attatch_info[1] +=  atta[1]
                    attatch_info[2] +=  atta[2]
                if attatch_info[0] != [] and checking != 2:
                    checking = 2
                    print(f"******** Successfully converted in {self.doc_name+'.pdf'}  ********")
        except: 
            print(f"######## Failed in converting of {self.doc_name+'.pdf'}  ########")
            pass

        return main_prop, attatch_info, checking