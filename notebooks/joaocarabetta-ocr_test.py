import pdf2image
from PIL import Image
import pytesseract
import argparse
import os
from pdf2image import convert_from_path

pages = convert_from_path('adenda_15_uniformes_1542118660553.pdf', 500)
text = pytesseract.image_to_string(pages[0])
text = text.lower().split('\n')
[line.split(' ')[1] for line in text if 'seccion' in line]