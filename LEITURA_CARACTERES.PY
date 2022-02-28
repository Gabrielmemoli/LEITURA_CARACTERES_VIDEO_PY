import cv2 
import imutils
import numpy as np
from matplotlib import pyplot as plt
from google.colab.patches import cv2_imshow
from imutils.object_detection import non_max_suppression
from PIL import Image
from PIL import ImageFont, ImageDraw, Image

from google.colab import drive
drive.mount('/content/gdrive')

!sudo apt install tesseract-ocr
!pip install pytesseract 
!mkdir tessdata
!wget -O ./tessdata/por.traineddata https://github.com/tesseract-ocr/tessdata/blob/main/por.traineddata?raw=true

import pytesseract

config_tesseract = "--tessdata-dir tessdata --psm 7"

def tesseract_OCR(img, config_tesseract):
  texto = pytesseract.image_to_string(img, lang='por', config=config_tesseract)
  return texto

def pre_processamento(img):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  maior = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
  valor, otsu = cv2.threshold(maior, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
  return otsu

!cp /content/frozen_east_text_detection.pb .
!cp -R /content/fontes .

detector = "/content/frozen_east_text_detection.pb"
largura_EAST, altura_EAST = 640, 640 # 320x320

min_conf_EAST = 0.9 #@param {type:"number", min:0.0, max:1.0}]

camadas_EAST = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

def dados_geometricos(geometry, y):
  x0_data = geometry[0, 0, y]
  x1_data = geometry[0, 1, y]
  x2_data = geometry[0, 2, y]
  x3_data = geometry[0, 3, y]
  data_angulos = geometry[0, 4, y]
  
  return data_angulos, x0_data, x1_data, x2_data, x3_data

def calculos_geometria(data_angulos, x0_data, x1_data, x2_data, x3_data, x, y):
  (offsetX, offsetY) = (x * 4.0, y * 4.0)

  angulo = data_angulos[x]
  cos = np.cos(angulo)
  sin = np.sin(angulo)
  
  h = x0_data[x] + x2_data[x]
  w = x1_data[x] + x3_data[x]

  fimX = int(offsetX + (cos * x1_data[x]) + (sin * x2_data[x]))
  fimY = int(offsetY - (sin * x1_data[x]) + (cos * x2_data[x]))
  inicioX = int(fimX - w)
  inicioY = int(fimY - h)

  return inicioX, inicioY, fimX, fimY

def EAST_processamento(img, largura, altura, net, nomes_camadas, min_confianca):
  original = img.copy()
  (H, W) = img.shape[:2]

  proporcao_W = W / float(largura)
  proporcao_H = H / float(altura)

  img = cv2.resize(img, (largura, altura))
  (H, W) = img.shape[:2]

  blob = cv2.dnn.blobFromImage(img, 1.0, (W, H), swapRB=True, crop=False)

  net.setInput(blob)
  (scores, geometry) = net.forward(nomes_camadas)

  (linhas, colunas) = scores.shape[2:4]

  caixas = []
  confiancas = []

  for y in range(0, linhas):
    data_scores = scores[0, 0, y]

    data_angulos, x0_data, x1_data, x2_data, x3_data = dados_geometricos(geometry, y)

    for x in range(0, colunas):
      if data_scores[x] < min_confianca:
        continue

      inicioX, inicioY, fimX, fimY = calculos_geometria(data_angulos, x0_data, x1_data, x2_data, x3_data, x, y)
      confiancas.append(data_scores[x])
      caixas.append((inicioX, inicioY, fimX, fimY))

  return proporcao_W, proporcao_H, confiancas, caixas

fonte = '/content/fontes/calibri.ttf'

def escreve_texto(texto, x, y, img, fonte, cor=(50, 50, 255), tamanho=22):
  fonte = ImageFont.truetype(fonte, tamanho)
  img_pil = Image.fromarray(img) 
  draw = ImageDraw.Draw(img_pil) 
  draw.text((x, y-tamanho), texto, font = fonte, fill = cor) 
  img = np.array(img_pil) 

  return img

def fundo_texto(texto, x, y, img, fonte, tamanho=32, cor_fundo=(200, 255, 0)):
  fundo = np.full((img.shape), (0,0,0), dtype=np.uint8)
  texto_fundo = escreve_texto(texto, x, y, fundo, fonte, (255,255,255), tamanho=tamanho)
  texto_fundo = cv2.dilate(texto_fundo,(np.ones((3,5),np.uint8)))
  fx,fy,fw,fh = cv2.boundingRect(texto_fundo[:,:,2])
  cv2.rectangle(img, (fx, fy), (fx + fw, fy + fh), cor_fundo, -1)

  return img

EASTnet = cv2.dnn.readNet(detector)

!cp -R /content/videos

arquivo_video = '/content/videos/Placa .mp4'
cap = cv2.VideoCapture(arquivo_video)
conectado, video = cap.read()
print(conectado, video)

from google.colab import drive
drive.mount('/content/drive')

video.shape

video_largura = video.shape[1]
video_altura = video.shape[0]

def redimensionar(largura, altura, largura_maxima = 600):
  if largura > largura_maxima:
    proporcao = largura / altura
    video_largura = largura_maxima
    video_altura = int(video_largura / proporcao)
  else:
    video_largura = largura
    video_altura = altura
  return video_largura, video_altura

video_largura, video_altura = redimensionar(video.shape[1], video.shape[0], 800)
print(video_largura, video_altura)

nome_arquivo = 'resultado_east_tesseract.avi'

fourcc = cv2.VideoWriter_fourcc(*'XVID')

fps = 24

saida_video = cv2.VideoWriter(nome_arquivo, fourcc, fps, (video_largura, video_altura))

amostras_exibir = 20
amostra_atual = 0

margem = 4

while (cv2.waitKey(1) < 0):
  conectado, frame = cap.read()

  if not conectado:
    break

  frame = cv2.resize(frame, (video_largura, video_altura))

  imagem_cp = frame.copy()

  proporcao_W, proporcao_H, confiancas, caixas = EAST_processamento(frame, largura_EAST, altura_EAST, EASTnet, camadas_EAST, min_conf_EAST)
  deteccoes = non_max_suppression(np.array(caixas), probs=confiancas)
  for (inicioX, inicioY, fimX, fimY) in deteccoes:
    inicioX = int(inicioX * proporcao_W)
    inicioY = int(inicioY * proporcao_H)
    fimX = int(fimX * proporcao_W)
    fimY = int(fimY * proporcao_H)

    cv2.rectangle(frame, (inicioX, inicioY), (fimX, fimY), (200,255,0), 2)

    roi = imagem_cp[inicioY - margem:fimY + margem, inicioX - margem:fimX + margem]

    img_process = pre_processamento(roi)

    texto = tesseract_OCR(img_process, config_tesseract)
    # http://www.asciitable.com/
    texto = ''.join([c if ord(c) < 128 else '' for c in texto]).strip()

    frame = fundo_texto(texto, inicioX, inicioY, frame, fonte, 20, (200,255,0))
    frame = escreve_texto(texto, inicioX, inicioY, frame, fonte, (0,0,0), 20)

  if amostra_atual <= amostras_exibir:
    cv2_imshow(frame)
    amostra_atual += 1

  saida_video.write(frame)

print('Terminou!')
saida_video.release()
cv2.destroyAllWindows()
