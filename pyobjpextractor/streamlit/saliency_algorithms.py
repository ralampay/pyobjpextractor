import streamlit as st
import numpy as np
import cv2

def main():
  st.title("Saliency Mappings")

  uploaded_file = st.file_uploader("Choose a file:")

  if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes  = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image       = cv2.imdecode(file_bytes, 1)

    # Original Image
    st.header("Original Image")
    st.image(image, channels="BGR")

    # Fine Grain
    st.header("Fine Grain")
    sm_fine_grain, sm_fine_grain_thresh_map = process_fine_grain(image)

    st.image(sm_fine_grain, channels="G")
    st.image(sm_fine_grain_thresh_map, channels="G")

    # Spectral Residual
    st.header("Spectral Residual")
    sm_spectral_residual, sm_spectral_residual_thresh_map = process_spectral_residual(image)

    st.image(sm_spectral_residual, channels="G")
    st.image(sm_spectral_residual_thresh_map, channels="G")

def process_fine_grain(image):
  saliency  = cv2.saliency.StaticSaliencyFineGrained_create() 
  (success, saliency_map) = saliency.computeSaliency(image)


  saliency_map = (saliency_map * 255).astype("uint8")
  thresh_map = cv2.threshold(saliency_map.astype("uint8"), 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

  return (saliency_map, thresh_map)

def process_spectral_residual(image):
  saliency  = cv2.saliency.StaticSaliencySpectralResidual_create() 
  (success, saliency_map) = saliency.computeSaliency(image)


  saliency_map = (saliency_map * 255).astype("uint8")
  thresh_map = cv2.threshold(saliency_map.astype("uint8"), 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

  return (saliency_map, thresh_map)

if __name__ == '__main__':
  main()
