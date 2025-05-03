import streamlit as st
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from dcgan2 import Generator, nz 

st.set_page_config(layout="wide", page_title="Anime Face Generator")
st.markdown(
    """
    <style>
    body {background-color: #000000; color: white;}
    .stApp {background-color: #000000; color: white;}
    .css-1lcbzv3 {background-color: #000000;}
    .stButton>button {background-color: #4CAF50; color: white;}
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def lg():
    g = Generator()  
    g.load_state_dict(torch.load("generator.pth", map_location=torch.device("cpu"))) 
    g.eval() 
    return g

gen = lg()
st.title("Anime Face Generator") 
n_imgs = st.number_input("Enter the number of faces you want to generate and click the generate button", min_value=1, max_value=500, value=1, step=1)

if st.button("Generate"):
    noise = torch.randn(n_imgs, nz, 1, 1)
    with torch.no_grad():
        fake = gen(noise).detach().cpu()
    grid = vutils.make_grid(fake, nrow=min(n_imgs, 5), normalize=True)
    fig, ax = plt.subplots(figsize=(min(n_imgs, 5) * 0.8, (n_imgs // 5 + 1) * 0.8))
    ax.imshow(grid.permute(1, 2, 0))  
    ax.axis("off")
    fig.patch.set_facecolor('#000000')
    st.pyplot(fig)
