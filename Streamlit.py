import streamlit as st
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from LoadingGenerator import Generator, nz
import base64

st.set_page_config(layout="wide", page_title="Anime Face Generator")

def get_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_image_base64 = get_base64_image("animeimage.jpg")  
st.markdown(
    f"""
    <style>
    .stApp {{background-image: url("data:image/jpg;base64,{bg_image_base64}"); background-size: cover;
        background-position: center; background-repeat: no-repeat; }}

    .custom-title {{ background: linear-gradient(135deg, #ff66cc, #ff1493, #ffc0cb);
        color: white; font-size: 2.5em; text-align: center; padding: 8px 16px; border-radius: 12px;
        display: inline-block; margin-bottom: 25px; font-weight: bold;
        box-shadow: 0 0 10px rgba(255, 105, 180, 0.5);}}
    .custom-title-container {{ text-align: center; }}

    .custom-label {{ background: linear-gradient(135deg, #ff66cc, #ff1493, #ffc0cb); color: white;
        font-size: 1em; padding: 8px 14px; border-radius: 10px; display: inline-block; margin-bottom: 10px; font-weight: bold;
        box-shadow: 0 0 6px rgba(255, 105, 180, 0.4); }}

    .stButton>button {{display: block; margin: auto; background-color: #FF69B4;
        color: white;  border: none; padding: 0.75em 1.5em; border-radius: 12px;
        font-size: 1.1em; font-weight: bold;  box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
        position: relative; overflow: hidden; transition: 0.3s ease-in-out; }}

    .stButton>button::before {{content: '';
        position: absolute; top: 0;
        left: -75%; width: 50%;
        height: 100%;
        background: linear-gradient(120deg, rgba(255,255,255,0.2) 0%, rgba(255,255,255,0.5) 50%, rgba(255,255,255,0.2) 100%);
        transform: skewX(-20deg); }}

    .stButton>button:hover::before {{ left: 130%;
        transition: left 0.7s ease-in-out; }}

    .stButton>button:hover {{color: white !important;}}


    .stNumberInput>div>input {{ background-color: #2f3136;
        color: white; border-radius: 8px;
        border: none; padding: 10px; }}
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
st.markdown('<div class="custom-title-container"><div class="custom-title">Anime Face Generator</div></div>', unsafe_allow_html=True)
st.markdown('<div class="custom-label">Enter the number of faces you want to generate and click the button below:</div>', unsafe_allow_html=True)
n_imgs = st.number_input("", min_value=1, max_value=500, value=1, step=1)

if st.button("Generate"):
    noise = torch.randn(n_imgs, nz, 1, 1)
    with torch.no_grad():
        fake = gen(noise).detach().cpu()
    grid = vutils.make_grid(fake, nrow=min(n_imgs, 5), normalize=True)
    fig, ax = plt.subplots(figsize=(min(n_imgs, 5) * 1.0, (n_imgs // 5 + 1) * 1.0))
    ax.imshow(grid.permute(1, 2, 0))
    ax.axis("off")
    fig.patch.set_facecolor("none")
    st.pyplot(fig)


