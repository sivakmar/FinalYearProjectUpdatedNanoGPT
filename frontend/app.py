import streamlit as st
import requests
import whisper
import pytesseract as tess
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image
tess.pytesseract.tesseract_cmd=r'C:\Users\siva\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
import os
backend_url="https://0c91-34-136-12-119.ngrok-free.app/predict_sentiment"

enc = tiktoken.get_encoding("gpt2")
vocab_size = 50257

batch_size = 16 
block_size = 32 
max_iters = 3000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
torch.manual_seed(1337)

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   
        q = self.query(x) 
        wei = q @ k.transpose(-2,-1) * C**-0.5 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        wei = F.softmax(wei, dim=-1) 
        wei = self.dropout(wei)
        v = self.value(x) 
        out = wei @ v 
        return out
class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
class FeedFoward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
class GPT(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) 
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) 
        x = tok_emb + pos_emb 
        x = self.blocks(x) 
        x = self.ln_f(x) 
        logits = self.lm_head(x) 

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1) 
        return idx
model = GPT()
m = model.to(device)


m.load_state_dict(torch.load('model_weights.pth'))

# Streamlit app
st.title("Provide Inputs:")

# Sidebar content with increased font size for the header
st.sidebar.markdown("<h2 style='color:blue;'>Enhancement of sales and decision making<br>through the utilization of GEN AI</h2>", unsafe_allow_html=True)

# Select input type
option = st.selectbox('Select your input type:', ["Text", "Audio","Image"])
department_name = st.selectbox("Enter product department:",["Tops","Bottoms","Jackets","Intimate","Dresses"])
class_name = st.selectbox("Enter product class:", ["Knits","Jeans","Skirts","Pants","Blouses","Outwear","Shorts","Swim","Lounge"])

# Function to handle text input
def handle_text_input():
    user_input = st.text_input("Enter your review:", "")
    return user_input

# Function to handle audio input
def handle_audio_input():
    uploaded_audio = st.file_uploader("Choose an audio file...", type=["mp3", "wav"])
    if uploaded_audio is not None:
        st.audio(uploaded_audio, format="audio/*")
        return uploaded_audio

def handle_image_input():
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        return uploaded_image

def calculated_sentiment(payload):
        try:
            response = requests.post(backend_url, json=payload)
            if response.status_code == 200:
                result = response.json()
                st.success(f"Sentiment Score: {result['sentiment_score']}, Sentiment Class: {result['sentiment_class']}")
            else:
                st.error(f"Failed to send data. Status code: {response.status_code}")
        except requests.RequestException as e:
            st.error(f"Error sending data to the backend: {e}")
        return result['sentiment_class']

def calculated_recommendations(payload):
        custom_context = f"{department_name},{class_name},{data},{senti_class},Recommendations:" 
        encoded_context = enc.encode(custom_context)
        tensor_context = torch.tensor(encoded_context, dtype=torch.long, device=device).unsqueeze(0)  # Add batch dimension
        generated_text = m.generate(tensor_context, max_new_tokens=500)
        decoded_text = enc.decode(generated_text[0].tolist())
        length=len(custom_context)
        decoded_text=decoded_text[length:]
        st.write(decoded_text)

# Handle input based on the selected option
if option == "Text":
    data= handle_text_input()
elif option == "Image":
    data= handle_image_input()
elif option == "Audio":
    data= handle_audio_input()

# Button to trigger sending data to the backend
if st.button("Send"):
    if option =="Text":
        payload = {'data': data}
        senti_class=calculated_sentiment(payload)
        calculated_recommendations(payload)


    elif option=="Audio":
        def save_uploaded_file(uploaded_file,save_location,save_name):
            file_path=os.path.join(save_location,save_name)
            with open(file_path, 'wb') as file:
                file.write(uploaded_file.read())
            return file_path
        
        save_location="./backend/uploads"
        os.makedirs(save_location,exist_ok=True)
        save_name="saved_audio.mp3"
        uploaded_audio=data
        if uploaded_audio is not None:
            saved_file_path=save_uploaded_file(uploaded_audio,save_location,save_name)
            st.success(f"Audio file save to: {saved_file_path}")
        model=whisper.load_model("base")
        result=model.transcribe('./backend/uploads/saved_audio.mp3')
        data=result['text']

        payload = {'data': data}
        senti_class=calculated_sentiment(payload)
        calculated_recommendations(payload)
        
    elif option=="Image":
        def save_uploaded_file(uploaded_file,save_location,save_name):
            file_path=os.path.join(save_location,save_name)
            with open(file_path, 'wb') as file:
                file.write(uploaded_file.read())
            return file_path
        
        save_location="./backend/uploads2"
        os.makedirs(save_location,exist_ok=True)
        save_name="saved_image.png"
        uploaded_image=data
        if uploaded_image is not None:
            saved_file_path=save_uploaded_file(uploaded_image,save_location,save_name)
            st.success(f"Image file save to: {saved_file_path}")
        img=Image.open('./backend/uploads2/saved_image.png')
        extracted_text=tess.image_to_string(img)
        data=extracted_text
        payload = {'data': data}
        senti_class=calculated_sentiment(payload)
        calculated_recommendations(payload)



