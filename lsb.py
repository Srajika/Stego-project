from PIL import Image
import numpy as np

STOP = "#####"   # stop marker: tells extract() where the message ends

def text_to_bits(text):
    return ''.join(format(ord(c), '08b') for c in text)

def bits_to_text(bits):
    chars = [bits[i:i+8] for i in range(0, len(bits), 8)]
    return ''.join(chr(int(c, 2)) for c in chars)

def hide(in_path, out_path, message):
    img = Image.open(in_path).convert("RGB")
    pixels = np.array(img)              # shape: (height, width, 3)
    flat = pixels.flatten()             # one long list of channel values
    bits = text_to_bits(message + STOP)

    if len(bits) > len(flat):
        raise ValueError("Message too long for this image")

    for i, bit in enumerate(bits):
        flat[i] = (flat[i] & 254) | int(bit)   # force last bit = message bit

    Image.fromarray(flat.reshape(pixels.shape)).save(out_path, format="PNG")

def extract(path):
    flat = np.array(Image.open(path).convert("RGB")).flatten()
    bits = ''.join(str(v & 1) for v in flat)    # last bit of every value
    bits = bits[:len(bits) // 8 * 8]            # keep whole bytes only
    text = bits_to_text(bits)
    idx = text.find(STOP)
    return text[:idx] if idx != -1 else None

hide("images/test.jpg", "images/stego.png", "Hello Srajika")
print(extract("images/stego.png"))

orig = np.array(Image.open("images/test.jpg").convert("RGB")).astype(int)
stego = np.array(Image.open("images/stego.png").convert("RGB")).astype(int)

diff = np.abs(orig - stego)
print("Max change in any value:", diff.max())
print("Values changed:", np.count_nonzero(diff), "out of", diff.size)

Image.open("images/stego.png").convert("RGB").save("images/stego_q90.jpg", quality=90)
print(extract("images/stego_q90.jpg"))