from PIL import Image
import numpy as np
import hashlib

STOP = "#####"

def text_to_bits(text):
    return ''.join(format(ord(c), '08b') for c in text)

def bits_to_text(bits):
    chars = [bits[i:i+8] for i in range(0, len(bits), 8)]
    return ''.join(chr(int(c, 2)) for c in chars)

def key_order(password, n):
    # password -> big number -> seeds the random generator -> shuffled order of positions
    seed = int(hashlib.sha256(password.encode()).hexdigest(), 16)
    return np.random.default_rng(seed).permutation(n)

def hide_key(in_path, out_path, message, password):
    pixels = np.array(Image.open(in_path).convert("RGB"))
    flat = pixels.flatten()
    bits = text_to_bits(message + STOP)
    order = key_order(password, len(flat))

    for pos, bit in zip(order[:len(bits)], bits):
        flat[pos] = (flat[pos] & 254) | int(bit)

    Image.fromarray(flat.reshape(pixels.shape)).save(out_path, format="PNG")

def _find_message(flat, order=None):
    values = flat if order is None else flat[order]
    bits = ''.join(str(v & 1) for v in values)
    bits = bits[:len(bits) // 8 * 8]
    text = bits_to_text(bits)
    idx = text.find(STOP)
    return text[:idx] if idx != -1 else None

def extract_key(path, password):
    flat = np.array(Image.open(path).convert("RGB")).flatten()
    return _find_message(flat, key_order(password, len(flat)))

def extract_plain(path):
    flat = np.array(Image.open(path).convert("RGB")).flatten()
    return _find_message(flat)

# make a small test image (about 1000 px wide) so runs are fast
cover = Image.open("images/test.jpg").convert("RGB")
cover.thumbnail((1000, 1000))
cover.save("images/small.png")

hide_key("images/small.png", "images/stego_key.png", "Hello Srajika", "mypassword")
print("Right password:", extract_key("images/stego_key.png", "mypassword"))
print("Wrong password:", extract_key("images/stego_key.png", "wrongpass"))
print("Plain extractor:", extract_plain("images/stego_key.png"))