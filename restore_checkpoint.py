"""Decrypt a private checkpoint archive. Requires: pip install cryptography."""
import argparse,base64,os
from pathlib import Path
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
p=argparse.ArgumentParser()
p.add_argument("encrypted",type=Path)
p.add_argument("output",type=Path)
a=p.parse_args()
if a.output.exists():p.error("output already exists")
k=os.environ.get("CHECKPOINT_KEY")
if not k:p.error("set CHECKPOINT_KEY to the private key supplied in the chat")
key=base64.urlsafe_b64decode(k+"="*((-len(k))%4))
magic=b"APPLECIDER_PRIVATE_V1\n"
b=a.encrypted.read_bytes()
if not b.startswith(magic):p.error("unexpected checkpoint format")
i=len(magic)
plain=AESGCM(key).decrypt(b[i:i+12],b[i+12:],magic)
a.output.write_bytes(plain)
print("Decrypted:",a.output)
