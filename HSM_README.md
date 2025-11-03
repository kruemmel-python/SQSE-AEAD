# HSM Integration Examples (patterns)

This file shows example patterns to integrate HSM for key storage / derivation.
These examples are templates/pseudocode and must be adapted to your HSM vendor and environment.

## YubiHSM2 (python-yubihsm)
Install: `pip install yubihsm` (or use yubihsm2 python package per vendor docs)

Example pattern (pseudocode):
```
from yubihsm import YubiHsm, Client, Connector

connector = Connector('http://127.0.0.1:12345')  # or usb
client = Client(connector)
session = client.create_session(key_id=1, password='password')

# Use HSM to derive a file-specific key by HMAC or AES-KDF
# e.g. HSM HMAC-SHA256 of nonce||fileid with a stored HSM key
file_key = session.hmac_sign(key_id=1, data=nonce + fileid)
# Use file_key (32 bytes) as context key; never export master key.
```
Refer to Yubico docs for exact APIs and secure session management.

## AWS CloudHSM / PKCS#11 pattern
If using CloudHSM / any PKCS#11 HSM, use pkcs11 lib or vendor SDK.

Example pattern (pseudocode with python-pkcs11):
```
from pkcs11 import PKCS11Lib
lib = PKCS11Lib('/opt/cloudhsm/lib/libcloudhsm_pkcs11.so')
slot = lib.get_slot(0)
with slot.open(user_pin='...') as session:
    # Derive key using HMAC or other mechanism inside HSM
    # session.find_objects / session.create_secret_key etc.
    derived = session.digest_find_mechanism(...)
```
Always follow vendor recommended key management, backup and access control practices.
