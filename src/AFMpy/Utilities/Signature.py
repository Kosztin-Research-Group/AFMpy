import hashlib
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.exceptions import InvalidSignature

import logging

logger = logging.getLogger(__name__)

def generate_keys(private_key_filepath: str,
                  public_key_filepath: str,
                  key_size: int = 2048) -> None:
    '''
    Generate a public/private key pair and write them to files. Necessary for signing and verifying files.
    
    Args:
        private_key_filepath (str):
            The path to the private key file.
        public_key_filepath (str):
            The path to the public key file.
        key_size (int):
            The size of the key in bits. Default is 2048.
    Returns:
        None
    '''
    logger.debug(f'Generating a {key_size}-bit RSA key pair.')

    # Generate the private key.
    logger.debug('Generating the private key.')
    private_key = rsa.generate_private_key(public_exponent=65537,
                                           key_size=key_size)

    # Generate the public key.
    logger.debug('Generating the public key.')
    public_key = private_key.public_key()

    # Write the private key to a file.
    logger.debug(f'Writing the private key to {private_key_filepath}.')
    pem_private_key = private_key.private_bytes(encoding=serialization.Encoding.PEM,
                                                format=serialization.PrivateFormat.PKCS8,
                                                encryption_algorithm=serialization.NoEncryption())
    with open(private_key_filepath, 'wb') as file:
        file.write(pem_private_key)

    # Write the public key to a file.
    logger.debug(f'Writing the public key to {public_key_filepath}.')
    pem_public_key = public_key.public_bytes(encoding=serialization.Encoding.PEM,
                                             format=serialization.PublicFormat.SubjectPublicKeyInfo)
    with open(public_key_filepath, 'wb') as file:
        file.write(pem_public_key)

def sign_file(filepath: str,
              private_key_filepath: str) -> None:
    '''
    Digitally signs a file with a private key.

    Args:
        filepath (str):
            The path to the file to sign.
        private_key_filepath (str):
            The path to the private key file.
    Returns:
        None
    '''

    # Load the private key file.
    logger.debug(f'Loading the private key from {private_key_filepath}.')
    with open(private_key_filepath, 'rb') as key_file:
        private_key = serialization.load_pem_private_key(key_file.read(), password=None)

    # Hash the pickle file.
    logger.debug(f'Hashing and generating the signature for {filepath}.')
    digest = hashlib.sha256()
    with open(filepath, 'rb') as pickle_file:
        for chunk in iter(lambda: pickle_file.read(4096), b''):
            digest.update(chunk)
    hash = digest.digest()

    # Sign the hash.
    sig = private_key.sign(hash,
                           padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                           hashes.SHA256())
    
    # Write the signature to a file.
    logger.debug(f'Writing the signature to {filepath}.sig.')
    with open(f'{filepath}.sig', 'wb') as sig_file:
        sig_file.write(sig)

def verify_file(filepath: str,
                public_key_filepath: str) -> None:
    '''
    Verifies the digital signature of a signed file.

    Args:
        filepath (str):
            The path to the signed file.
        public_key_filepath (str):
            The path to the public key file.
    Returns:
        None
    Raises:
        InvalidSignature:
            If the signature is invalid.
    '''
    # Read the public key file.
    logger.debug(f'Loading the public key from {public_key_filepath}.')
    with open(public_key_filepath, 'rb') as file:
        public_key = serialization.load_pem_public_key(file.read())

    # Load the signature file.
    logger.debug(f'Loading the signature from {filepath}.sig.')
    with open(f'{filepath}.sig', 'rb') as file:
        sig = file.read()

    # Hash the file.
    logger.debug(f'Hashing the file {filepath}.')
    digest = hashlib.sha256()
    with open(filepath, 'rb') as file:
        for chunk in iter(lambda: file.read(4096), b''):
            digest.update(chunk)
    hash = digest.digest()

    # Verify the signature.
    logger.debug(f'Verifying the signature of {filepath}.')
    try:
        public_key.verify(sig,
                        hash,
                        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                        hashes.SHA256())
    except InvalidSignature:
        # If the signature is invalid, log it and raise an exception.
        logger.error(f'The signature of {filepath} is invalid.')
        raise InvalidSignature(f'The signature of {filepath} is invalid.')
    
    # If the signature is valid, log it.
    logger.info(f'The signature of {filepath} is valid.')