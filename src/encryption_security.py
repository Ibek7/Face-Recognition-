# Data Encryption & Security Layer

import os
import hashlib
import hmac
import secrets
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import base64
from abc import ABC, abstractmethod

class EncryptionAlgorithm(Enum):
    """Supported encryption algorithms."""
    AES_256_GCM = "aes_256_gcm"
    AES_128_CBC = "aes_128_cbc"
    CHACHA20 = "chacha20"

class HashAlgorithm(Enum):
    """Supported hash algorithms."""
    SHA256 = "sha256"
    SHA512 = "sha512"
    BLAKE2B = "blake2b"

@dataclass
class EncryptionKey:
    """Encryption key with metadata."""
    key_id: str
    algorithm: EncryptionAlgorithm
    key_material: bytes
    created_at: float = field(default_factory=lambda: __import__('time').time())
    rotation_date: Optional[float] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_key_size(self) -> int:
        """Get key size in bits."""
        return len(self.key_material) * 8

@dataclass
class EncryptedData:
    """Encrypted data with metadata."""
    ciphertext: bytes
    key_id: str
    algorithm: EncryptionAlgorithm
    iv: Optional[bytes] = None  # Initialization vector
    tag: Optional[bytes] = None  # Authentication tag (for AEAD)
    salt: Optional[bytes] = None
    timestamp: float = field(default_factory=lambda: __import__('time').time())
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'ciphertext': base64.b64encode(self.ciphertext).decode(),
            'key_id': self.key_id,
            'algorithm': self.algorithm.value,
            'iv': base64.b64encode(self.iv).decode() if self.iv else None,
            'tag': base64.b64encode(self.tag).decode() if self.tag else None,
            'salt': base64.b64encode(self.salt).decode() if self.salt else None,
            'timestamp': self.timestamp
        }
    
    @staticmethod
    def from_dict(data: Dict) -> 'EncryptedData':
        """Create from dictionary."""
        return EncryptedData(
            ciphertext=base64.b64decode(data['ciphertext']),
            key_id=data['key_id'],
            algorithm=EncryptionAlgorithm(data['algorithm']),
            iv=base64.b64decode(data['iv']) if data.get('iv') else None,
            tag=base64.b64decode(data['tag']) if data.get('tag') else None,
            salt=base64.b64decode(data['salt']) if data.get('salt') else None,
            timestamp=data.get('timestamp')
        )

class Encryptor(ABC):
    """Base class for encryption."""
    
    @abstractmethod
    def encrypt(self, plaintext: bytes, key: EncryptionKey) -> EncryptedData:
        """Encrypt data."""
        pass
    
    @abstractmethod
    def decrypt(self, encrypted_data: EncryptedData, key: EncryptionKey) -> bytes:
        """Decrypt data."""
        pass

class AESGCMEncryptor(Encryptor):
    """AES-256-GCM encryptor."""
    
    def encrypt(self, plaintext: bytes, key: EncryptionKey) -> EncryptedData:
        """Encrypt with AES-256-GCM."""
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            from cryptography.hazmat.backends import default_backend
            
            # Generate random IV
            iv = secrets.token_bytes(12)
            
            cipher = AESGCM(key.key_material)
            ciphertext = cipher.encrypt(iv, plaintext, None)
            
            # Extract tag (last 16 bytes in GCM)
            tag = ciphertext[-16:]
            ciphertext_only = ciphertext[:-16]
            
            return EncryptedData(
                ciphertext=ciphertext_only,
                key_id=key.key_id,
                algorithm=key.algorithm,
                iv=iv,
                tag=tag
            )
        except ImportError:
            raise ImportError("cryptography library required for AES-GCM")
    
    def decrypt(self, encrypted_data: EncryptedData, key: EncryptionKey) -> bytes:
        """Decrypt AES-256-GCM."""
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            
            cipher = AESGCM(key.key_material)
            
            # Combine ciphertext and tag
            ciphertext_with_tag = encrypted_data.ciphertext + encrypted_data.tag
            
            plaintext = cipher.decrypt(
                encrypted_data.iv,
                ciphertext_with_tag,
                None
            )
            
            return plaintext
        except ImportError:
            raise ImportError("cryptography library required for AES-GCM")

class HashFunction:
    """Hash computation."""
    
    @staticmethod
    def hash_data(data: bytes, algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> str:
        """Hash data."""
        if algorithm == HashAlgorithm.SHA256:
            return hashlib.sha256(data).hexdigest()
        elif algorithm == HashAlgorithm.SHA512:
            return hashlib.sha512(data).hexdigest()
        elif algorithm == HashAlgorithm.BLAKE2B:
            return hashlib.blake2b(data).hexdigest()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    @staticmethod
    def hash_string(text: str, algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> str:
        """Hash string."""
        return HashFunction.hash_data(text.encode(), algorithm)

class KeyManager:
    """Manage encryption keys."""
    
    def __init__(self):
        self.keys: Dict[str, EncryptionKey] = {}
        self.active_key_id: Optional[str] = None
    
    def create_key(self, key_id: str, algorithm: EncryptionAlgorithm,
                  key_size: int = 256) -> EncryptionKey:
        """Create new key."""
        if key_id in self.keys:
            raise ValueError(f"Key {key_id} already exists")
        
        # Generate random key material
        key_bytes = key_size // 8
        key_material = secrets.token_bytes(key_bytes)
        
        key = EncryptionKey(
            key_id=key_id,
            algorithm=algorithm,
            key_material=key_material
        )
        
        self.keys[key_id] = key
        
        if not self.active_key_id:
            self.active_key_id = key_id
        
        return key
    
    def get_key(self, key_id: str) -> Optional[EncryptionKey]:
        """Get key by ID."""
        return self.keys.get(key_id)
    
    def get_active_key(self) -> Optional[EncryptionKey]:
        """Get active key."""
        if self.active_key_id:
            return self.keys.get(self.active_key_id)
        return None
    
    def rotate_key(self, key_id: str, new_key_id: str):
        """Rotate key."""
        if key_id not in self.keys:
            raise ValueError(f"Key {key_id} not found")
        
        # Mark old as inactive
        self.keys[key_id].is_active = False
        
        # Set new as active
        if new_key_id in self.keys:
            self.active_key_id = new_key_id
            self.keys[new_key_id].is_active = True

class PasswordHasher:
    """Hash passwords securely."""
    
    @staticmethod
    def hash_password(password: str, iterations: int = 100000) -> Tuple[str, str]:
        """Hash password with salt. Returns (hash, salt)."""
        salt = secrets.token_hex(32)
        
        # PBKDF2
        hashed = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode(),
            salt.encode(),
            iterations
        )
        
        return hashed.hex(), salt
    
    @staticmethod
    def verify_password(password: str, password_hash: str, salt: str,
                       iterations: int = 100000) -> bool:
        """Verify password."""
        hashed = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode(),
            salt.encode(),
            iterations
        )
        
        return hmac.compare_digest(hashed.hex(), password_hash)

class SecurityContext:
    """Security context for operations."""
    
    def __init__(self, user_id: str, tenant_id: str, 
                permissions: List[str] = None):
        self.user_id = user_id
        self.tenant_id = tenant_id
        self.permissions = permissions or []
        self.created_at = __import__('time').time()
    
    def has_permission(self, permission: str) -> bool:
        """Check permission."""
        return permission in self.permissions
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'user_id': self.user_id,
            'tenant_id': self.tenant_id,
            'permissions': self.permissions,
            'created_at': self.created_at
        }

class SecretManager:
    """Manage secrets securely."""
    
    def __init__(self, key_manager: KeyManager):
        self.key_manager = key_manager
        self.encryptor = AESGCMEncryptor()
        self.secrets: Dict[str, EncryptedData] = {}
    
    def store_secret(self, secret_id: str, secret_value: str) -> str:
        """Store secret."""
        key = self.key_manager.get_active_key()
        if not key:
            raise ValueError("No active encryption key")
        
        encrypted = self.encryptor.encrypt(
            secret_value.encode(),
            key
        )
        
        self.secrets[secret_id] = encrypted
        return secret_id
    
    def retrieve_secret(self, secret_id: str) -> Optional[str]:
        """Retrieve secret."""
        encrypted = self.secrets.get(secret_id)
        if not encrypted:
            return None
        
        key = self.key_manager.get_key(encrypted.key_id)
        if not key:
            raise ValueError(f"Key {encrypted.key_id} not found")
        
        plaintext = self.encryptor.decrypt(encrypted, key)
        return plaintext.decode()

# Example usage
if __name__ == "__main__":
    # Create key manager
    key_mgr = KeyManager()
    key_mgr.create_key("key_1", EncryptionAlgorithm.AES_256_GCM)
    
    # Hash data
    data_hash = HashFunction.hash_string("sensitive_data")
    print(f"SHA256 Hash: {data_hash}")
    
    # Hash password
    pwd_hash, salt = PasswordHasher.hash_password("mypassword123")
    print(f"Password Hash: {pwd_hash[:20]}...")
    
    # Verify password
    is_valid = PasswordHasher.verify_password("mypassword123", pwd_hash, salt)
    print(f"Password Valid: {is_valid}")
    
    # Store and retrieve secret
    secret_mgr = SecretManager(key_mgr)
    secret_mgr.store_secret("api_key", "secret_api_key_value")
    retrieved = secret_mgr.retrieve_secret("api_key")
    print(f"Retrieved Secret: {retrieved}")
