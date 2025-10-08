# Security and Privacy Guide

This guide covers security best practices, privacy considerations, and compliance requirements for the Face Recognition System.

## Table of Contents

1. [Security Architecture](#security-architecture)
2. [Data Protection](#data-protection)
3. [Privacy Compliance](#privacy-compliance)
4. [Authentication & Authorization](#authentication--authorization)
5. [Secure Development Practices](#secure-development-practices)
6. [Vulnerability Management](#vulnerability-management)
7. [Incident Response](#incident-response)
8. [Audit and Compliance](#audit-and-compliance)
9. [Data Retention](#data-retention)
10. [User Rights Management](#user-rights-management)

## Security Architecture

### Defense in Depth Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                    External Security Layer                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │    WAF      │  │     CDN     │  │   DDoS      │         │
│  │ Protection  │  │ Protection  │  │ Protection  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                  Network Security Layer                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Firewall   │  │     VPN     │  │   Network   │         │
│  │   Rules     │  │   Access    │  │ Monitoring  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                Application Security Layer                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ API Gateway │  │    mTLS     │  │    Rate     │         │
│  │   Security  │  │Certificates │  │  Limiting   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                   Data Security Layer                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Encryption  │  │    Access   │  │   Audit     │         │
│  │   at Rest   │  │   Control   │  │   Logging   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

### Security Controls

**1. Input Validation and Sanitization**

```python
# src/security/input_validation.py
import re
from typing import Optional, List
from fastapi import HTTPException
from PIL import Image
import magic
import hashlib

class InputValidator:
    # Allowed file types
    ALLOWED_IMAGE_TYPES = {
        'image/jpeg': [b'\xff\xd8\xff'],
        'image/png': [b'\x89\x50\x4e\x47'],
        'image/bmp': [b'\x42\x4d'],
        'image/tiff': [b'\x49\x49', b'\x4d\x4d']
    }
    
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_IMAGE_DIMENSION = 4096
    
    @staticmethod
    def validate_image_file(file_content: bytes) -> bool:
        """Validate image file type and content."""
        if len(file_content) > InputValidator.MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large")
        
        # Check file signature
        mime_type = magic.from_buffer(file_content, mime=True)
        if mime_type not in InputValidator.ALLOWED_IMAGE_TYPES:
            raise HTTPException(status_code=400, detail="Invalid file type")
        
        # Verify file header
        file_signatures = InputValidator.ALLOWED_IMAGE_TYPES[mime_type]
        if not any(file_content.startswith(sig) for sig in file_signatures):
            raise HTTPException(status_code=400, detail="File signature mismatch")
        
        try:
            # Validate with PIL
            with Image.open(io.BytesIO(file_content)) as img:
                if img.width > InputValidator.MAX_IMAGE_DIMENSION or \
                   img.height > InputValidator.MAX_IMAGE_DIMENSION:
                    raise HTTPException(status_code=400, detail="Image dimensions too large")
                
                # Check for malicious metadata
                if hasattr(img, '_getexif') and img._getexif():
                    # Remove EXIF data for privacy
                    img_clean = img.copy()
                    img_clean.save(io.BytesIO(), format=img.format)
                    
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        return True
    
    @staticmethod
    def validate_person_name(name: str) -> str:
        """Validate and sanitize person name."""
        if not name or len(name.strip()) == 0:
            raise HTTPException(status_code=400, detail="Name cannot be empty")
        
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>"\';()&+]', '', name.strip())
        
        if len(sanitized) > 100:
            raise HTTPException(status_code=400, detail="Name too long")
        
        if len(sanitized) < 2:
            raise HTTPException(status_code=400, detail="Name too short")
        
        return sanitized
    
    @staticmethod
    def validate_email(email: str) -> str:
        """Validate email format."""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            raise HTTPException(status_code=400, detail="Invalid email format")
        return email.lower()
    
    @staticmethod
    def generate_secure_filename(original_filename: str) -> str:
        """Generate secure filename."""
        # Remove path components
        filename = os.path.basename(original_filename)
        
        # Remove dangerous characters
        filename = re.sub(r'[^a-zA-Z0-9._-]', '', filename)
        
        # Add timestamp and hash for uniqueness
        timestamp = int(time.time())
        file_hash = hashlib.md5(filename.encode()).hexdigest()[:8]
        
        name, ext = os.path.splitext(filename)
        return f"{timestamp}_{file_hash}_{name}{ext}"

# Usage in API endpoints
@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    content = await file.read()
    InputValidator.validate_image_file(content)
    
    secure_filename = InputValidator.generate_secure_filename(file.filename)
    # Process file...
```

**2. SQL Injection Prevention**

```python
# src/security/database_security.py
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
import logging

class SecureDatabase:
    def __init__(self, session):
        self.session = session
    
    def safe_query(self, query: str, params: dict = None):
        """Execute parameterized queries safely."""
        try:
            if params:
                # Use SQLAlchemy's text() with bound parameters
                result = self.session.execute(text(query), params)
            else:
                result = self.session.execute(text(query))
            return result
        except SQLAlchemyError as e:
            logging.error(f"Database query error: {e}")
            self.session.rollback()
            raise
    
    def get_person_by_id(self, person_id: int):
        """Safe person lookup with parameterized query."""
        query = """
        SELECT id, name, email, created_at 
        FROM persons 
        WHERE id = :person_id AND deleted_at IS NULL
        """
        return self.safe_query(query, {"person_id": person_id}).fetchone()
    
    def search_persons(self, search_term: str, limit: int = 50):
        """Safe person search with input sanitization."""
        # Sanitize search term
        search_term = search_term.replace('%', '\\%').replace('_', '\\_')
        
        query = """
        SELECT id, name, email, created_at 
        FROM persons 
        WHERE name ILIKE :search_term 
        AND deleted_at IS NULL
        ORDER BY name
        LIMIT :limit
        """
        return self.safe_query(query, {
            "search_term": f"%{search_term}%",
            "limit": limit
        }).fetchall()

# ORM-based approach (preferred)
from sqlalchemy.orm import Session

def get_person_embeddings(session: Session, person_id: int):
    """Use ORM for automatic parameterization."""
    return session.query(FaceEmbedding)\
                 .filter(FaceEmbedding.person_id == person_id)\
                 .filter(FaceEmbedding.deleted_at.is_(None))\
                 .all()
```

**3. API Security Headers**

```python
# src/security/headers.py
from fastapi import FastAPI
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.cors import CORSMiddleware
import secrets

def add_security_headers(app: FastAPI):
    """Add security headers to all responses."""
    
    @app.middleware("http")
    async def security_headers_middleware(request, call_next):
        response = await call_next(request)
        
        # Generate nonce for CSP
        nonce = secrets.token_urlsafe(16)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        
        # Content Security Policy
        csp = f"""
        default-src 'self';
        script-src 'self' 'nonce-{nonce}';
        style-src 'self' 'unsafe-inline';
        img-src 'self' data: blob:;
        connect-src 'self';
        font-src 'self';
        object-src 'none';
        base-uri 'self';
        form-action 'self';
        frame-ancestors 'none';
        upgrade-insecure-requests;
        """.replace('\n', ' ').strip()
        
        response.headers["Content-Security-Policy"] = csp
        
        # HSTS (if HTTPS)
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = \
                "max-age=31536000; includeSubDomains; preload"
        
        return response
    
    # Trusted host middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", "your-domain.com"]
    )
    
    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://your-frontend-domain.com"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"]
    )
```

## Data Protection

### Encryption at Rest

```python
# src/security/encryption.py
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import os
import base64
import json

class DataEncryption:
    def __init__(self, password: str = None):
        if password is None:
            password = os.environ.get('ENCRYPTION_KEY')
        
        if not password:
            raise ValueError("Encryption password not provided")
        
        self.key = self._derive_key(password)
        self.cipher = Fernet(self.key)
    
    def _derive_key(self, password: str) -> bytes:
        """Derive encryption key from password."""
        salt = os.environ.get('ENCRYPTION_SALT', 'default_salt').encode()
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))
    
    def encrypt_data(self, data: dict) -> str:
        """Encrypt dictionary data."""
        json_data = json.dumps(data).encode()
        encrypted = self.cipher.encrypt(json_data)
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt_data(self, encrypted_data: str) -> dict:
        """Decrypt data back to dictionary."""
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
        decrypted = self.cipher.decrypt(encrypted_bytes)
        return json.loads(decrypted.decode())
    
    def encrypt_file(self, file_path: str) -> str:
        """Encrypt file and return encrypted file path."""
        with open(file_path, 'rb') as file:
            file_data = file.read()
        
        encrypted_data = self.cipher.encrypt(file_data)
        encrypted_path = file_path + '.encrypted'
        
        with open(encrypted_path, 'wb') as encrypted_file:
            encrypted_file.write(encrypted_data)
        
        # Remove original file
        os.remove(file_path)
        return encrypted_path
    
    def decrypt_file(self, encrypted_file_path: str) -> str:
        """Decrypt file and return original file path."""
        with open(encrypted_file_path, 'rb') as encrypted_file:
            encrypted_data = encrypted_file.read()
        
        decrypted_data = self.cipher.decrypt(encrypted_data)
        original_path = encrypted_file_path.replace('.encrypted', '')
        
        with open(original_path, 'wb') as original_file:
            original_file.write(decrypted_data)
        
        return original_path

# Database field encryption
from sqlalchemy_utils import EncryptedType
from sqlalchemy_utils.types.encrypted.encrypted_type import AesEngine

class EncryptedPersonModel(Base):
    __tablename__ = 'persons_encrypted'
    
    id = Column(Integer, primary_key=True)
    
    # Encrypted fields
    name = Column(EncryptedType(String, secret_key, AesEngine, 'pkcs5'))
    email = Column(EncryptedType(String, secret_key, AesEngine, 'pkcs5'))
    phone = Column(EncryptedType(String, secret_key, AesEngine, 'pkcs5'))
    
    # Non-encrypted fields
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

# Usage
encryption = DataEncryption()

# Encrypt sensitive data before storage
sensitive_data = {
    "ssn": "123-45-6789",
    "biometric_template": face_embedding.tolist()
}
encrypted = encryption.encrypt_data(sensitive_data)

# Store encrypted data in database
person.encrypted_data = encrypted
```

## Privacy Compliance

### GDPR Compliance

```python
# src/privacy/gdpr_compliance.py
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import json

class GDPRCompliance:
    def __init__(self, db_session):
        self.db = db_session
    
    def export_user_data(self, person_id: int) -> Dict:
        """Export all personal data for GDPR Article 15 (Right of Access)."""
        person = self.db.query(Person).filter_by(id=person_id).first()
        if not person:
            raise ValueError("Person not found")
        
        # Collect all personal data
        embeddings = self.db.query(FaceEmbedding)\
                           .filter_by(person_id=person_id)\
                           .all()
        
        recognition_results = self.db.query(RecognitionResult)\
                                    .filter_by(person_id=person_id)\
                                    .all()
        
        audit_logs = self.db.query(AuditLog)\
                           .filter_by(user_id=person_id)\
                           .all()
        
        export_data = {
            "personal_information": {
                "id": person.id,
                "name": person.name,
                "email": person.email,
                "created_at": person.created_at.isoformat(),
                "updated_at": person.updated_at.isoformat()
            },
            "biometric_data": [
                {
                    "id": emb.id,
                    "quality_score": emb.quality_score,
                    "created_at": emb.created_at.isoformat(),
                    "embedding_hash": hashlib.sha256(emb.embedding).hexdigest()
                }
                for emb in embeddings
            ],
            "recognition_history": [
                {
                    "id": result.id,
                    "confidence": result.confidence,
                    "timestamp": result.created_at.isoformat(),
                    "location": result.location
                }
                for result in recognition_results
            ],
            "audit_trail": [
                {
                    "operation": log.operation,
                    "timestamp": log.timestamp.isoformat(),
                    "table": log.table_name
                }
                for log in audit_logs
            ],
            "export_metadata": {
                "export_date": datetime.utcnow().isoformat(),
                "data_controller": "Your Organization",
                "contact_dpo": "dpo@yourorganization.com"
            }
        }
        
        return export_data
    
    def delete_user_data(self, person_id: int, deletion_reason: str) -> Dict:
        """Delete all personal data for GDPR Article 17 (Right to Erasure)."""
        person = self.db.query(Person).filter_by(id=person_id).first()
        if not person:
            raise ValueError("Person not found")
        
        deletion_summary = {
            "person_id": person_id,
            "deletion_date": datetime.utcnow().isoformat(),
            "deletion_reason": deletion_reason,
            "deleted_data": {}
        }
        
        try:
            # Delete face embeddings
            embeddings_count = self.db.query(FaceEmbedding)\
                                     .filter_by(person_id=person_id)\
                                     .count()
            self.db.query(FaceEmbedding)\
                   .filter_by(person_id=person_id)\
                   .delete()
            
            # Delete recognition results
            results_count = self.db.query(RecognitionResult)\
                                  .filter_by(person_id=person_id)\
                                  .count()
            self.db.query(RecognitionResult)\
                   .filter_by(person_id=person_id)\
                   .delete()
            
            # Delete person record
            self.db.delete(person)
            
            # Log deletion in audit trail
            deletion_log = AuditLog(
                table_name="gdpr_deletion",
                operation="DELETE",
                user_id=person_id,
                old_values={"reason": deletion_reason},
                timestamp=datetime.utcnow()
            )
            self.db.add(deletion_log)
            
            self.db.commit()
            
            deletion_summary["deleted_data"] = {
                "person_record": 1,
                "face_embeddings": embeddings_count,
                "recognition_results": results_count
            }
            
        except Exception as e:
            self.db.rollback()
            raise Exception(f"Deletion failed: {str(e)}")
        
        return deletion_summary

# Consent management
class ConsentManager:
    def __init__(self, db_session):
        self.db = db_session
    
    def record_consent(self, person_id: int, purpose: str, 
                      consent_method: str = "web_form") -> Dict:
        """Record explicit consent with timestamp and method."""
        consent = Consent(
            person_id=person_id,
            purpose=purpose,
            consent_given=True,
            consent_method=consent_method,
            consent_date=datetime.utcnow(),
            ip_address=request.client.host if 'request' in globals() else None
        )
        
        self.db.add(consent)
        self.db.commit()
        
        return {
            "consent_id": consent.id,
            "person_id": person_id,
            "purpose": purpose,
            "consent_date": consent.consent_date.isoformat()
        }
```

This guide provides comprehensive security and privacy coverage for the face recognition system. Let me know if you need me to continue with more sections or move to the next component!