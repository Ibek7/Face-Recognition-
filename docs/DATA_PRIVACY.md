# Data Privacy & Compliance Guide

## Overview

This document provides comprehensive guidance on data privacy and regulatory compliance for the Face Recognition system. Face recognition technology processes sensitive biometric data and must comply with various privacy regulations worldwide.

## Table of Contents

- [Regulatory Landscape](#regulatory-landscape)
- [GDPR Compliance](#gdpr-compliance)
- [CCPA Compliance](#ccpa-compliance)
- [Biometric Privacy Laws](#biometric-privacy-laws)
- [Data Processing Principles](#data-processing-principles)
- [User Rights](#user-rights)
- [Consent Management](#consent-management)
- [Data Retention & Deletion](#data-retention--deletion)
- [Privacy by Design](#privacy-by-design)
- [Compliance Checklist](#compliance-checklist)

## Regulatory Landscape

### Key Regulations

| Regulation | Jurisdiction | Scope | Key Requirements |
|------------|-------------|-------|------------------|
| **GDPR** | European Union | Personal data of EU residents | Consent, data minimization, right to erasure |
| **CCPA** | California, USA | California residents' data | Right to know, delete, opt-out |
| **BIPA** | Illinois, USA | Biometric data | Written consent, retention policy |
| **LGPD** | Brazil | Personal data of Brazilian residents | Similar to GDPR |
| **PDPA** | Singapore | Personal data | Consent, purpose limitation |
| **PIPEDA** | Canada | Personal information | Consent, accountability |

### Face Recognition Specific Regulations

**European Union:**
- GDPR Article 9: Special category data (biometric data)
- Requires explicit consent
- High standards for data protection

**United States:**
- Illinois BIPA (Biometric Information Privacy Act)
- Texas Capture or Use of Biometric Identifier Act
- Washington State Biometric Privacy Law
- Local ordinances (San Francisco, Portland ban certain uses)

**Asia-Pacific:**
- China's Personal Information Protection Law (PIPL)
- Australia's Privacy Act 1988
- Japan's Act on the Protection of Personal Information (APPI)

## GDPR Compliance

### Legal Basis for Processing

Under GDPR, biometric data is "special category data" requiring one of the following legal bases:

1. **Explicit Consent** (most common for face recognition)
2. Contractual necessity
3. Legal obligation
4. Vital interests
5. Public interest
6. Legitimate interests (limited for biometric data)

### Explicit Consent Implementation

```python
from datetime import datetime
from typing import Optional
from pydantic import BaseModel

class ConsentRecord(BaseModel):
    user_id: str
    purpose: str  # e.g., "face_recognition_access_control"
    timestamp: datetime
    consent_given: bool
    withdrawal_timestamp: Optional[datetime] = None
    ip_address: str
    user_agent: str

class ConsentManager:
    async def obtain_consent(
        self,
        user_id: str,
        purpose: str,
        metadata: dict
    ) -> ConsentRecord:
        """
        Obtain explicit consent for biometric data processing
        
        Consent must be:
        - Freely given
        - Specific
        - Informed
        - Unambiguous
        """
        consent = ConsentRecord(
            user_id=user_id,
            purpose=purpose,
            timestamp=datetime.utcnow(),
            consent_given=True,
            ip_address=metadata['ip_address'],
            user_agent=metadata['user_agent']
        )
        
        await self.store_consent(consent)
        await self.audit_log.log_consent(consent)
        
        return consent
    
    async def withdraw_consent(self, user_id: str, purpose: str):
        """Users can withdraw consent at any time"""
        consent = await self.get_consent(user_id, purpose)
        consent.withdrawal_timestamp = datetime.utcnow()
        
        await self.update_consent(consent)
        await self.delete_user_biometric_data(user_id, purpose)
```

### Data Protection Impact Assessment (DPIA)

**When Required:**
- Large-scale processing of biometric data
- Systematic monitoring
- High risk to individual rights

**DPIA Template:**

```markdown
# Data Protection Impact Assessment

## Project Description
- Purpose: [Face recognition for access control]
- Data processed: [Face images, face embeddings, metadata]
- Scale: [Number of individuals]

## Necessity and Proportionality
- Why is this processing necessary?
- Are there less intrusive alternatives?
- Is the data collection proportionate to the purpose?

## Risk Assessment
| Risk | Likelihood | Severity | Mitigation |
|------|-----------|----------|------------|
| Unauthorized access | Medium | High | Encryption, access control |
| Data breach | Low | Critical | Security measures, monitoring |
| Misidentification | Medium | Medium | Quality thresholds, human review |

## Measures to Address Risks
- Technical measures: [Encryption, pseudonymization]
- Organizational measures: [Access policies, training]
- Safeguards: [Regular audits, incident response]

## Consultation
- Data Protection Officer consulted: [Yes/No]
- Stakeholders consulted: [List]
```

### GDPR Requirements Implementation

**Data Minimization:**

```python
class BiometricDataStorage:
    """
    Store only necessary biometric data
    - Store face embeddings, not raw images (when possible)
    - Pseudonymize identifiers
    - Limit metadata collection
    """
    
    async def store_face_data(self, person_id: str, image: np.ndarray):
        # Extract embedding only, discard image
        embedding = self.face_encoder.encode(image)
        
        # Pseudonymize person ID
        pseudonym = self.pseudonymize(person_id)
        
        # Store minimal data
        await self.db.execute(
            """
            INSERT INTO embeddings (pseudonym, embedding, created_at)
            VALUES (:pseudonym, :embedding, :timestamp)
            """,
            {
                'pseudonym': pseudonym,
                'embedding': embedding.tolist(),
                'timestamp': datetime.utcnow()
            }
        )
        
        # Don't store: age, gender, ethnicity, emotion, etc.
```

**Purpose Limitation:**

```python
class PurposeLimiter:
    ALLOWED_PURPOSES = {
        'access_control': ['building_entry', 'secure_area_access'],
        'time_tracking': ['attendance_recording'],
        'security': ['threat_detection', 'unauthorized_access']
    }
    
    def validate_purpose(self, requested_purpose: str, consent_purpose: str) -> bool:
        """Ensure data is only used for consented purpose"""
        return requested_purpose == consent_purpose
```

**Storage Limitation:**

```python
class DataRetentionPolicy:
    RETENTION_PERIODS = {
        'active_user': 365,      # days
        'inactive_user': 90,
        'deleted_user': 30,      # grace period
        'audit_logs': 2555       # 7 years
    }
    
    async def apply_retention_policy(self):
        """Automatically delete data after retention period"""
        for status, days in self.RETENTION_PERIODS.items():
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            await self.db.execute(
                """
                DELETE FROM embeddings
                WHERE user_status = :status
                AND created_at < :cutoff_date
                """,
                {'status': status, 'cutoff_date': cutoff_date}
            )
```

## CCPA Compliance

### Consumer Rights Under CCPA

1. **Right to Know**
2. **Right to Delete**
3. **Right to Opt-Out**
4. **Right to Non-Discrimination**

### Implementation

```python
class CCPACompliance:
    async def handle_data_request(self, request_type: str, user_id: str):
        """Handle CCPA consumer requests"""
        
        if request_type == 'know':
            return await self.provide_data_disclosure(user_id)
        
        elif request_type == 'delete':
            return await self.delete_consumer_data(user_id)
        
        elif request_type == 'opt_out':
            return await self.opt_out_sale(user_id)
    
    async def provide_data_disclosure(self, user_id: str) -> dict:
        """
        Provide disclosure of:
        - Categories of personal information collected
        - Sources of information
        - Business purpose for collection
        - Third parties with whom data is shared
        """
        return {
            'categories': [
                'Biometric information (face embeddings)',
                'Identifiers (name, ID number)',
                'Metadata (timestamps, locations)'
            ],
            'sources': ['Direct collection from user'],
            'purposes': ['Access control', 'Security'],
            'third_parties': ['None'],
            'data': await self.export_user_data(user_id)
        }
    
    async def delete_consumer_data(self, user_id: str):
        """Delete all consumer data within 45 days"""
        deletion_request = {
            'user_id': user_id,
            'requested_at': datetime.utcnow(),
            'status': 'pending',
            'deadline': datetime.utcnow() + timedelta(days=45)
        }
        
        await self.queue_deletion(deletion_request)
        return {'message': 'Deletion request submitted'}
```

### "Do Not Sell My Personal Information"

```python
class SaleOptOut:
    async def check_opt_out_status(self, user_id: str) -> bool:
        """Check if user has opted out of data sale"""
        result = await self.db.fetch_one(
            "SELECT opt_out FROM users WHERE id = :user_id",
            {'user_id': user_id}
        )
        return result['opt_out'] if result else False
    
    async def respect_opt_out(self, user_id: str, third_party: str):
        """Don't share data if user has opted out"""
        if await self.check_opt_out_status(user_id):
            raise ValueError("User has opted out of data sharing")
```

## Biometric Privacy Laws

### Illinois BIPA Compliance

**Key Requirements:**

1. **Written Policy**
2. **Written Consent**
3. **Retention Schedule**
4. **Data Destruction**

```python
class BIPACompliance:
    RETENTION_POLICY = """
    Biometric Data Retention and Destruction Policy
    
    1. Purpose Limitation
       - Biometric data collected solely for [access control]
       - No secondary use without additional consent
    
    2. Retention Period
       - Active employees: Duration of employment + 30 days
       - Former employees: Immediately upon termination
       - Visitors: 24 hours after visit
    
    3. Destruction
       - Permanent deletion from all systems
       - Deletion of backups within 90 days
       - Certification of destruction provided
    """
    
    async def obtain_bipa_consent(self, user_id: str) -> bool:
        """
        Obtain written consent including:
        - What biometric data is collected
        - Purpose of collection
        - Duration of storage
        """
        consent_text = """
        I consent to the collection and storage of my biometric 
        identifier (facial recognition data) for the purpose of 
        building access control. This data will be stored for the 
        duration of my employment and will be permanently deleted 
        within 30 days of my departure.
        """
        
        # Present to user for signature
        # Store signed consent
        return await self.record_written_consent(user_id, consent_text)
    
    async def destroy_biometric_data(self, user_id: str):
        """Permanent destruction of biometric identifiers"""
        # Delete from primary database
        await self.db.execute(
            "DELETE FROM embeddings WHERE user_id = :user_id",
            {'user_id': user_id}
        )
        
        # Delete from backups
        await self.mark_for_backup_deletion(user_id)
        
        # Log destruction
        await self.audit_log.log_destruction(user_id)
        
        # Issue destruction certificate
        return await self.issue_destruction_certificate(user_id)
```

## Data Processing Principles

### Privacy by Design

**7 Foundational Principles:**

1. **Proactive not Reactive**
2. **Privacy as Default**
3. **Privacy Embedded into Design**
4. **Full Functionality (Positive-Sum)**
5. **End-to-End Security**
6. **Visibility and Transparency**
7. **Respect for User Privacy**

### Implementation

```python
class PrivacyByDesign:
    def __init__(self):
        # 1. Proactive: Security measures before deployment
        self.setup_security()
        
        # 2. Privacy by default: Minimal data collection
        self.config.collect_minimal_data = True
        self.config.store_embeddings_only = True
        
        # 3. Embedded: Privacy in architecture
        self.enable_encryption()
        self.enable_pseudonymization()
        
        # 4. Full functionality: Privacy doesn't reduce functionality
        self.optimize_performance()
        
        # 5. End-to-end security: Lifecycle protection
        self.secure_data_lifecycle()
        
        # 6. Visibility: Transparent processing
        self.enable_audit_logs()
        self.publish_privacy_policy()
        
        # 7. User-centric: Respect user rights
        self.implement_user_controls()
```

### Data Minimization

```python
class MinimalDataCollection:
    """Collect only what's necessary"""
    
    def process_face_recognition(self, image: np.ndarray):
        # ❌ Don't store
        # - Original image
        # - Age, gender, emotion
        # - Location metadata
        # - Device information
        
        # ✅ Store only
        # - Face embedding (mathematical representation)
        # - Timestamp
        # - Pseudonymized ID
        
        embedding = self.extract_embedding(image)
        pseudonym = self.generate_pseudonym()
        
        return {
            'embedding': embedding,
            'timestamp': datetime.utcnow(),
            'pseudonym': pseudonym
        }
```

## User Rights

### Right to Access

```python
class DataAccessRequest:
    async def export_user_data(self, user_id: str) -> dict:
        """Provide all data held about user"""
        return {
            'personal_info': await self.get_personal_info(user_id),
            'biometric_data': {
                'embeddings_count': await self.count_embeddings(user_id),
                'note': 'Embeddings are not human-readable'
            },
            'access_logs': await self.get_access_logs(user_id),
            'consent_records': await self.get_consents(user_id),
            'retention_period': self.get_retention_period(user_id)
        }
```

### Right to Rectification

```python
class DataRectification:
    async def update_user_data(self, user_id: str, updates: dict):
        """Allow users to correct their data"""
        allowed_fields = ['name', 'email', 'phone']
        
        for field, value in updates.items():
            if field in allowed_fields:
                await self.db.execute(
                    f"UPDATE users SET {field} = :value WHERE id = :user_id",
                    {'value': value, 'user_id': user_id}
                )
        
        # For biometric data: re-enrollment
        if 'biometric' in updates:
            await self.re_enroll_user(user_id)
```

### Right to Erasure

```python
class RightToErasure:
    async def delete_user_data(self, user_id: str, reason: str):
        """
        Delete all user data when:
        - User withdraws consent
        - Purpose fulfilled
        - Retention period expired
        - User requests deletion
        """
        async with self.db.transaction():
            # Delete biometric data
            await self.db.execute(
                "DELETE FROM embeddings WHERE user_id = :user_id",
                {'user_id': user_id}
            )
            
            # Delete images
            await self.delete_user_images(user_id)
            
            # Delete personal info
            await self.db.execute(
                "DELETE FROM users WHERE id = :user_id",
                {'user_id': user_id}
            )
            
            # Anonymize logs (keep for audit, remove PII)
            await self.anonymize_user_logs(user_id)
            
            # Log deletion
            await self.audit_log.log_erasure(user_id, reason)
        
        return {'status': 'deleted', 'user_id': user_id}
```

### Right to Data Portability

```python
class DataPortability:
    async def export_in_portable_format(self, user_id: str) -> bytes:
        """Export data in machine-readable format (JSON, CSV)"""
        data = await self.export_user_data(user_id)
        
        # Export as JSON
        return json.dumps(data, indent=2, default=str).encode()
```

## Consent Management

### Consent Requirements

**Valid Consent Must Be:**
- **Freely given**: No coercion
- **Specific**: For specific purpose
- **Informed**: User understands what they consent to
- **Unambiguous**: Clear affirmative action
- **Withdrawable**: Can be withdrawn anytime

### Consent Interface Example

```javascript
// Frontend consent form
const ConsentForm = () => {
  return (
    <div className="consent-form">
      <h2>Biometric Data Consent</h2>
      
      <div className="consent-info">
        <h3>What data do we collect?</h3>
        <p>We will create a mathematical representation (embedding) 
           of your facial features for identity verification.</p>
        
        <h3>How will it be used?</h3>
        <p>Solely for building access control and security purposes.</p>
        
        <h3>How long will it be stored?</h3>
        <p>For the duration of your employment plus 30 days.</p>
        
        <h3>Your rights</h3>
        <ul>
          <li>You can withdraw consent at any time</li>
          <li>You can request deletion of your data</li>
          <li>You can request a copy of your data</li>
        </ul>
      </div>
      
      <label>
        <input type="checkbox" required />
        I consent to the collection and processing of my biometric data 
        as described above. I understand I can withdraw this consent at 
        any time by contacting privacy@example.com.
      </label>
      
      <button>I Consent</button>
      <button>I Do Not Consent</button>
    </div>
  );
};
```

## Compliance Checklist

### Legal & Regulatory

- [ ] Privacy policy published and accessible
- [ ] Terms of service include biometric data processing
- [ ] DPIA completed for high-risk processing
- [ ] Legal basis for processing documented
- [ ] Compliance with local biometric laws verified
- [ ] Data Processing Agreement with processors
- [ ] Cross-border transfer mechanisms (if applicable)

### Consent

- [ ] Explicit consent mechanism implemented
- [ ] Consent is freely given (not bundled)
- [ ] Consent is specific to purpose
- [ ] Consent is informed (clear information provided)
- [ ] Consent is unambiguous (affirmative action)
- [ ] Consent can be withdrawn easily
- [ ] Consent records maintained

### Data Protection

- [ ] Data minimization applied
- [ ] Purpose limitation enforced
- [ ] Storage limitation implemented
- [ ] Encryption at rest enabled
- [ ] Encryption in transit enabled
- [ ] Access controls configured
- [ ] Audit logging enabled
- [ ] Regular security assessments

### User Rights

- [ ] Right to access implemented
- [ ] Right to rectification implemented
- [ ] Right to erasure implemented
- [ ] Right to data portability implemented
- [ ] Right to object implemented
- [ ] User rights request process documented
- [ ] Response within legal timeframes

### Organizational

- [ ] Data Protection Officer appointed (if required)
- [ ] Privacy training for staff
- [ ] Data breach notification procedure
- [ ] Records of processing activities maintained
- [ ] Vendor/processor agreements in place
- [ ] Regular compliance audits
- [ ] Incident response plan

## Contact

For privacy questions or to exercise your rights:
- **Email:** privacy@example.com
- **Data Protection Officer:** dpo@example.com
- **Address:** [Physical address for written requests]

**Response Time:** We will respond to your request within:
- GDPR: 30 days (extendable to 60)
- CCPA: 45 days (extendable to 90)
- BIPA: As soon as reasonably practicable

---

**Last Updated:** January 2025  
**Version:** 1.0  
**Review Schedule:** Quarterly
