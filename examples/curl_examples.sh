#!/bin/bash
#
# cURL examples for the Face Recognition API
# These examples demonstrate all major API endpoints
#

# Set API base URL
API_URL="${API_URL:-http://localhost:8000}"

echo "=========================================="
echo "Face Recognition API - cURL Examples"
echo "API URL: $API_URL"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 1. Health Check
echo -e "\n${BLUE}1. Health Check${NC}"
curl -s -X GET "$API_URL/health" | jq '.'

# 2. Liveness Probe
echo -e "\n${BLUE}2. Liveness Probe${NC}"
curl -s -X GET "$API_URL/health/live" | jq '.'

# 3. Readiness Probe
echo -e "\n${BLUE}3. Readiness Probe${NC}"
curl -s -X GET "$API_URL/health/ready" | jq '.'

# 4. Get API Version
echo -e "\n${BLUE}4. API Version${NC}"
curl -s -X GET "$API_URL/version" | jq '.'

# 5. Root Endpoint
echo -e "\n${BLUE}5. Root Endpoint (API Info)${NC}"
curl -s -X GET "$API_URL/" | jq '.'

# 6. Create a Person
echo -e "\n${BLUE}6. Create a New Person${NC}"
PERSON_RESPONSE=$(curl -s -X POST "$API_URL/persons" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Jane Smith",
    "description": "Test person created via cURL"
  }')
echo "$PERSON_RESPONSE" | jq '.'
PERSON_ID=$(echo "$PERSON_RESPONSE" | jq -r '.id')
echo -e "${GREEN}Created person with ID: $PERSON_ID${NC}"

# 7. List All Persons
echo -e "\n${BLUE}7. List All Persons${NC}"
curl -s -X GET "$API_URL/persons" | jq '.'

# 8. Upload Face Image for Person
echo -e "\n${BLUE}8. Upload Face Image${NC}"
# Check if test image exists
if [ -f "data/images/test_face.jpg" ]; then
    curl -s -X POST "$API_URL/persons/$PERSON_ID/images" \
      -F "file=@data/images/test_face.jpg" | jq '.'
else
    echo "Test image not found at data/images/test_face.jpg"
    echo "Create a test image or update the path"
fi

# 9. Recognize Faces from File Upload
echo -e "\n${BLUE}9. Recognize Faces (File Upload)${NC}"
if [ -f "data/images/test_face.jpg" ]; then
    curl -s -X POST "$API_URL/upload?threshold=0.7" \
      -F "file=@data/images/test_face.jpg" | jq '.'
else
    echo "Test image not found"
fi

# 10. Recognize Faces from Base64
echo -e "\n${BLUE}10. Recognize Faces (Base64)${NC}"
if [ -f "data/images/test_face.jpg" ]; then
    BASE64_IMAGE=$(base64 -i data/images/test_face.jpg)
    curl -s -X POST "$API_URL/recognize" \
      -H "Content-Type: application/json" \
      -d "{
        \"image_base64\": \"$BASE64_IMAGE\",
        \"threshold\": 0.7,
        \"top_k\": 5
      }" | jq '.success, .faces | length'
else
    echo "Test image not found"
fi

# 11. Get System Statistics
echo -e "\n${BLUE}11. System Statistics${NC}"
curl -s -X GET "$API_URL/stats" | jq '.'

# 12. Get Performance Metrics
echo -e "\n${BLUE}12. Performance Metrics${NC}"
curl -s -X GET "$API_URL/performance/metrics" | jq '.'

# 13. Test Error Handling
echo -e "\n${BLUE}13. Test Error Handling (Invalid Person ID)${NC}"
curl -s -X POST "$API_URL/persons/99999/images" \
  -F "file=@data/images/test_face.jpg" | jq '.'

# 14. Test Invalid Image Upload
echo -e "\n${BLUE}14. Test Invalid File Type${NC}"
echo "This is not an image" > /tmp/test.txt
curl -s -X POST "$API_URL/persons/$PERSON_ID/images" \
  -F "file=@/tmp/test.txt" | jq '.'
rm /tmp/test.txt

# 15. Batch Recognition Example
echo -e "\n${BLUE}15. Batch Recognition (Multiple Images)${NC}"
if [ -d "data/images" ]; then
    for img in data/images/*.jpg; do
        if [ -f "$img" ]; then
            echo "Processing: $img"
            curl -s -X POST "$API_URL/upload?threshold=0.7" \
              -F "file=@$img" | jq -c '{success: .success, faces: (.faces | length)}'
            break  # Process only first image for demo
        fi
    done
else
    echo "Images directory not found"
fi

echo -e "\n=========================================="
echo "Examples completed!"
echo "=========================================="
echo ""
echo "Tip: Install 'jq' for better JSON formatting:"
echo "  brew install jq  # macOS"
echo "  apt-get install jq  # Ubuntu/Debian"
