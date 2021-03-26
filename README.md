# person-counter

 \+ nice documentation
 
 \+ works with video xor camera stream
 
 \+ customisable magic values
 
 \+ focusuf disables camera autofocus
 
 ## Usage
`detect try 3.py` deals with multiple people, not suitable to large amounts of noise

`detect try 4.py` deals with at most 1 person, suitable to large amounts of noise (using extra restrictions e.g. distance travelled per frame)

## Implementation
subtract background, threshold to BW image

detect significant countours, identify centroids

track between frames: minimum sum of distances between prev and new centroids

(NOTE: standard object trackers don't work well)

